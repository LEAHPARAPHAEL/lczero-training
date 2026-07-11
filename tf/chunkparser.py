#!/usr/bin/env python3
#
#    This file is part of Leela Chess.
#    Copyright (C) 2018 Folkert Huizinga
#    Copyright (C) 2017-2018 Gian-Carlo Pascutto
#
#    Leela Chess is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Leela Chess is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
"""
General comments on how chunkparser works.

A "training record" or just "record" is a fixed-length packed byte array. Typically
records are generated during training and are stored together by game, one record for 
each position in the game, but this arrangement is not required.
Over dev time additional fields have been added to the training record, most of which 
just put additional information after the end of the byte array used in the previous 
version. Currently supported training record versions are V3, V4, V5, and V6.

shufflebuffer.ShuffleBuffer is a simple structure holding an array of training
records that are efficiently randomized and replaced as needed. All records in
ShuffleBuffer are adjusted to be the same number of bytes by appending unused 
bytes *before* being put in the shuffler. 
byte padding is done in chunkparser.ChunkParser.sample_record()
sample_record() also skips most training records to avoid sampling over-correlated
positions since they typically are from sequential positions in a game.

Current implementation of "diff focus" also is in sample_record() and works by
probabilistically skipping records according to how accurate the no-search 
eval ('orig_q') is compared to eval after search ('best_q') as well as the
recorded policy_kld (a measure of difference between no search policy and the
final policy distribution). It does not use draw values at this point. Putting
diff focus here is efficient because it runs in parallel workers and peeks at
the records without requiring any unpacking.

The constructor for chunkparser.ChunkParser() sets a bunch of class constants
and creates a fixed number of parallel Python multiprocessing.Pipe objects,
which consist of a "reader" and a "writer". The writer(s) get data directly
from training data files and write them into the pipe using the writer.send_bytes()
method. The reader(s) get data out of the pipe using the reader.rev_bytes()
method and feed them to the ShuffleBuffer using its insert_or_replace() method,
which also handles the shuffling itself.

Records come back out of the ShuffleBuffer (already a fixed byte number
regardless of training version) using the multiplexed generators specified in
the ChunkParser.parse() method. They are first recovered as raw byte records
in the vX_gen() method (currently v6_gen), then converted to tuples of more
interpretable data in the convert_vX_to_tuple() method and finally sent on
to tensorflow in training batches by the batch_gen() method.
"""

import itertools
import multiprocessing as mp
import numpy as np
import random
import shufflebuffer as sb
import struct
import unittest
import gzip
from select import select
import os

V6_VERSION = struct.pack('i', 6)
V5_VERSION = struct.pack('i', 5)
CLASSICAL_INPUT = struct.pack('i', 1)
V4_VERSION = struct.pack('i', 4)
V3_VERSION = struct.pack('i', 3)
V7_VERSION = struct.pack('i', 7)
V7B_VERSION = struct.pack('i', 70)
V6_STRUCT_STRING = '4si7432s832sBBBBBBBbfffffffffffffffIHH4H'
V7_STRUCT_STRING = "4si7432s832sBBBBBBBbfffffffffffffffIHHfffHHffffffff"
V7B_STRUCT_STRING = V7_STRUCT_STRING + "7432s7432s"
V5_STRUCT_STRING = '4si7432s832sBBBBBBBbfffffff'
V4_STRUCT_STRING = '4s7432s832sBBBBBBBbffff'
V3_STRUCT_STRING = '4s7432s832sBBBBBBBb'


v7_struct = struct.Struct(V7_STRUCT_STRING)
v7b_struct = struct.Struct(V7B_STRUCT_STRING)
v6_struct = struct.Struct(V6_STRUCT_STRING)
v5_struct = struct.Struct(V5_STRUCT_STRING)
v4_struct = struct.Struct(V4_STRUCT_STRING)
v3_struct = struct.Struct(V3_STRUCT_STRING)

struct_sizes = {V7_VERSION: v7_struct.size,
                V7B_VERSION : v7b_struct.size,
                V6_VERSION: v6_struct.size, V5_VERSION: v5_struct.size,
                V4_VERSION: v4_struct.size, V3_VERSION: v3_struct.size}



def reverse_expand_bits(plane):
    return np.unpackbits(np.array([plane], dtype=np.uint8))[::-1].astype(
        np.float32).tobytes()


# Interface for a chunk data source.
class ChunkDataSrc:
    def __init__(self, items):
        self.items = items

    def next(self):
        if not self.items:
            return None
        return self.items.pop()


def chunk_reader(chunk_filenames, chunk_filename_queue):
    """
    Reads chunk filenames from a list and writes them in shuffled
    order to output_pipes.
    """
    chunks = []
    done = chunk_filenames

    while True:
        if not chunks:
            chunks, done = done, chunks
            random.shuffle(chunks)
        if not chunks:
            print("chunk_reader didn't find any chunks.")
            return None
        while len(chunks):
            filename = chunks.pop()
            done.append(filename)
            chunk_filename_queue.put(filename)
    print("chunk_reader exiting.")
    return None


class ChunkParser:

    def __init__(self,
                 chunks,
                 expected_input_format,
                 shuffle_size=1,
                 sample=1,
                 buffer_size=1,
                 batch_size=256,
                 diff_focus_min=1,
                 diff_focus_slope=0,
                 diff_focus_q_weight=6.0,
                 diff_focus_pol_scale=3.5,
                 workers=None,
                 pc_min = None,
                 pc_max = None):
        self.inner = ChunkParserInner(self, chunks, expected_input_format,
                                      shuffle_size, sample, buffer_size,
                                      batch_size, diff_focus_min,
                                      diff_focus_slope, diff_focus_q_weight,
                                      diff_focus_pol_scale, workers, pc_min, pc_max)

    def shutdown(self):
        """
        Terminates all the workers
        """
        for i in range(len(self.processes)):
            self.processes[i].terminate()
            self.processes[i].join()
            self.inner.readers[i].close()
            self.inner.writers[i].close()
        self.chunk_process.terminate()
        self.chunk_process.join()

    def parse(self):
        return self.inner.parse()

    def sequential(self):
        return self.inner.sequential()


class ChunkParserInner:
    def __init__(self, parent, chunks, expected_input_format, shuffle_size,
                 sample, buffer_size, batch_size, diff_focus_min,
                 diff_focus_slope, diff_focus_q_weight, diff_focus_pol_scale,
                 workers, pc_min, pc_max):
        """
        Read data and yield batches of raw tensors.

        'parent' the outer chunk parser to store processes. Must not be stored by self directly or indirectly.
        'chunks' list of chunk filenames.
        'shuffle_size' is the size of the shuffle buffer.
        'sample' is the rate to down-sample.
        'diff_focus_min', 'diff_focus_slope', 'diff_focus_q_weight' and 'diff_focus_pol_scale' control diff focus
        'workers' is the number of child workers to use.

        The data is represented in a number of formats through this dataflow
        pipeline. In order, they are:

        chunk: The name of a file containing chunkdata

        chunkdata: type Bytes. Multiple records of v6 format where each record
        consists of (state, policy, result, q)

        raw: A byte string holding raw tensors contenated together. This is
        used to pass data from the workers to the parent. Exists because
        TensorFlow doesn't have a fast way to unpack bit vectors. 7950 bytes
        long.
        """

        self.expected_input_format = expected_input_format

        # Build 2 flat float32 planes with values 0,1
        self.flat_planes = []
        for i in range(2):
            self.flat_planes.append(
                (np.zeros(64, dtype=np.float32) + i).tobytes())

        # set the down-sampling rate
        self.sample = sample
        # set the details for diff focus, defaults accept all positions
        self.diff_focus_min = diff_focus_min
        self.diff_focus_slope = diff_focus_slope
        self.diff_focus_q_weight = diff_focus_q_weight
        self.diff_focus_pol_scale = diff_focus_pol_scale
        # set the mini-batch size
        self.batch_size = batch_size
        # set number of elements in the shuffle buffer.
        self.shuffle_size = shuffle_size
        self.pc_min = pc_min
        self.pc_max = pc_max
        # Start worker processes, leave 2 for TensorFlow
        if workers is None:
            workers = max(1, mp.cpu_count() - 2)

        if workers > 0:
            print("Using {} worker processes.".format(workers))

            # Start the child workers running
            self.readers = []
            self.writers = []
            parent.processes = []
            self.chunk_filename_queue = mp.Queue(maxsize=4096)
            for _ in range(workers):
                read, write = mp.Pipe(duplex=False)
                p = mp.Process(target=self.task,
                               args=(self.chunk_filename_queue, write))
                p.daemon = True
                parent.processes.append(p)
                p.start()
                self.readers.append(read)
                self.writers.append(write)

            parent.chunk_process = mp.Process(target=chunk_reader,
                                              args=(chunks,
                                                    self.chunk_filename_queue))
            parent.chunk_process.daemon = True
            parent.chunk_process.start()
        else:
            self.chunks = chunks



    def convert_v7_to_tuple(self, content):
        """
        Unpack a v7 binary record to 8-tuple (state, policy pi, result, root_wdl, plies_left, st_wdl, opp_idx, next_idx)

        v7 struct format is (8396 bytes total):
                                  size         1st byte index
        uint32_t version;                               0
        uint32_t input_format;                          4
        float probabilities[1858];  7432 bytes          8
        uint64_t planes[104];        832 bytes       7440
        ... (skipping standard bytes) ...
        uint16_t opp_played_idx;                     8360
        uint16_t next_played_idx;                    8362
        float extra[8];                              8364
        """
        (ver, input_format, probs, planes, us_ooo, us_oo, them_ooo, them_oo,
         stm, rule50_count, invariance_info, dep_result, root_q, best_q,
         root_d, best_d, root_m, best_m, plies_left, result_q, result_d,
         played_q, played_d, played_m, orig_q, orig_d, orig_m, visits,
         played_idx, best_idx, pol_kld, st_q, st_d, opp_played_idx, next_played_idx,
         f1, f2, f3, f4, f5, f6, f7, f8, opp_probs, next_probs) = v7b_struct.unpack(content)

        if plies_left == 0:
            plies_left = invariance_info
        plies_left = struct.pack('f', plies_left)

        assert input_format == self.expected_input_format

        # Unpack bit planes and cast to 32 bit float
        planes = np.unpackbits(np.frombuffer(planes, dtype=np.uint8)).astype(
            np.float32)
        rule50_divisor = 99.0
        if input_format > 3:
            rule50_divisor = 100.0
        rule50_plane = struct.pack('f', rule50_count / rule50_divisor) * 64

        if input_format == 1:
            middle_planes = self.flat_planes[us_ooo] + \
                            self.flat_planes[us_oo] + \
                            self.flat_planes[them_ooo] + \
                            self.flat_planes[them_oo] + \
                            self.flat_planes[stm]
        elif input_format == 2:
            them_ooo_bytes = reverse_expand_bits(them_ooo)
            us_ooo_bytes = reverse_expand_bits(us_ooo)
            them_oo_bytes = reverse_expand_bits(them_oo)
            us_oo_bytes = reverse_expand_bits(us_oo)
            middle_planes = us_ooo_bytes + (6*8*4) * b'\x00' + them_ooo_bytes + \
                            us_oo_bytes + (6*8*4) * b'\x00' + them_oo_bytes + \
                            self.flat_planes[0] + \
                            self.flat_planes[0] + \
                            self.flat_planes[stm]
        elif input_format in [3, 4, 132, 5, 133]:
            them_ooo_bytes = reverse_expand_bits(them_ooo)
            us_ooo_bytes = reverse_expand_bits(us_ooo)
            them_oo_bytes = reverse_expand_bits(them_oo)
            us_oo_bytes = reverse_expand_bits(us_oo)
            enpassant_bytes = reverse_expand_bits(stm)
            middle_planes = us_ooo_bytes + (6*8*4) * b'\x00' + them_ooo_bytes + \
                            us_oo_bytes + (6*8*4) * b'\x00' + them_oo_bytes + \
                            self.flat_planes[0] + \
                            self.flat_planes[0] + \
                            (7*8*4) * b'\x00' + enpassant_bytes

        aux_plus_6_plane = self.flat_planes[0]
        if (input_format == 132 or input_format == 133) and invariance_info >= 128:
            aux_plus_6_plane = self.flat_planes[1]
            
        planes = planes.tobytes() + \
                 middle_planes + \
                 rule50_plane + \
                 aux_plus_6_plane + \
                 self.flat_planes[1]

        assert len(planes) == ((8 * 13 * 1 + 8 * 1 * 1) * 8 * 8 * 4)

        if ver in [V6_VERSION, V7_VERSION]:
            winner = struct.pack('fff', 0.5 * (1.0 - result_d + result_q),
                                 result_d, 0.5 * (1.0 - result_d - result_q))
        else:
            dep_result = float(dep_result)
            winner = struct.pack('fff', dep_result == 1.0, dep_result == 0.0, dep_result == -1.0)

        def clip(x, lo, hi):
            return min(max(x, lo), hi)

        def qd_to_wdl(q, d):
            q = clip(q, -1.0, 1.0)
            d = clip(d, 0.0, 1.0)
            w = 0.5 * (1.0 - d + q)
            l = 0.5 * (1.0 - d - q)
            return (w, d, l)

        root_wdl = struct.pack('fff', *(qd_to_wdl(root_q, root_d)))
        
        st_wdl = struct.pack('fff', *(qd_to_wdl(st_q, st_d)))

        # Pack the 16-bit opponent/next moves as 32-bit (i) integers so they align 
        # with standard TensorFlow/NumPy 32-bit tensor arrays in batch_gen
        opp_played_idx = struct.pack('i', opp_played_idx)
        next_played_idx = struct.pack('i', next_played_idx)

        return (planes, probs, winner, root_wdl, plies_left, st_wdl, opp_probs, next_probs)

    def sample_record(self, chunkdata):
        """
        Randomly sample through the chunk data.
        Downsampling to avoid highly correlated positions skips most records, and 
        diff focus may also skip some records.
        """
        version = chunkdata[0:4]
        assert (version == V7_VERSION)
        record_size = v7_struct.size
        total_records = len(chunkdata) // record_size

        # 1. Gather all probabilities upfront (Unchanged)
        probs = [chunkdata[i + 8:i + 8 + 1858 * 4] for i in range(0, len(chunkdata), record_size)]
        dummy_prob = struct.pack("f", 1.0) + struct.pack("f", -1.0) * 1857
        probs.extend(2 * [dummy_prob])

        all_plies = [struct.unpack("f", chunkdata[i + 8304 : i + 8308])[0] for i in range(0, len(chunkdata), record_size)]
        all_plies.extend([0.0, 0.0])  # Safe buffer to prevent IndexOutOfBounds lookups

        for i in range(0, len(chunkdata), record_size):
            if self.sample > 1:
                # Downsample, using only 1/Nth of the items.
                if random.randint(0, self.sample - 1) != 0:
                    continue  # Skip this record.
            
            idx = i//record_size
            record = chunkdata[i:i + record_size]

            # diff focus code, peek at best_q, orig_q and pol_kld from record (unpacks as tuple with one item)
            best_q = struct.unpack("f", record[8284:8288])[0]
            orig_q = struct.unpack("f", record[8328:8332])[0]
            pol_kld = struct.unpack("f", record[8348:8352])[0]

            # if orig_q is NaN or pol_kld is 0, accept, else accept based on diff focus
            root_q = struct.unpack("f", record[8280:8284])[0]
            root_d = struct.unpack("f", record[8288:8292])[0]
            plies_left = struct.unpack("f", record[8304:8308])[0]
            st_q = struct.unpack("f", record[8352:8356])[0]
            st_d = struct.unpack("f", record[8356:8360])[0]

            # If any of these training targets are corrupted, drop the position instantly
            if (np.isnan(plies_left) or 
                np.isnan(st_q) or np.isnan(st_d) or 
                np.isnan(root_q) or np.isnan(root_d)):
                print(f"🚨 DEBUG: Skipped a record due to NaN! plies={plies_left}, st_q={st_q}", flush=True)
                continue

            try:
                if self.pc_min is not None or self.pc_max is not None:
                    planes = record[7440: 7440+104]
                    planes = np.unpackbits(np.frombuffer(planes, dtype=np.uint8)).astype(np.uint8)
                    planes = np.reshape(planes, [13, 64])
                    # pieces are listed our PNBRQKpnbrqk
                    pc = np.sum(planes[1:5, :]) + np.sum(planes[7:11, :])
                    if self.pc_min is not None and pc < self.pc_min:
                        continue
                    if self.pc_max is not None and pc > self.pc_max:
                        continue
            except Exception as e:
                print(e)

            if not np.isnan(orig_q) and pol_kld > 0:
                diff_q = abs(best_q - orig_q)
                q_weight = self.diff_focus_q_weight
                pol_scale = self.diff_focus_pol_scale
                total = (q_weight * diff_q + pol_kld) / (q_weight +
                                                            pol_scale)
                thresh_p = self.diff_focus_min + self.diff_focus_slope * total
                if thresh_p < 1.0 and random.random() > thresh_p:
                    continue

            current_ply = all_plies[idx]
            next_ply = all_plies[idx + 1]
            next_next_ply = all_plies[idx + 2]

            # Case A: Next position belongs to a new game (or EOF reached) -> Pad both slots completely
            if idx + 1 >= total_records or abs((current_ply - next_ply) - 1.0) > 0.01:
                record += dummy_prob + dummy_prob
            
            # Case B: Next position is valid, but the second one (+2) breaks continuity -> Pad only slot 2
            elif idx + 2 >= total_records or abs((next_ply - next_next_ply) - 1.0) > 0.01:
                record += probs[idx + 1] + dummy_prob
            
            # Case C: Safe sequence continuation -> Peek look-ahead normally
            else:
                record += b"".join(probs[idx + 1 : idx + 3])

            yield record

    def single_file_gen(self, filename):
        with gzip.open(filename, "rb") as chunk_file:
            version = chunk_file.read(4)
            chunk_file.seek(0)
            if version == b'':
                return
            record_size = struct_sizes.get(version, None)
            if record_size is None:
                print("Unknown version {} in file {}".format(
                    version, filename))
                return
            chunkdata = chunk_file.read()
            for item in self.sample_record(chunkdata):
                yield item

    def sequential_gen(self):
        for filename in self.chunks:
            for item in self.single_file_gen(filename):
                yield item

    def task(self, chunk_filename_queue, writer):
        """
        Run in fork'ed process, read data from chunkdatasrc, parsing, shuffling and
        sending v6 data through pipe back to main process.
        """
        while True:
            filename = chunk_filename_queue.get()
            for item in self.single_file_gen(filename):
                writer.send_bytes(item)



    def v7_gen(self):
        """
        Read v7 records from child workers, shuffle, and yield records.
        """
        sbuff = sb.ShuffleBuffer(v7b_struct.size, self.shuffle_size)
        while len(self.readers):
            for r in self.readers:
                try:
                    s = r.recv_bytes()
                    s = sbuff.insert_or_replace(s)
                    if s is None:
                        continue  # shuffle buffer not yet full
                    yield s
                except EOFError:
                    print("Reader EOF")
                    self.readers.remove(r)
        # drain the shuffle buffer.
        while True:
            s = sbuff.extract()
            if s is None:
                return
            yield s

    '''
    def tuple_gen(self, gen):
        """
        Take a generator producing v6 records and convert them to tuples.
        applying a random symmetry on the way.
        """
        for r in gen:
            yield self.convert_v6_to_tuple(r)

    def batch_gen(self, gen, allow_partial=True):
        """
        Pack multiple records into a single batch
        """
        # Get N records. We flatten the returned generator to
        # a list because we need to reuse it.
        while True:
            s = list(itertools.islice(gen, self.batch_size))
            if not len(s) or (not allow_partial and len(s) != self.batch_size):
                return
            yield (b''.join([x[0] for x in s]), b''.join([x[1] for x in s]),
                   b''.join([x[2] for x in s]), b''.join([x[3] for x in s]),
                   b''.join([x[4] for x in s]))

    def parse(self):
        """
        Read data from child workers and yield batches of unpacked records
        """
        gen = self.v6_gen()  # read from workers
        gen = self.tuple_gen(gen)  # convert v6->tuple
        gen = self.batch_gen(gen)  # assemble into batches
        for b in gen:
            yield b

    def sequential(self):
        gen = self.sequential_gen()  # read from all files in order in this process.
        gen = self.tuple_gen(gen)  # convert v6->tuple
        gen = self.batch_gen(gen, allow_partial=False)  # assemble into batches
        for b in gen:
            yield b
    '''

    def tuple_gen(self, gen):
        """
        Take a generator producing v7 records and convert them to tuples.
        """
        for r in gen:
            yield self.convert_v7_to_tuple(r)

    def batch_gen(self, gen, allow_partial=True):
        """
        Pack multiple records into a single batch dynamically based on tuple size.
        """
        while True:
            s = list(itertools.islice(gen, self.batch_size))
            if not len(s) or (not allow_partial and len(s) != self.batch_size):
                return
            # Dynamically handle all 8 outputs from convert_v7_to_tuple
            n_entries = len(s[0])
            yield tuple([b''.join([x[i] for x in s]) for i in range(n_entries)])

    def parse(self):
        """
        Read data from child workers and yield batches of unpacked records
        """
        gen = self.v7_gen()  # read from workers (V7)
        gen = self.tuple_gen(gen)  # convert v7->tuple
        gen = self.batch_gen(gen)  # assemble into batches
        for b in gen:
            yield b

    def sequential(self):
        gen = self.sequential_gen()  # read from all files in order in this process.
        gen = self.tuple_gen(gen)  # convert v7->tuple
        gen = self.batch_gen(gen, allow_partial=False)  # assemble into batches
        for b in gen:
            yield b




def apply_alpha(qs, alpha, alt_signs=True):
    if not isinstance(qs, np.ndarray):
        qs = np.array(qs)

    n = len(qs)
    signs = (-1)**np.arange(n) if alt_signs else 1
    qs = qs * signs
    # Create an array with alpha^(i-j) at (i, j) if this is at most 1 and 0 otherwise.
    q_st = np.zeros(n)
    val = 0
    for i in range(n):
        if i == 0:
            val = qs[-1]
        else:
            val = alpha * val + qs[-i-1] * (1 - alpha)
        q_st[-i-1] = val

    q_st = q_st * signs

    return q_st

import glob
import os
import struct
import gzip
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Set up HPC-friendly Logging ---
logger = logging.getLogger(__name__)
# If you haven't configured the root logger elsewhere in your main script, 
# you can uncomment the basicConfig below:
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def check_v7_file(filename):
    v7_struct = struct.Struct(V7_STRUCT_STRING)
    record_size = v7_struct.size
    with gzip.open(filename, "rb") as chunk_file:
        chunk_file.seek(0)
        chunkdata = chunk_file.read()
        if len(chunkdata) == 0:
            return
        version = chunkdata[0:4]
        assert version == V7_VERSION
        assert len(chunkdata) % v7_struct.size == 0
        n_chunks = len(chunkdata) // v7_struct.size

        for i in range(n_chunks):
            chunk = chunkdata[i*record_size:(i+1)*record_size]
            st_q = struct.unpack("f", chunk[8352:8356])[0]
            st_d = struct.unpack("f", chunk[8356:8360])[0]

            opp_play = struct.unpack("H", chunk[8360:8362])[0]
            my_next_play = struct.unpack("H", chunk[8362:8364])[0]

            logger.info(f"st_q: {st_q}, st_d: {st_d}, opp_play: {opp_play}, my_next_play: {my_next_play}")


'''
def rescore_file(filename, st_alpha=1-1/6, lt_alpha=1-1/24):
    v6_struct = struct.Struct(V6_STRUCT_STRING)
    v7_struct = struct.Struct(V7_STRUCT_STRING)
    record_size = v6_struct.size

    try:
        # STEP 1: Ultra-fast 4-byte peek (Matches your check_data script)
        with gzip.open(filename, "rb") as chunk_file:
            version = chunk_file.read(4)
            if version == V7_VERSION:
                return 'already_v7'
            if version != V6_VERSION:
                return 'unknown_version'
            
            # STEP 2: Only read the rest if it's confirmed V6
            chunk_file.seek(0)
            chunkdata = chunk_file.read()
            
        if len(chunkdata) == 0:
            return 'empty'

        n_chunks = len(chunkdata) // record_size
        qs, ds, play_idx = [], [], []
        
        for i in range(n_chunks):
            qs.append(struct.unpack("f", chunkdata[i*record_size+8280:i*record_size+8284])[0])
            ds.append(struct.unpack("f", chunkdata[i*record_size+8288:i*record_size+8292])[0])
            play_idx.append(chunkdata[i*record_size+8344:i*record_size+8346])
        play_idx += [struct.pack("H", 65535)] * 2

        st_q = apply_alpha(qs, st_alpha)
        st_d = apply_alpha(ds, st_alpha, alt_signs=False)
        cd_array = b""
        
        for i in range(n_chunks):
            new_chunk = bytearray(chunkdata[i*record_size:(i+1)*record_size] + b"\x00" * (v7_struct.size - record_size))
            new_chunk[8352:8356] = struct.pack("f", st_q[i])
            new_chunk[8356:8360] = struct.pack("f", max(st_d[i], 0))
            new_chunk[0:4] = V7_VERSION
            new_chunk[8360:8362] = play_idx[i+1]
            new_chunk[8362:8364] = play_idx[i+2]
            assert len(new_chunk) == v7_struct.size
            cd_array += new_chunk

    except Exception as e:
        print(f"ERROR: Could not read {filename}, got {e}", file=sys.stderr, flush=True)
        return 'failed'
        
    if cd_array == bytearray():
        return 'failed'
        
    # Atomic write protection preserved
    tmp_filename = filename + ".tmp"
    with gzip.open(tmp_filename, 'wb') as chunk_file:
        chunk_file.write(bytes(cd_array))
    os.replace(tmp_filename, filename)
    return 'rescored'

def rescore_files(filenames, **kwargs):
    """Worker function: Processes a chunk of files and returns status tuples."""
    results = []
    for filename in filenames:
        status = rescore_file(filename, **kwargs)
        results.append((filename, status))
    return results


def rescore(filenames, n_workers=16, n_jobs=1000, manifest_path="processed_chunks.txt", **kwargs):
    if isinstance(filenames, str):
        if not filenames.endswith(".gz"):
            filenames = filenames + "/*.gz"
        filenames = glob.glob(filenames)

    total_files = len(filenames)
    if total_files == 0:
        print("No files found to rescore.", flush=True)
        return

    actual_jobs = min(n_jobs, total_files)
    if actual_jobs == 0:
        actual_jobs = 1

    print(f"Rescoring {total_files} files with {n_workers} workers partitioned into {actual_jobs} batches...", flush=True)

    futures = []
    
    with open(manifest_path, "a", buffering=1) as manifest_file:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for n in range(0, actual_jobs):
                lo = n * total_files // actual_jobs
                hi = min((n + 1) * total_files // actual_jobs, total_files)
                
                if lo < hi:
                    chunk = filenames[lo:hi]
                    futures.append(executor.submit(rescore_files, chunk, **kwargs))

            completed_files = 0
            completed_jobs = 0
            
            # Use standard text tracking instead of progress bars
            for future in as_completed(futures):
                try:
                    job_results = future.result()
                    for filename, status in job_results:
                        completed_files += 1
                        if status in ['rescored', 'already_v7']:
                            manifest_file.write(filename + "\n")
                    
                    # Force data synchronization onto disk
                    manifest_file.flush()
                    
                    completed_jobs += 1
                    # Progress track printout every batch collection
                    print(f"[PROGRESS] Completed batch {completed_jobs}/{actual_jobs} | Total files processed: {completed_files}/{total_files}", flush=True)
                        
                except Exception as e:
                    print(f"BATCH ERROR: A multiprocessing batch failed with error: {e}", file=sys.stderr, flush=True)
'''

def rescore_file(filename, st_alpha=1-1/6, lt_alpha=1-1/24):
    v6_struct = struct.Struct(V6_STRUCT_STRING)
    v7_struct = struct.Struct(V7_STRUCT_STRING)
    record_size = v6_struct.size
    cd_array = bytearray()

    try:
        # STEP 1: Fast 4-byte peek to check version instantly
        with gzip.open(filename, "rb") as chunk_file:
            version = chunk_file.read(4)
            if version == V7_VERSION:
                return 'already_v7'
            if version != V6_VERSION:
                return 'unknown_version'
            
            # STEP 2: Only decompress the rest if it's verified V6
            chunk_file.seek(0)
            chunkdata = chunk_file.read()
            
        if len(chunkdata) == 0:
            return 'empty'

        n_chunks = len(chunkdata) // record_size
        qs = []
        ds = []
        play_idx = []
        for i in range(n_chunks):
            qs.append(struct.unpack(
                "f", chunkdata[i*record_size+8280:i*record_size+8284])[0])
            ds.append(struct.unpack(
                "f", chunkdata[i*record_size+8288:i*record_size+8292])[0])
            play_idx.append(
                chunkdata[i*record_size+8344:i*record_size+8346])
        play_idx += [struct.pack("H", 65535)] * 2

        st_q = apply_alpha(qs, st_alpha)
        st_d = apply_alpha(ds, st_alpha, alt_signs=False)
        cd_array = b""
        for i in range(n_chunks):
            new_chunk = bytearray(
                chunkdata[i*record_size:(i+1)*record_size] + b"\x00" * (v7_struct.size - record_size))
            new_chunk[8352:8356] = struct.pack("f", st_q[i])
            new_chunk[8356:8360] = struct.pack("f", max(st_d[i], 0))
            new_chunk[0:4] = V7_VERSION
            new_chunk[8360:8362] = play_idx[i+1]
            new_chunk[8362:8364] = play_idx[i+2]
            assert len(new_chunk) == v7_struct.size
            cd_array += new_chunk

    except Exception as e:
        print(f"ERROR: Could not read {filename}, got {e}", file=sys.stderr, flush=True)
        return 'failed'
        
    if cd_array == bytearray():
        return 'failed'
        
    # STEP 3: Atomic write protection against mid-write job crashes
    tmp_filename = filename + ".tmp"
    with gzip.open(tmp_filename, 'wb') as chunk_file:
        chunk_file.write(bytes(cd_array))
    os.replace(tmp_filename, filename)
    return 'rescored'


def rescore_batch(filenames, **kwargs):
    """Worker function: Processes a micro-batch and returns status tuples to Master."""
    results = []
    for filename in filenames:
        status = rescore_file(filename, **kwargs)
        results.append((filename, status))
    return results


def rescore(filenames, n_workers=16, micro_batch_size=64, manifest_path="processed_chunks.txt", **kwargs):
    if isinstance(filenames, str):
        if not filenames.endswith(".gz"):
            filenames = filenames + "/*.gz"
        filenames = glob.glob(filenames)

    total_files = len(filenames)
    if total_files == 0:
        print("No files assigned to this run phase.", flush=True)
        return

    print(f"Rescoring {total_files} filtered files via dynamic micro-batches...", flush=True)

    def get_micro_batches(iterable, size):
        it = iter(iterable)
        while True:
            batch = list(itertools.islice(it, size))
            if not batch:
                break
            yield batch

    futures = []
    total_rescored = 0
    total_already_v7 = 0
    total_failed = 0
    
    # Open manifest in append-mode with line buffering (buffering=1)
    with open(manifest_path, "a", buffering=1) as manifest_file:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for batch in get_micro_batches(filenames, micro_batch_size):
                futures.append(executor.submit(rescore_batch, batch, **kwargs))

            total_batches = len(futures)
            completed_batches = 0

            for future in as_completed(futures):
                try:
                    job_results = future.result()
                    for filename, status in job_results:
                        if status == 'rescored':
                            total_rescored += 1
                            manifest_file.write(filename + "\n")
                        elif status == 'already_v7':
                            total_already_v7 += 1
                            manifest_file.write(filename + "\n")
                        else:
                            total_failed += 1
                    
                    # Force manifest text to commit to disk immediately
                    manifest_file.flush()
                    
                    completed_batches += 1
                    total_processed = total_rescored + total_already_v7 + total_failed
                    
                    print(f"[PROGRESS] Batch {completed_batches}/{total_batches} finished | "
                          f"Processed: {total_processed}/{total_files} | "
                          f"Newly Rescored: {total_rescored} | "
                          f"Confirmed V7: {total_already_v7} | "
                          f"Failures: {total_failed}", flush=True)
                            
                except Exception as e:
                    print(f"BATCH CRITICAL ERROR: An entire execution chunk failed: {e}", file=sys.stderr, flush=True)