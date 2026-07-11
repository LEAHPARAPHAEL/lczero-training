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

import itertools
import multiprocessing as mp
import numpy as np
import random
import shufflebuffer as sb
import struct
import unittest
import gzip
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

v7_struct = struct.Struct(V7_STRUCT_STRING)
v7b_struct = struct.Struct(V7B_STRUCT_STRING)

struct_sizes = {V7_VERSION: v7_struct.size, V6_VERSION: 8356}

def reverse_expand_bits(plane):
    return np.unpackbits(np.array([plane], dtype=np.uint8))[::-1].astype(
        np.float32).tobytes()

class ChunkDataSrc:
    def __init__(self, items):
        self.items = items

    def next(self):
        if not self.items:
            return None
        return self.items.pop()

def chunk_reader(chunk_filenames, chunk_filename_queue):
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
    return None


class ChunkParser:
    def __init__(self, chunks, expected_input_format, shuffle_size=1, sample=1,
                 buffer_size=1, batch_size=256, diff_focus_min=1, diff_focus_slope=0,
                 diff_focus_q_weight=6.0, diff_focus_pol_scale=3.5, workers=None,
                 pc_min=None, pc_max=None):
        self.inner = ChunkParserInner(self, chunks, expected_input_format,
                                      shuffle_size, sample, buffer_size,
                                      batch_size, diff_focus_min,
                                      diff_focus_slope, diff_focus_q_weight,
                                      diff_focus_pol_scale, workers, pc_min, pc_max)

    def shutdown(self):
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
        
        self.expected_input_format = expected_input_format
        self.flat_planes = [np.zeros(64, dtype=np.float32).tobytes(), np.ones(64, dtype=np.float32).tobytes()]

        self.sample = sample
        self.diff_focus_min = diff_focus_min
        self.diff_focus_slope = diff_focus_slope
        self.diff_focus_q_weight = diff_focus_q_weight
        self.diff_focus_pol_scale = diff_focus_pol_scale
        self.batch_size = batch_size
        self.shuffle_size = shuffle_size
        self.pc_min = pc_min
        self.pc_max = pc_max
        
        if workers is None:
            workers = max(1, mp.cpu_count() - 2)

        if workers > 0:
            print("Using {} worker processes (HPC Block-Stream Fork Mode).".format(workers))
            self.readers = []
            self.writers = []
            parent.processes = []
            self.chunk_filename_queue = mp.Queue(maxsize=4096)
            
            for _ in range(workers):
                read, write = mp.Pipe(duplex=False)
                p = mp.Process(target=self.task, args=(self.chunk_filename_queue, write))
                p.daemon = True
                parent.processes.append(p)
                p.start()
                self.readers.append(read)
                self.writers.append(write)

            parent.chunk_process = mp.Process(target=chunk_reader, args=(chunks, self.chunk_filename_queue))
            parent.chunk_process.daemon = True
            parent.chunk_process.start()
        else:
            self.chunks = chunks

    def convert_v7_to_tuple(self, content):
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
        planes = np.unpackbits(np.frombuffer(planes, dtype=np.uint8)).astype(np.float32)
        
        rule50_divisor = 99.0 if input_format <= 3 else 100.0
        rule50_plane = struct.pack('f', rule50_count / rule50_divisor) * 64

        if input_format == 1:
            middle_planes = self.flat_planes[us_ooo] + self.flat_planes[us_oo] + self.flat_planes[them_ooo] + self.flat_planes[them_oo] + self.flat_planes[stm]
        elif input_format == 2:
            middle_planes = reverse_expand_bits(us_ooo) + (24)*b'\x00' + reverse_expand_bits(them_ooo) + reverse_expand_bits(us_oo) + (24)*b'\x00' + reverse_expand_bits(them_oo) + self.flat_planes[0] + self.flat_planes[0] + self.flat_planes[stm]
        elif input_format in [3, 4, 132, 5, 133]:
            middle_planes = reverse_expand_bits(us_ooo) + (24)*b'\x00' + reverse_expand_bits(them_ooo) + reverse_expand_bits(us_oo) + (24)*b'\x00' + reverse_expand_bits(them_oo) + self.flat_planes[0] + self.flat_planes[0] + (28)*b'\x00' + reverse_expand_bits(stm)

        aux_plus_6_plane = self.flat_planes[1] if (input_format in [132, 133] and invariance_info >= 128) else self.flat_planes[0]
        planes = planes.tobytes() + middle_planes + rule50_plane + aux_plus_6_plane + self.flat_planes[1]

        if ver in [V6_VERSION, V7_VERSION]:
            winner = struct.pack('fff', 0.5 * (1.0 - result_d + result_q), result_d, 0.5 * (1.0 - result_d - result_q))
        
        def qd_to_wdl(q, d):
            q = min(max(q, -1.0), 1.0)
            d = min(max(d, 0.0), 1.0)
            return (0.5 * (1.0 - d + q), d, 0.5 * (1.0 - d - q))

        root_wdl = struct.pack('fff', *(qd_to_wdl(root_q, root_d)))
        st_wdl = struct.pack('fff', *(qd_to_wdl(st_q, st_d)))
        return (planes, probs, winner, root_wdl, plies_left, st_wdl, opp_probs, next_probs)

    def single_file_gen(self, filename):
        """
        🌟 LECTURE PAR MICRO-BATCHES (FLUX CONTINU)
        Sature la bande passante sans charger le fichier master entier en RAM.
        Compatible nativement avec les fichiers géants packagés ou les petits fichiers unitaires.
        """
        record_size = 8396
        dummy_prob = struct.pack("f", 1.0) + struct.pack("f", -1.0) * 1857
        BLOCK_RECORDS = 4000 
        BLOCK_SIZE = BLOCK_RECORDS * record_size

        try:
            with gzip.open(filename, "rb") as chunk_file:
                version = chunk_file.read(4)
                if version != V7_VERSION:
                    return
                chunk_file.seek(0)
                
                carry_over = b""
                while True:
                    raw_block = chunk_file.read(BLOCK_SIZE)
                    if not raw_block:
                        break
                        
                    block_data = carry_over + raw_block
                    n_block_records = len(block_data) // record_size
                    if n_block_records == 0:
                        break
                        
                    probs = [block_data[j + 8 : j + 8 + 1858 * 4] for j in range(0, n_block_records * record_size, record_size)]
                    probs.extend(2 * [dummy_prob])

                    all_plies = [struct.unpack("f", block_data[j + 8304 : j + 8308])[0] for j in range(0, n_block_records * record_size, record_size)]
                    all_plies.extend([0.0, 0.0])
                    
                    is_eof = len(raw_block) < BLOCK_SIZE
                    end_range = n_block_records if is_eof else (n_block_records - 2)
                    
                    for idx in range(end_range):
                        if self.sample > 1 and random.randint(0, self.sample - 1) != 0:
                            continue

                        offset = idx * record_size
                        record = block_data[offset : offset + record_size]
                        
                        # Déballage des targets de validation
                        root_q = struct.unpack("f", record[8280:8284])[0]
                        root_d = struct.unpack("f", record[8288:8292])[0]
                        plies_left = all_plies[idx]
                        st_q = struct.unpack("f", record[8352:8356])[0]
                        st_d = struct.unpack("f", record[8356:8360])[0]

                        if (np.isnan(plies_left) or np.isnan(st_q) or np.isnan(st_d) or np.isnan(root_q) or np.isnan(root_d)):
                            continue

                        # Filtre Pièces
                        if self.pc_min is not None or self.pc_max is not None:
                            planes = record[7440 : 7440 + 104]
                            planes = np.unpackbits(np.frombuffer(planes, dtype=np.uint8)).astype(np.uint8)
                            planes = np.reshape(planes, [13, 64])
                            pc = np.sum(planes[1:5, :]) + np.sum(planes[7:11, :])
                            if self.pc_min is not None and pc < self.pc_min:
                                continue
                            if self.pc_max is not None and pc > self.pc_max:
                                continue

                        # Filtre Diff Focus
                        if self.diff_focus_slope != 0 or self.diff_focus_min < 1.0:
                            best_q = struct.unpack("f", record[8284:8288])[0]
                            orig_q = struct.unpack("f", record[8328:8332])[0]
                            pol_kld = struct.unpack("f", record[8348:8352])[0]
                            if not np.isnan(orig_q) and pol_kld > 0:
                                diff_q = abs(best_q - orig_q)
                                total = (self.diff_focus_q_weight * diff_q + pol_kld) / (self.diff_focus_q_weight + self.diff_focus_pol_scale)
                                thresh_p = self.diff_focus_min + self.diff_focus_slope * total
                                if thresh_p < 1.0 and random.random() > thresh_p:
                                    continue

                        # Sécurisation des Look-Aheads aux frontières (Détecteur d'écart de pli)
                        current_ply = plies_left
                        next_ply = all_plies[idx + 1]
                        next_next_ply = all_plies[idx + 2]

                        if idx + 1 >= n_block_records or abs((current_ply - next_ply) - 1.0) > 0.01:
                            record += dummy_prob + dummy_prob
                        elif idx + 2 >= n_block_records or abs((next_ply - next_next_ply) - 1.0) > 0.01:
                            record += probs[idx + 1] + dummy_prob
                        else:
                            record += b"".join(probs[idx + 1 : idx + 3])

                        yield record
                        
                    if not is_eof:
                        carry_over = block_data[(n_block_records - 2) * record_size : n_block_records * record_size]
                    else:
                        break
        except Exception:
            return

    def sequential_gen(self):
        for filename in self.chunks:
            for item in self.single_file_gen(filename):
                yield item

    def task(self, chunk_filename_queue, writer):
        while True:
            filename = chunk_filename_queue.get()
            for item in self.single_file_gen(filename):
                writer.send_bytes(item)

    def v7_gen(self):
        sbuff = sb.ShuffleBuffer(v7b_struct.size, self.shuffle_size)
        while len(self.readers):
            for r in self.readers:
                try:
                    s = r.recv_bytes()
                    s = sbuff.insert_or_replace(s)
                    if s is None:
                        continue  # Le ShuffleBuffer n'est pas encore plein
                    yield s
                except EOFError:
                    print("Reader EOF")
                    self.readers.remove(r)
                    
        # Vidage final du ShuffleBuffer
        while True:
            s = sbuff.extract()
            if s is None:
                return
            yield s

    def tuple_gen(self, gen):
        for r in gen:
            yield self.convert_v7_to_tuple(r)

    def batch_gen(self, gen, allow_partial=True):
        while True:
            s = list(itertools.islice(gen, self.batch_size))
            if not len(s) or (not allow_partial and len(s) != self.batch_size):
                return
            n_entries = len(s[0])
            yield tuple([b''.join([x[i] for x in s]) for i in range(n_entries)])

    def parse(self):
        if hasattr(self, 'readers') and self.readers:
            gen = self.v7_gen()
        else:
            gen = self.sequential_gen()
        gen = self.tuple_gen(gen)
        gen = self.batch_gen(gen)
        for b in gen:
            yield b

    def sequential(self):
        gen = self.sequential_gen()
        gen = self.tuple_gen(gen)
        gen = self.batch_gen(gen, allow_partial=False)
        for b in gen:
            yield b