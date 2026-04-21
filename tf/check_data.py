import struct
import math
import gzip
from pathlib import Path

# V6 Constants based on chunkparser.py
V6_RECORD_SIZE = 8356
V6_VERSION = struct.pack('i', 6)

def scan_for_nans(directory: str):
    path = Path(directory)
    # Grab all files, ignoring directories
    files = [f for f in path.rglob('*') if f.is_file()]

    total_records = 0
    corrupted_records = 0  # NaN in best_q or best_d (Fatal)
    cache_miss_records = 0 # NaN in orig_q (Expected)

    print(f"\nScanning {len(files):,} files in {directory}...")

    for i, file_path in enumerate(files, 1):
        try:
            with gzip.open(file_path, 'rb') as f:
                # Verify it's a V6 file before processing
                version = f.read(4)
                if version != V6_VERSION:
                    continue
                
                f.seek(0)
                while True:
                    # Read in fast batches of 256 records, just like chunkparser
                    chunkdata = f.read(256 * V6_RECORD_SIZE)
                    if not chunkdata:
                        break

                    for j in range(0, len(chunkdata), V6_RECORD_SIZE):
                        record = chunkdata[j:j + V6_RECORD_SIZE]
                        
                        # Break if we hit a partial record at EOF
                        if len(record) != V6_RECORD_SIZE:
                            break 

                        total_records += 1

                        # Slice out exactly the floats we care about
                        best_q = struct.unpack('f', record[8284:8288])[0]
                        best_d = struct.unpack('f', record[8292:8296])[0]
                        orig_q = struct.unpack('f', record[8328:8332])[0]

                        if math.isnan(best_q) or math.isnan(best_d):
                            corrupted_records += 1
                        
                        if math.isnan(orig_q):
                            cache_miss_records += 1

        except Exception as e:
            # Catch bad gzip files or unreadable chunks without crashing the loop
            print(f"  [!] Error reading {file_path.name}: {e}")
            
        # Progress update
        if i % 10000 == 0:
            print(f"  Processed {i:,} / {len(files):,} files...")

    print(f"\n=== Results for {directory} ===")
    print(f"  Total V6 Records: {total_records:,}")
    print(f"  Critical Corruptions (best_q/d NaN): {corrupted_records:,}")
    print(f"  Normal Cache Misses (orig_q NaN):    {cache_miss_records:,}")
    print("=====================================")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    TRAIN_DIR = "/home/raph/leela/data/train/"
    TEST_DIR  = "/home/raph/leela/data/test/"
    # ---------------------

    scan_for_nans(TRAIN_DIR)
    scan_for_nans(TEST_DIR)