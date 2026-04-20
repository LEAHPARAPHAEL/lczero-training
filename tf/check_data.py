import gzip
import struct
import math
import glob
import os
import numpy as np
from tqdm import tqdm

V7_STRUCT_STRING = "4si7432s832sBBBBBBBbfffffffffffffffIHHfffHHffffffff"
v7_struct = struct.Struct(V7_STRUCT_STRING)
record_size = v7_struct.size


'''
def find_out_of_bounds(file_path, max_records=100):
    print(f"\n--- Scanning {os.path.basename(file_path)} ---")
    with gzip.open(file_path, "rb") as f:
        chunkdata = f.read()

    if len(chunkdata) == 0:
        print("Error: Empty file.")
        return

    n_chunks = len(chunkdata) // record_size
    records_to_check = min(n_chunks, max_records)
    
    print(f"Hunting for out-of-bounds values across {records_to_check} records...")

    found = False
    for i in range(records_to_check):
        chunk = chunkdata[i*record_size:(i+1)*record_size]
        
        # Extract the key bytes
        root_q = struct.unpack("f", chunk[8280:8284])[0]
        best_q = struct.unpack("f", chunk[8284:8288])[0]
        orig_q = struct.unpack("f", chunk[8328:8332])[0]
        st_q   = struct.unpack("f", chunk[8352:8356])[0]
        st_d   = struct.unpack("f", chunk[8356:8360])[0]

        # Check if the values drift past our strict threshold
        st_q_bad = not (-1.01 <= st_q <= 1.01)
        st_d_bad = not (-1.01 <= st_d <= 1.01)

        if st_q_bad or st_d_bad:
            print(f"\n>>> [ANOMALY CAUGHT AT RECORD {i}] <<<")
            
            # Explicitly flag if the anomaly is a NaN vs a Floating Point Drift
            if math.isnan(st_q) or math.isnan(st_d):
                print("Reason: NaN Poisoning Detected.")
            else:
                print("Reason: Floating Point bounds exceeded.")
                
            print(f"root_q={root_q:>+6.3f} | best_q={best_q:>+6.3f} | orig_q={orig_q:>+6.3f}")
            print(f"st_q={st_q:>+6.3f} | st_d={st_d:>+6.3f}")
            
            found = True
            break # Stop immediately once we catch one!
            
    if not found:
        print(f"\nClean! No anomalies found within the first {records_to_check} records.")

if __name__ == '__main__':
    train_dir = os.path.expanduser("~/leela/data/train/*/")
    files = glob.glob(os.path.expanduser(train_dir + "*.gz"))
    
    if files:
        # You can increase max_records here if you want it to scan deeper into the file!
        find_out_of_bounds(files[4], max_records=10000)
    else:
        print(f"Could not find any .gz files in {train_dir}")
'''


# Exact struct definition extracted from your chunkparser.py
V7_STRUCT_STRING = "4si7432s832sBBBBBBBbfffffffffffffffIHHfffHHffffffff"
v7_struct = struct.Struct(V7_STRUCT_STRING)

# Exact struct definition extracted from your chunkparser.py

def verify_v7_dataset(directories):
    # Initialize counters for the v7 format
    nan_counts = {
        'policy': 0,
        'plies_left': 0,
        'root_wdl': 0,
        'st_wdl': 0,
        'result_wdl': 0,
        'total_games': 0
    }

    # Find all chunk files
    filepaths = []
    for d in directories:
        filepaths.extend(glob.glob(os.path.expanduser(d + '/*.gz')))
    
    print(f"Found {len(filepaths)} v7 chunk files to verify...")
    if not filepaths:
        return nan_counts

    record_size = v7_struct.size 

    # Wrap the filepaths in tqdm for the progress bar
    pbar = tqdm(filepaths, desc="Verifying Chunks", unit="file")

    for filepath in pbar:
        try:
            with gzip.open(filepath, 'rb') as f:
                while True:
                    content = f.read(record_size)
                    if not content or len(content) < record_size:
                        break # EOF or partial record
                    
                    nan_counts['total_games'] += 1

                    # 1. Unpack exactly using the v7 struct
                    unpacked = v7_struct.unpack(content)
                    
                    # 2. Extract only the relevant fields
                    probs_bytes = unpacked[2]  # 7432s
                    root_q     = unpacked[12]
                    root_d     = unpacked[14]
                    plies_left = unpacked[18]
                    result_q   = unpacked[19]
                    result_d   = unpacked[20]
                    st_q       = unpacked[31]
                    st_d       = unpacked[32]

                    # 3. VERIFICATION CHECKS
                    
                    # Check Policy
                    probs_array = np.frombuffer(probs_bytes, dtype=np.float32)
                    if np.isnan(probs_array).any():
                        nan_counts['policy'] += 1

                    # Check Moves Left
                    if np.isnan(plies_left):
                        nan_counts['plies_left'] += 1

                    # Check Root WDL 
                    if np.isnan(root_q) or np.isnan(root_d):
                        nan_counts['root_wdl'] += 1

                    # Check ST WDL 
                    if np.isnan(st_q) or np.isnan(st_d):
                        nan_counts['st_wdl'] += 1

                    # Check Game Result 
                    if np.isnan(result_q) or np.isnan(result_d):
                        nan_counts['result_wdl'] += 1
                        
        except Exception as e:
            # Use tqdm.write so error messages don't break the progress bar formatting
            tqdm.write(f"Error reading {filepath}: {e}")
            continue

        # Update the progress bar's right-side text after every file
        # This keeps the loop lightning fast while giving you live stats
        total_errors = sum(v for k, v in nan_counts.items() if k != 'total_games')
        pbar.set_postfix({
            'Games': f"{nan_counts['total_games']:,}",
            'Errors': f"{total_errors:,}"
        })

    return nan_counts

# Run the routine
if __name__ == "__main__":
    directories_to_check = [
        '~/leela/data/train/*/',
        '~/leela/data/test/*/'
    ]
    
    print("Starting V7 Data Verification...")
    results = verify_v7_dataset(directories_to_check)
    
    print("\n" + "="*40)
    print("VERIFICATION RESULTS")
    print("="*40)
    print(f"Total Games Parsed: {results['total_games']:,}")
    print("-" * 40)
    print(f"Policy Corruptions:      {results['policy']:,}")
    print(f"Moves Left Corruptions:  {results['plies_left']:,}")
    print(f"Root WDL Corruptions:    {results['root_wdl']:,}")
    print(f"ST WDL Corruptions:      {results['st_wdl']:,}")
    print(f"Result WDL Corruptions:  {results['result_wdl']:,}")
    print("="*40)