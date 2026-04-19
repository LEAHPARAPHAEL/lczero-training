import gzip
import struct
import math
import glob
import os

V7_STRUCT_STRING = "4si7432s832sBBBBBBBbfffffffffffffffIHHfffHHffffffff"
v7_struct = struct.Struct(V7_STRUCT_STRING)
record_size = v7_struct.size

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