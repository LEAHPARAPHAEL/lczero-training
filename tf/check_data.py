import os
import glob
import gzip
import random
import struct

def verify_v7_chunks(directory, num_to_check=1000):
    # Define the V7 magic bytes exactly as chunkparser.py does
    V7_VERSION = struct.pack('i', 7)

    print(f"Scanning for .gz files in {directory}...")
    # Grab all .gz files (using recursive glob to handle subdirectories)
    all_files = glob.glob(os.path.join(directory, "**/*.gz"), recursive=True)
    
    if not all_files:
        print(f"No .gz files found in {directory}!")
        return

    total_files = len(all_files)
    print(f"Found {total_files} files.")

    # Ensure we don't try to sample more files than actually exist
    sample_size = min(num_to_check, total_files)
    sampled_files = random.sample(all_files, sample_size)

    print(f"Randomly selected {sample_size} files for V7 verification...\n")

    v7_count = 0
    non_v7_count = 0
    error_count = 0

    for file_path in sampled_files:
        try:
            with gzip.open(file_path, 'rb') as f:
                version_bytes = f.read(4)
                
                if version_bytes == V7_VERSION:
                    v7_count += 1
                else:
                    non_v7_count += 1
                    # Attempt to unpack the wrong version to see what it actually is
                    try:
                        actual_version = struct.unpack('i', version_bytes)[0]
                        print(f"[Mismatch] {file_path} is Version {actual_version}")
                    except struct.error:
                        print(f"[Mismatch] {file_path} has an unknown or corrupted header: {version_bytes}")
                        
        except Exception as e:
            error_count += 1
            print(f"[Error] Could not read {file_path}: {e}")

    # --- Print Summary ---
    print("\n" + "="*40)
    print(f" VERIFICATION RESULTS: {os.path.basename(os.path.normpath(directory))}")
    print("="*40)
    print(f"Total checked : {sample_size}")
    print(f"Valid V7      : {v7_count}")
    print(f"Not V7        : {non_v7_count}")
    print(f"Read Errors   : {error_count}")
    print("="*40)
    
    if v7_count == sample_size:
        print("\n[SUCCESS] All sampled files are strictly V7!")
    else:
        print("\n[WARNING] Found non-V7 or corrupted files in the sample.")

if __name__ == '__main__':
    # Define absolute paths (Update these to match your actual Jean Zay paths if needed)
    TRAIN_DIR = os.path.expanduser("/lustre/fsn1/projects/rech/kwf/uzr96yg/leela/data/train/")
    TEST_DIR = os.path.expanduser("/lustre/fsn1/projects/rech/kwf/uzr96yg/leela/data/test/")
    
    # Set how many random files you want to check per folder
    CHUNKS_TO_CHECK = 100
    
    print("--- Verifying Training Data ---")
    verify_v7_chunks(TRAIN_DIR, CHUNKS_TO_CHECK)
    
    print("\n\n--- Verifying Testing Data ---")
    verify_v7_chunks(TEST_DIR, CHUNKS_TO_CHECK)