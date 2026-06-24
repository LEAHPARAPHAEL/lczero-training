import glob
import os
import sys
from chunkparser import rescore

def load_manifest(manifest_path):
    if not os.path.exists(manifest_path):
        return set()
    with open(manifest_path, "r") as f:
        return set(line.strip() for line in f if line.strip())

def main():
    manifest_path = os.path.expanduser("~/processed_chunks.txt")
    
    train_dir = os.path.expanduser("/lustre/fsn1/projects/rech/kwf/uzr96yg/leela/data/train/")
    test_dir = os.path.expanduser("/lustre/fsn1/projects/rech/kwf/uzr96yg/leela/data/test/")
    
    print("Scanning storage directories for files...", flush=True)
    raw_train_files = glob.glob(os.path.join(train_dir, "**/*.gz"), recursive=True)
    raw_test_files = glob.glob(os.path.join(test_dir, "**/*.gz"), recursive=True)
    
    print(f"Found {len(raw_train_files)} training files and {len(raw_test_files)} testing files on disk.", flush=True)
    
    # Load manifest and drop completed files in RAM instantly
    completed_set = load_manifest(manifest_path)
    print(f"Loaded {len(completed_set)} already verified/processed records from manifest.", flush=True)
    
    train_files = [f for f in raw_train_files if f not in completed_set]
    test_files = [f for f in raw_test_files if f not in completed_set]
    
    print(f"Net workload remaining: {len(train_files)} training files and {len(test_files)} testing files.", flush=True)
    
    if train_files:
        print("\n--- Processing Training Data Pipeline ---", flush=True)
        rescore(train_files, n_workers=40, manifest_path=manifest_path)
    else:
        print("\nTraining data already fully completed according to manifest!", flush=True)
        
    if test_files:
        print("\n--- Processing Testing Data Pipeline ---", flush=True)
        rescore(test_files, n_workers=40, manifest_path=manifest_path)
    else:
        print("\nTesting data already fully completed according to manifest!", flush=True)
    
    print("\nAll remaining operations finished!", flush=True)

if __name__ == '__main__':
    main()