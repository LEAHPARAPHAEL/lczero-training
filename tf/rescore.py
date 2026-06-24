'''
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
    
    train_dir = os.path.expanduser("/lustre/fsn1/projects/rech/kwf/uzr96yg/leela/data/train/*/")
    test_dir = os.path.expanduser("/lustre/fsn1/projects/rech/kwf/uzr96yg/leela/data/test/*/")
    
    raw_train_files = glob.glob(train_dir + "*.gz")
    raw_test_files = glob.glob(test_dir + "*.gz")
    
    print(f"Found {len(raw_train_files)} training files and {len(raw_test_files)} testing files on disk.", flush=True)
    
    completed_set = load_manifest(manifest_path)
    print(f"Loaded {len(completed_set)} already verified/processed records from manifest.", flush=True)
    
    train_files = [f for f in raw_train_files if f not in completed_set]
    test_files = [f for f in raw_test_files if f not in completed_set]
    
    print(f"Remaining to process: {len(train_files)} training files and {len(test_files)} testing files.", flush=True)
    
    if train_files:
        print("\n--- Upgrading Training Data to v7 ---", flush=True)
        rescore(train_files, manifest_path=manifest_path)
    else:
        print("\nTraining data already fully completed!", flush=True)
        
    if test_files:
        print("\n--- Upgrading Testing Data to v7 ---", flush=True)
        rescore(test_files, manifest_path=manifest_path)
    else:
        print("\nTesting data already fully completed!", flush=True)
    
    print("\nAll remaining operations finished!", flush=True)

if __name__ == '__main__':
    main()
'''

import glob
import os
from chunkparser import rescore

def main():
    # Gather directory patterns cleanly
    train_dir = os.path.expanduser("/lustre/fsn1/projects/rech/kwf/uzr96yg/leela/data/train/")
    test_dir = os.path.expanduser("/lustre/fsn1/projects/rech/kwf/uzr96yg/leela/data/test/")
    
    # Using recursive wildcards to ensure no nested run directories are missed
    print("Scanning directories for files...", flush=True)
    train_files = glob.glob(os.path.join(train_dir, "**/*.gz"), recursive=True)
    test_files = glob.glob(os.path.join(test_dir, "**/*.gz"), recursive=True)
    
    print(f"Found {len(train_files)} training files and {len(test_files)} testing files on disk.", flush=True)
    
    if train_files:
        print("\n--- Processing Training Data Pipeline ---", flush=True)
        rescore(train_files)
        
    if test_files:
        print("\n--- Processing Testing Data Pipeline ---", flush=True)
        rescore(test_files)
    
    print("\nAll data validation and upgrading routines completed!", flush=True)

if __name__ == '__main__':
    main()