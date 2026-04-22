import glob
import os
# Import Daniel's custom rescoring tool from his chunkparser
from chunkparser import rescore

def main():
    # Define absolute paths using os.path.expanduser
    train_dir = os.path.expanduser("/home/raph/leela/data/train/*/")
    test_dir = os.path.expanduser("/home/raph/leela/data/test/*/")
    
    # Grab all the .gz files
    train_files = glob.glob(train_dir + "*.gz")
    test_files = glob.glob(test_dir + "*.gz")
    
    print(f"Found {len(train_files)} training files and {len(test_files)} testing files.")
    
    # We must use multiprocessing protection for his ProcessPoolExecutor
    print("\n--- Upgrading Training Data to v7 ---")
    rescore(train_files)
    
    print("\n--- Upgrading Testing Data to v7 ---")
    rescore(test_files)
    
    print("\nAll data successfully upgraded to v7!")

if __name__ == '__main__':
    main()