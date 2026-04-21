import os
import random
import shutil
from pathlib import Path

def rebalance_dataset(train_dir: str, test_dir: str, train_proportion: float):
    train_path = Path(train_dir)
    test_path = Path(test_dir)

    print("Scanning directories for chunks (this might take a moment)...")
    
    # 1. Gather all files (ignoring directories)
    train_files = [f for f in train_path.rglob('*') if f.is_file()]
    test_files = [f for f in test_path.rglob('*') if f.is_file()]

    n_train = len(train_files)
    n_test = len(test_files)
    n_total = n_train + n_test

    if n_total == 0:
        print("No files found in either directory!")
        return

    print(f"Found {n_total:,} total chunks ({n_train:,} in train, {n_test:,} in test).")

    # 2. Calculate targets
    target_train = int(n_total * train_proportion)
    target_test = n_total - target_train

    print(f"\nTarget distribution (train_proportion={train_proportion}):")
    print(f"  Target Train: {target_train:,}")
    print(f"  Target Test:  {target_test:,}")

    # 3. Determine movement direction and amount
    diff_train = target_train - n_train

    if diff_train == 0:
        print("\nDataset is already perfectly balanced!")
        return
    elif diff_train > 0:
        # We need more train data. Move from test -> train.
        n_to_move = diff_train
        source_files = test_files
        source_root = test_path
        dest_root = train_path
        print(f"\nAction: Moving {n_to_move:,} chunks from TEST to TRAIN.")
    else:
        # We have too much train data. Move from train -> test.
        n_to_move = abs(diff_train)
        source_files = train_files
        source_root = train_path
        dest_root = test_path
        print(f"\nAction: Moving {n_to_move:,} chunks from TRAIN to TEST.")

    # 4. Select random files to move
    print("Selecting random files...")
    files_to_move = random.sample(source_files, n_to_move)

    # 5. Move the files while preserving subdirectory structure
    print("Moving files...")
    for i, f in enumerate(files_to_move, 1):
        # Get the relative path (e.g., 'training-run1/chunk_123')
        rel_path = f.relative_to(source_root)
        
        # Create the new destination path
        dest_file = dest_root / rel_path
        
        # Ensure the parent directory exists in the destination
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Move the file
        shutil.move(str(f), str(dest_file))
        
        # Print progress every 10,000 files so you know it isn't frozen
        if i % 10000 == 0:
            print(f"  Moved {i:,} / {n_to_move:,} files...")

    print("\nRebalancing complete!")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    TRAIN_DIR = "/home/raph/leela/data/train/"
    TEST_DIR = "/home/raph/leela/data/test/"
    TRAIN_PROPORTION = 0.95
    # ---------------------

    rebalance_dataset(TRAIN_DIR, TEST_DIR, TRAIN_PROPORTION)