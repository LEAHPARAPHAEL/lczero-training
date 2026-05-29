import os
import glob
import requests
import tarfile
import json
import random
import logging
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

# Import Daniel's custom rescoring tool
from chunkparser import rescore

# --- Set up HPC-friendly Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def clean_tmp_files(directory):
    """Deletes any incomplete .tmp files from a previous crashed run."""
    tmp_files = glob.glob(os.path.join(directory, "**/*.tmp"), recursive=True)
    for tmp in tmp_files:
        try:
            os.remove(tmp)
        except OSError:
            pass
    if tmp_files:
        logger.info(f"Cleaned up {len(tmp_files)} incomplete .tmp files in {directory}")

def load_state(state_file):
    """Loads the progress state from disk, or creates a fresh one."""
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
            return {
                "total_games_extracted": state.get("total_games_extracted", 0),
                "train_games_extracted": state.get("train_games_extracted", 0),
                "test_games_extracted": state.get("test_games_extracted", 0),
                "total_bytes_extracted": state.get("total_bytes_extracted", 0),
                "processed_archives": state.get("processed_archives", [])
            }
    return {
        "total_games_extracted": 0,
        "train_games_extracted": 0,
        "test_games_extracted": 0,
        "total_bytes_extracted": 0,
        "processed_archives": []
    }

def save_state(state_file, state):
    """Saves the current progress to disk."""
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=4)

def download_extract_and_rescore(index_url, base_dir, target_games, max_gb, train_ratio=0.8):
    """
    Downloads, extracts, splits, and rescores data on the fly. 
    Fully resumable, crash-safe, and optimized for supercomputer logging.
    """
    base_dir = os.path.expanduser(base_dir)
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    logger.info("Running pre-flight cleanup of aborted rescores...")
    clean_tmp_files(train_dir)
    clean_tmp_files(test_dir)
    
    max_bytes = max_gb * 1024 * 1024 * 1024
    state_file = os.path.join(base_dir, "resume_state.json")
    state = load_state(state_file)

    logger.info(f"Base Directory: {base_dir}")
    logger.info(f"Resuming from state: {state['total_games_extracted']} total games "
                f"({state['train_games_extracted']} train | {state['test_games_extracted']} test), "
                f"{state['total_bytes_extracted'] / (1024**3):.2f} GB used.")
    
    if state["total_games_extracted"] >= target_games or state["total_bytes_extracted"] >= max_bytes:
        logger.info("Target limits already reached in previous runs. Exiting.")
        return

    logger.info(f"Fetching index from: {index_url}")
    try:
        response = requests.get(index_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch the URL: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    file_links = [
        a.get('href') for a in soup.find_all('a') 
        if a.get('href') and not a.get('href').startswith('?') 
        and a.get('href') != '../' 
        and (a.get('href').endswith('.tar') or a.get('href').endswith('.tar.gz'))
    ]

    if not file_links:
        logger.warning("No valid archive files found on the page.")
        return

    for file_name in file_links:
        if file_name in state["processed_archives"]:
            continue

        if state["total_games_extracted"] >= target_games or state["total_bytes_extracted"] >= max_bytes:
            break

        file_url = urljoin(index_url, file_name)
        clean_file_name = os.path.basename(urlparse(file_url).path)
        file_path = os.path.join(base_dir, clean_file_name)

        logger.info(f"{'='*50}")
        logger.info(f"Processing: {clean_file_name}")
        logger.info(f"{'='*50}")
        
        # --- PHASE A: Download ---
        logger.info(f"Starting download of {clean_file_name}...")
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded_bytes = 0
            last_log_percent = 0
            
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        size = f.write(chunk)
                        downloaded_bytes += size
                        
                        # Log progress every 10%
                        if total_size > 0:
                            percent = int((downloaded_bytes / total_size) * 100)
                            if percent >= last_log_percent + 1:
                                logger.info(f"Downloading: {percent}% ({downloaded_bytes / (1024**2):.1f} MB / {total_size / (1024**2):.1f} MB)")
                                last_log_percent = percent
        
        logger.info(f"Download complete for {clean_file_name}.")

        # --- PHASE B: Extract & Route ---
        logger.info("Extracting and routing files...")
        local_train_count = 0
        local_test_count = 0
        local_bytes = 0
        current_batch_train_files = []
        current_batch_test_files = []
        
        try:
            with tarfile.open(file_path, 'r:*') as tar:
                members = tar.getmembers()
                files_to_extract = [
                    m for m in members 
                    if m.isfile() and (m.name.endswith('.gz') or m.name.endswith('.chunk'))
                ]
                total_files = len(files_to_extract)
                
                for i, member in enumerate(files_to_extract, 1):
                    if (state["total_games_extracted"] + local_train_count + local_test_count) >= target_games or \
                       (state["total_bytes_extracted"] + local_bytes) >= max_bytes:
                        break

                    if random.random() < train_ratio:
                        dest_dir = train_dir
                        local_train_count += 1
                        if member.name.endswith('.gz'):
                            current_batch_train_files.append(os.path.join(dest_dir, member.name))
                    else:
                        dest_dir = test_dir
                        local_test_count += 1
                        if member.name.endswith('.gz'):
                            current_batch_test_files.append(os.path.join(dest_dir, member.name))

                    tar.extract(member, path=dest_dir)
                    local_bytes += member.size
                    
                    # Log extraction progress every 1000 files
                    if i % 1000 == 0 or i == total_files:
                        logger.info(f"Extracted {i}/{total_files} files...")
                    
            # --- PHASE C: On-the-fly Rescoring ---
            if current_batch_train_files:
                logger.info(f"Rescoring {len(current_batch_train_files)} Training files from {clean_file_name}...")
                rescore(current_batch_train_files)
                
            if current_batch_test_files:
                logger.info(f"Rescoring {len(current_batch_test_files)} Testing files from {clean_file_name}...")
                rescore(current_batch_test_files)
                
            # --- PHASE D: Commit State & Cleanup ---
            state["train_games_extracted"] += local_train_count
            state["test_games_extracted"] += local_test_count
            state["total_games_extracted"] += (local_train_count + local_test_count)
            state["total_bytes_extracted"] += local_bytes
            state["processed_archives"].append(file_name)
            
            save_state(state_file, state)
            
            current_gb = state["total_bytes_extracted"] / (1024**3)
            logger.info(f"[Archive Complete] Running Total: {state['total_games_extracted']}/{target_games} games | {current_gb:.2f}/{max_gb} GB")
            logger.info(f"Current Split: Train({state['train_games_extracted']}) / Test({state['test_games_extracted']})")
                
        except Exception as e:
            logger.error(f"Error processing {clean_file_name}: {e}")
            logger.warning("State not committed for this archive. It will be retried on next run.")

        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info("Deleted archive to reclaim disk space.")

    logger.info("[SUCCESS] Pipeline execution finished.")
    logger.info(f"Final Count -> Train: {state['train_games_extracted']} | Test: {state['test_games_extracted']}")

if __name__ == "__main__":
    URL = "https://data.lczero.org/files/training_data/test91/"
    DESTINATION_FOLDER = "/lustre/fsn1/projects/rech/kwf/uzr96yg/leela/data"
    
    TARGET_GAMES = 10000000
    MAX_DISK_SPACE_GB = 200 
    TRAIN_RATIO = 0.95  
    
    download_extract_and_rescore(URL, DESTINATION_FOLDER, TARGET_GAMES, MAX_DISK_SPACE_GB, TRAIN_RATIO)