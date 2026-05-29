import os
import requests
import tarfile
import json
import random
import logging
import threading
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Set up HPC-friendly Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Global signals for thread synchronization
abort_event = threading.Event()
state_lock = threading.Lock()

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
    """Saves the current progress to disk. (Must be called within a lock)"""
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=4)

def process_single_archive(file_url, file_name, base_dir, train_dir, test_dir, train_ratio):
    """Worker thread: Handles downloading, extracting, and splitting a single archive."""
    if abort_event.is_set():
        return {"success": False, "reason": "Aborted before start", "file_name": file_name}

    file_path = os.path.join(base_dir, file_name)
    
    # We accumulate stats locally so we don't lock the main thread during extraction
    stats = {
        "file_name": file_name,
        "train_count": 0, 
        "test_count": 0, 
        "bytes": 0, 
        "success": False
    }

    try:
        # --- PHASE A: Download ---
        logger.info(f"[{file_name}] Starting download...")
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded_bytes = 0
            last_log_percent = 0
            
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    # Check if another thread hit the limit, abort if so
                    if abort_event.is_set():
                        return {"success": False, "reason": "Aborted during download", "file_name": file_name}
                    
                    if chunk:
                        size = f.write(chunk)
                        downloaded_bytes += size
                        
                        # Log progress every 10%
                        if total_size > 0:
                            percent = int((downloaded_bytes / total_size) * 100)
                            if percent >= last_log_percent + 10:
                                logger.info(f"[{file_name}] Downloading: {percent}% ({downloaded_bytes / (1024**2):.1f} MB / {total_size / (1024**2):.1f} MB)")
                                last_log_percent = percent
                                
        logger.info(f"[{file_name}] Download complete.")

        # --- PHASE B: Extract & Route ---
        if abort_event.is_set(): 
            return {"success": False, "reason": "Aborted before extraction", "file_name": file_name}
            
        logger.info(f"[{file_name}] Extracting files...")
        
        with tarfile.open(file_path, 'r:*') as tar:
            members = tar.getmembers()
            files_to_extract = [
                m for m in members 
                if m.isfile() and (m.name.endswith('.gz') or m.name.endswith('.chunk'))
            ]
            
            total_files = len(files_to_extract)
            
            for i, member in enumerate(files_to_extract, 1):
                if abort_event.is_set():
                    return {"success": False, "reason": "Aborted during extraction", "file_name": file_name}

                # Random split logic
                if random.random() < train_ratio:
                    dest_dir = train_dir
                    stats["train_count"] += 1
                else:
                    dest_dir = test_dir
                    stats["test_count"] += 1

                tar.extract(member, path=dest_dir)
                stats["bytes"] += member.size
                
                # Log extraction progress every 1000 files
                if i % 1000 == 0 or i == total_files:
                    logger.info(f"[{file_name}] Extracted {i}/{total_files} files...")

        stats["success"] = True
        logger.info(f"[{file_name}] Extraction and routing 100% Complete.")

    except Exception as e:
        logger.error(f"[{file_name}] Failed with error: {e}")
        stats["success"] = False
    finally:
        # --- PHASE C: Cleanup ---
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"[{file_name}] Deleted archive to reclaim disk space.")
            
    return stats

def download_and_split_parallel(index_url, base_dir, target_games, max_gb, train_ratio=0.8, concurrent_archives=4):
    """
    Downloads, extracts, and randomly splits data into train/test directories
    using parallel threads to saturate network and I/O.
    """
    base_dir = os.path.expanduser(base_dir)
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
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

    pending_archives = [f for f in file_links if f not in state["processed_archives"]]

    if not pending_archives:
        logger.warning("No new archives found to process.")
        return

    logger.info(f"Found {len(pending_archives)} pending archives. Starting parallel processing...")
    logger.info(f"Downloading and extracting {concurrent_archives} archives concurrently.")

    # Execute archives in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=concurrent_archives) as executor:
        futures = []
        for file_name in pending_archives:
            # Quick check before launching thread
            with state_lock:
                if state["total_games_extracted"] >= target_games or state["total_bytes_extracted"] >= max_bytes:
                    abort_event.set()
                    break
                    
            file_url = urljoin(index_url, file_name)
            clean_name = os.path.basename(urlparse(file_url).path)
            
            futures.append(executor.submit(
                process_single_archive, file_url, clean_name, base_dir, 
                train_dir, test_dir, train_ratio
            ))

        # Handle results as they finish (in order of completion)
        for future in as_completed(futures):
            result = future.result()
            
            if result["success"]:
                # Use a lock to safely update global state
                with state_lock:
                    state["train_games_extracted"] += result["train_count"]
                    state["test_games_extracted"] += result["test_count"]
                    state["total_games_extracted"] += (result["train_count"] + result["test_count"])
                    state["total_bytes_extracted"] += result["bytes"]
                    state["processed_archives"].append(result["file_name"])
                    
                    save_state(state_file, state)
                    
                    current_gb = state["total_bytes_extracted"] / (1024**3)
                    logger.info(f"{'='*50}")
                    logger.info(f"[{result['file_name']}] STATE SAVED")
                    logger.info(f"Running Total: {state['total_games_extracted']}/{target_games} games | {current_gb:.2f}/{max_gb} GB")
                    logger.info(f"Current Split: Train({state['train_games_extracted']}) / Test({state['test_games_extracted']})")
                    logger.info(f"{'='*50}")
                    
                    # Check if global limits were reached by this latest commit
                    if state["total_games_extracted"] >= target_games or state["total_bytes_extracted"] >= max_bytes:
                        logger.info("Global Limits Reached! Signaling remaining workers to abort safely...")
                        abort_event.set()

    logger.info("\n[SUCCESS] Parallel Download Pipeline execution finished.")

# --- Usage Example ---
if __name__ == "__main__":
    URL = "https://data.lczero.org/files/training_data/test91/"
    DESTINATION_FOLDER = "/lustre/fsn1/projects/rech/kwf/uzr96yg/leela/data"
    
    TARGET_GAMES = 10000000
    MAX_DISK_SPACE_GB = 400  
    TRAIN_RATIO = 0.95  # 95% to train, 5% to test
    
    # --- I/O TUNING ---
    # Processing 4-8 archives concurrently is ideal for high-speed HPC networks. 
    # If the Lustre metadata server complains about too many files being created at once, lower this to 4.
    CONCURRENT_ARCHIVES = 8
    
    download_and_split_parallel(URL, DESTINATION_FOLDER, TARGET_GAMES, MAX_DISK_SPACE_GB, TRAIN_RATIO, CONCURRENT_ARCHIVES)