import os
import requests
import tarfile
import json
import random
import logging
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

# --- Set up HPC-friendly Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_state(state_file):
    """Loads the progress state from disk, or creates a fresh one."""
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
            # Fallbacks in case an older version of the state file is found
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

def download_and_split_smart(index_url, base_dir, target_games, max_gb, train_ratio=0.8):
    """
    Downloads, extracts, and randomly splits data into train/test directories
    until limits are reached. Fully resumable.
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
    
    if state["total_games_extracted"] >= target_games:
        logger.info("Target game count already reached in previous runs. Exiting.")
        return
    if state["total_bytes_extracted"] >= max_bytes:
        logger.info("Maximum disk space limit already reached in previous runs. Exiting.")
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
                            if percent >= last_log_percent + 10:
                                logger.info(f"Downloading: {percent}% ({downloaded_bytes / (1024**2):.1f} MB / {total_size / (1024**2):.1f} MB)")
                                last_log_percent = percent
        
        logger.info(f"Download complete for {clean_file_name}.")

        # --- Inside Phase B: Extract & Split ---
        logger.info("Extracting and splitting files...")
        try:
            with tarfile.open(file_path, 'r:*') as tar:
                members = tar.getmembers()
                
                # FILTERING: Only include files that look like games/chunks
                files_to_extract = [
                    m for m in members 
                    if m.isfile() and (m.name.endswith('.gz') or m.name.endswith('.chunk'))
                ]
                
                total_files = len(files_to_extract)
                games_extracted_this_archive = 0
                bytes_extracted_this_archive = 0
                
                for i, member in enumerate(files_to_extract, 1):
                    if state["total_games_extracted"] >= target_games or state["total_bytes_extracted"] >= max_bytes:
                        break

                    # Random split logic
                    if random.random() < train_ratio:
                        dest_dir = train_dir
                        state["train_games_extracted"] += 1
                    else:
                        dest_dir = test_dir
                        state["test_games_extracted"] += 1

                    # EXTRACTING WITH NESTED LAYOUT
                    tar.extract(member, path=dest_dir)
                    
                    state["total_games_extracted"] += 1
                    state["total_bytes_extracted"] += member.size
                    games_extracted_this_archive += 1
                    bytes_extracted_this_archive += member.size
                    
                    # Log extraction progress every 1000 files
                    if i % 1000 == 0 or i == total_files:
                        logger.info(f"Extracted {i}/{total_files} files...")
                
                # Mark archive as processed only if we finished it or hit our global limits
                state["processed_archives"].append(file_name)
                save_state(state_file, state)
                
                current_gb = state["total_bytes_extracted"] / (1024**3)
                logger.info(f"Extracted {games_extracted_this_archive} games ({bytes_extracted_this_archive / (1024**2):.2f} MB).")
                logger.info(f"Running Total: {state['total_games_extracted']}/{target_games} games | {current_gb:.2f}/{max_gb} GB")
                logger.info(f"Current Split: Train({state['train_games_extracted']}) / Test({state['test_games_extracted']})")
                
        except tarfile.TarError as e:
            logger.error(f"Error extracting {clean_file_name}: {e}")

        # --- PHASE C: Cleanup ---
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info("Deleted archive to reclaim disk space.")

    logger.info("[SUCCESS] Pipeline execution finished.")
    logger.info(f"Final Count -> Train: {state['train_games_extracted']} | Test: {state['test_games_extracted']}")

# --- Usage Example ---
if __name__ == "__main__":
    URL = "https://data.lczero.org/files/training_data/test91/"
    DESTINATION_FOLDER = "/lustre/fsn1/projects/rech/kwf/uzr96yg/leela/data"
    
    TARGET_GAMES = 10000000
    MAX_DISK_SPACE_GB = 400  
    TRAIN_RATIO = 0.95  # 95% to train, 5% to test
    
    download_and_split_smart(URL, DESTINATION_FOLDER, TARGET_GAMES, MAX_DISK_SPACE_GB, TRAIN_RATIO)