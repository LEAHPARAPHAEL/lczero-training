import os
import requests
import tarfile
import json
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from tqdm import tqdm

def load_state(state_file):
    """Loads the progress state from disk, or creates a fresh one."""
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            return json.load(f)
    return {
        "total_games_extracted": 0,
        "total_bytes_extracted": 0,
        "processed_archives": []
    }

def save_state(state_file, state):
    """Saves the current progress to disk."""
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=4)

def download_and_extract_smart(index_url, target_dir, target_games, max_gb):
    """
    Downloads and extracts data until either the game count or disk space 
    threshold is reached. Fully resumable.
    """
    target_dir = os.path.expanduser(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    # Convert GB to exact Bytes for calculation
    max_bytes = max_gb * 1024 * 1024 * 1024
    
    state_file = os.path.join(target_dir, "resume_state.json")
    state = load_state(state_file)

    print(f"Target Directory: {target_dir}")
    print(f"Resuming from state: {state['total_games_extracted']} games, "
          f"{state['total_bytes_extracted'] / (1024**3):.2f} GB used.")
    
    # Check if we already hit the limits from a previous run
    if state["total_games_extracted"] >= target_games:
        print("\nTarget game count already reached in previous runs. Exiting.")
        return
    if state["total_bytes_extracted"] >= max_bytes:
        print("\nMaximum disk space limit already reached in previous runs. Exiting.")
        return

    print(f"Fetching index from: {index_url}")
    try:
        response = requests.get(index_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch the URL: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    file_links = [
        a.get('href') for a in soup.find_all('a') 
        if a.get('href') and not a.get('href').startswith('?') 
        and a.get('href') != '../' 
        and (a.get('href').endswith('.tar') or a.get('href').endswith('.tar.gz'))
    ]

    if not file_links:
        print("No valid archive files found on the page.")
        return

    # Process files
    for file_name in file_links:
        # 1. Skip files we already successfully processed
        if file_name in state["processed_archives"]:
            continue

        # 2. Check limits before starting a new file
        if state["total_games_extracted"] >= target_games:
            print(f"\n[SUCCESS] Target of {target_games} games reached!")
            break
        if state["total_bytes_extracted"] >= max_bytes:
            print(f"\n[SUCCESS] Max disk limit of {max_gb}GB reached!")
            break

        file_url = urljoin(index_url, file_name)
        clean_file_name = os.path.basename(urlparse(file_url).path)
        file_path = os.path.join(target_dir, clean_file_name)

        print(f"\n--- Processing: {clean_file_name} ---")
        
        # --- PHASE A: Download ---
        # Note: If a download was partially completed before a crash, 
        # 'wb' mode cleanly overwrites the broken file and starts fresh.
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as f, tqdm(
                desc="Downloading", total=total_size, unit='iB',
                unit_scale=True, unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)

        # --- PHASE B: Extract & Measure ---
        print("Extracting and calculating disk footprint...")
        try:
            with tarfile.open(file_path, 'r:*') as tar:
                members = tar.getmembers()
                files_to_extract = [m for m in members if m.isfile()]
                
                # Calculate exactly how much space these files will take
                extracted_size_bytes = sum(m.size for m in files_to_extract)
                games_in_archive = len(files_to_extract)
                
                tar.extractall(path=target_dir, members=files_to_extract)
                
                # Update our live state
                state["total_games_extracted"] += games_in_archive
                state["total_bytes_extracted"] += extracted_size_bytes
                state["processed_archives"].append(file_name)
                
                # Save progress immediately. If it crashes 1 second from now, we are safe.
                save_state(state_file, state)
                
                current_gb = state["total_bytes_extracted"] / (1024**3)
                print(f"Extracted {games_in_archive} games ({extracted_size_bytes / (1024**2):.2f} MB).")
                print(f"Running Total: {state['total_games_extracted']}/{target_games} games | {current_gb:.2f}/{max_gb} GB")
                
        except tarfile.TarError as e:
            print(f"Error extracting {clean_file_name}: {e}")

        # --- PHASE C: Cleanup ---
        if os.path.exists(file_path):
            os.remove(file_path)
            print("Deleted archive to reclaim disk space.")

    print("\nScript execution finished.")

# --- Usage Example ---
if __name__ == "__main__":
    URL = "https://data.lczero.org/files/training_data/test91/"
    DESTINATION_FOLDER = "~/leela/data"
    
    TARGET_GAMES = 500000
    MAX_DISK_SPACE_GB = 20  # Will stop automatically if it hits 20GB of extracted data
    
    download_and_extract_smart(URL, DESTINATION_FOLDER, TARGET_GAMES, MAX_DISK_SPACE_GB)