#!/usr/bin/env python3
import os
import sys
import json
import random
import struct
import gzip
import logging
import tarfile
import threading
import requests
import numpy as np
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

# --- Structural Constants from Chunkparser ---
V6_VERSION = struct.pack('i', 6)
V7_VERSION = struct.pack('i', 7)
V6_STRUCT_STRING = '4si7432s832sBBBBBBBbfffffffffffffffIHH4H'
V7_STRUCT_STRING = "4si7432s832sBBBBBBBbfffffffffffffffIHHfffHHffffffff"

v6_struct = struct.Struct(V6_STRUCT_STRING)
v7_struct = struct.Struct(V7_STRUCT_STRING)
V6_RECORD_SIZE = v6_struct.size
V7_RECORD_SIZE = v7_struct.size

# --- Numerical Rescoring Utilities ---
def apply_alpha(qs, alpha, alt_signs=True):
    if not isinstance(qs, np.ndarray):
        qs = np.array(qs)
    n = len(qs)
    signs = (-1)**np.arange(n) if alt_signs else 1
    qs = qs * signs
    q_st = np.zeros(n)
    val = 0
    for i in range(n):
        if i == 0:
            val = qs[-1]
        else:
            val = alpha * val + qs[-i-1] * (1 - alpha)
        q_st[-i-1] = val
    q_st = q_st * signs
    return q_st

def rescore_chunk_data(chunkdata, st_alpha=1-1/6):
    """Processes uncompressed chunk data from V6 to V7 in memory."""
    n_chunks = len(chunkdata) // V6_RECORD_SIZE
    qs, ds, play_idx = [], [], []
    
    for i in range(n_chunks):
        offset = i * V6_RECORD_SIZE
        qs.append(struct.unpack("f", chunkdata[offset+8280:offset+8284])[0])
        ds.append(struct.unpack("f", chunkdata[offset+8288:offset+8292])[0])
        play_idx.append(chunkdata[offset+8344:offset+8346])
    play_idx += [struct.pack("H", 65535)] * 2

    st_q = apply_alpha(qs, st_alpha)
    st_d = apply_alpha(ds, st_alpha, alt_signs=False)
    
    cd_array = bytearray()
    for i in range(n_chunks):
        offset = i * V6_RECORD_SIZE
        new_chunk = bytearray(chunkdata[offset:offset+V6_RECORD_SIZE] + b"\x00" * (V7_RECORD_SIZE - V6_RECORD_SIZE))
        new_chunk[8352:8356] = struct.pack("f", st_q[i])
        new_chunk[8356:8360] = struct.pack("f", max(st_d[i], 0))
        new_chunk[0:4] = V7_VERSION
        new_chunk[8360:8362] = play_idx[i+1]
        new_chunk[8362:8364] = play_idx[i+2]
        cd_array += new_chunk
        
    return bytes(cd_array)

# --- State Management ---
def load_state(state_file):
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
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=4)

# --- Core Pipeline Process ---
def process_single_archive(file_url, file_name, base_dir, train_dir, test_dir, train_ratio):
    if abort_event.is_set():
        return {"success": False, "reason": "Aborted before start", "file_name": file_name}

    file_path = os.path.join(base_dir, file_name)
    stats = {
        "file_name": file_name,
        "train_count": 0, 
        "test_count": 0, 
        "bytes": 0, 
        "success": False,
        "rescored": 0,
        "already_v7": 0,
        "skipped": 0
    }

    try:
        # --- PHASE A: Stream Download ---
        logger.info(f"[{file_name}] Starting download...")
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if abort_event.is_set():
                        return {"success": False, "reason": "Aborted during download", "file_name": file_name}
                    if chunk:
                        f.write(chunk)
                                
        logger.info(f"[{file_name}] Download complete. Processing elements entirely in RAM...")

        # --- PHASE B: Extract, Rescore, and Route in Memory ---
        if abort_event.is_set(): 
            return {"success": False, "reason": "Aborted before processing", "file_name": file_name}
            
        with tarfile.open(file_path, 'r:*') as tar:
            members = [
                m for m in tar.getmembers() 
                if m.isfile() and (m.name.endswith('.gz') or m.name.endswith('.chunk'))
            ]
            
            for member in members:
                if abort_event.is_set():
                    return {"success": False, "reason": "Aborted during extraction", "file_name": file_name}

                f_mem = tar.extractfile(member)
                if f_mem is None:
                    continue
                
                # Read compressed payload directly from Tar stream
                compressed_payload = f_mem.read()
                
                try:
                    chunkdata = gzip.decompress(compressed_payload)
                except Exception as decompress_err:
                    logger.error(f"[{file_name}] Decompression failed for sub-file {member.name}: {decompress_err}")
                    stats["skipped"] += 1
                    continue

                if len(chunkdata) == 0:
                    stats["skipped"] += 1
                    continue

                version = chunkdata[0:4]
                out_bytes = None

                # Version Filter Evaluation
                if version == V7_VERSION:
                    out_bytes = compressed_payload
                    stats["already_v7"] += 1
                elif version == V6_VERSION:
                    try:
                        rescored_data = rescore_chunk_data(chunkdata)
                        out_bytes = gzip.compress(rescored_data)
                        stats["rescored"] += 1
                    except Exception as rescore_err:
                        logger.error(f"[{file_name}] Mathematical rescoring failed on {member.name}: {rescore_err}")
                        stats["skipped"] += 1
                        continue
                else:
                    # Automatically drop unidentified or non-v6 versions without writing to disk
                    stats["skipped"] += 1
                    continue

                # Route dynamically into Train/Test partitions
                if random.random() < train_ratio:
                    dest_dir = train_dir
                    stats["train_count"] += 1
                else:
                    dest_dir = test_dir
                    stats["test_count"] += 1

                # Crash-proof Atomic File Persistence
                final_output_path = os.path.join(dest_dir, os.path.basename(member.name))
                tmp_output_path = final_output_path + ".tmp"
                
                with open(tmp_output_path, 'wb') as out_f:
                    out_f.write(out_bytes)
                os.replace(tmp_output_path, final_output_path)
                
                stats["bytes"] += len(out_bytes)

        stats["success"] = True
        logger.info(f"[{file_name}] Finished! Rescored: {stats['rescored']} | Confirmed V7: {stats['already_v7']} | Skipped: {stats['skipped']}")

    except Exception as e:
        logger.error(f"[{file_name}] Core worker failed with error: {e}")
        stats["success"] = False
    finally:
        # --- PHASE C: Cleanup Raw Tar ---
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"[{file_name}] Cleared temporary tar archive from scratch.")
            
    return stats

def download_and_rescore_pipeline(index_url, base_dir, target_games, max_gb, train_ratio=0.8, concurrent_archives=4):
    base_dir = os.path.expanduser(base_dir)
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    max_bytes = max_gb * 1024 * 1024 * 1024
    state_file = os.path.join(base_dir, "resume_state.json")
    state = load_state(state_file)

    logger.info(f"Base Destination: {base_dir}")
    logger.info(f"Resuming pipeline: {state['total_games_extracted']} total games processed, "
                f"{state['total_bytes_extracted'] / (1024**3):.2f} GB of safe V7 files on disk.")
    
    if state["total_games_extracted"] >= target_games or state["total_bytes_extracted"] >= max_bytes:
        logger.info("Target milestones reached. Exiting pipeline.")
        return

    logger.info(f"Scanning online directory structure: {index_url}")
    try:
        response = requests.get(index_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch resource index: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    file_links = [
        a.get('href') for a in soup.find_all('a') 
        if a.get('href') and not a.get('href').startswith('?') 
        and a.get('href') != '../' 
        and (a.get('href').endswith('.tar') or a.get('href').endswith('.tar.gz'))
    ]

    # Single manifest filter check
    pending_archives = [f for f in file_links if f not in state["processed_archives"]]

    if not pending_archives:
        logger.warning("No new online data bundles found to process.")
        return

    logger.info(f"Saturating pipeline with {len(pending_archives)} pending archives.")

    with ThreadPoolExecutor(max_workers=concurrent_archives) as executor:
        futures = []
        for file_name in pending_archives:
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

        for future in as_completed(futures):
            result = future.result()
            
            if result["success"]:
                with state_lock:
                    state["train_games_extracted"] += result["train_count"]
                    state["test_games_extracted"] += result["test_count"]
                    state["total_games_extracted"] += (result["train_count"] + result["test_count"])
                    state["total_bytes_extracted"] += result["bytes"]
                    state["processed_archives"].append(result["file_name"])
                    
                    save_state(state_file, state)
                    
                    current_gb = state["total_bytes_extracted"] / (1024**3)
                    logger.info(f"{'='*60}")
                    logger.info(f"--- ARCHIVE SUBMISSION TRACKED AND COMMITTED ---")
                    logger.info(f"Bundle Manifest Verified: {result['file_name']}")
                    logger.info(f"Global Pool Metric: {state['total_games_extracted']}/{target_games} games | {current_gb:.2f}/{max_gb} GB")
                    logger.info(f"Partition Matrix: Train({state['train_games_extracted']}) / Test({state['test_games_extracted']})")
                    logger.info(f"{'='*60}")
                    
                    if state["total_games_extracted"] >= target_games or state["total_bytes_extracted"] >= max_bytes:
                        logger.info("Limits achieved. Executing structured task wind-down...")
                        abort_event.set()

    logger.info("[SUCCESS] Integrated Download & Rescore Pipeline Finished.")

if __name__ == "__main__":
    URL = "https://data.lczero.org/files/training_data/test91/"
    DESTINATION_FOLDER = "/lustre/fsn1/projects/rech/kwf/uzr96yg/leela/data"
    
    TARGET_GAMES = 100000000
    MAX_DISK_SPACE_GB = 4000  
    TRAIN_RATIO = 0.95
    
    CONCURRENT_ARCHIVES = 8
    
    download_and_rescore_pipeline(URL, DESTINATION_FOLDER, TARGET_GAMES, MAX_DISK_SPACE_GB, TRAIN_RATIO, CONCURRENT_ARCHIVES)