#!/usr/bin/env python3
import os
import sys
import glob
import gzip
import random
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CHUNKS_PER_FILE = 20000  
N_WORKERS = 16           

def pack_single_batch(file_paths, output_path):
    try:
        combined_payload = bytearray()
        for path in file_paths:
            try:
                with gzip.open(path, 'rb') as f:
                    combined_payload.extend(f.read())
            except Exception:
                continue 
        
        if not combined_payload:
            return False
            
        tmp_output = output_path + ".tmp"
        with gzip.open(tmp_output, 'wb', compresslevel=1) as out_f:
            out_f.write(combined_payload)
        os.replace(tmp_output, output_path)
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la création de {output_path} : {e}")
        return False

def process_partition(source_dir, dest_dir, prefix):
    os.makedirs(dest_dir, exist_ok=True)
    
    logger.info(f"Analyse du dossier : {source_dir}...")
    all_files = glob.glob(os.path.join(source_dir, "*.gz"))
    total_files = len(all_files)
    
    if total_files == 0:
        logger.warning(f"Aucun fichier trouvé dans {source_dir}")
        return

    logger.info(f"Trouvé {total_files} fichiers. Lancement du Shuffling Global Déterministe...")
    # 🌟 REPRISE SÉCURISÉE : Fixer la graine rend le tri identique à chaque redémarrage du script
    random.seed(42)  
    random.shuffle(all_files)  
    logger.info("Shuffling Global achevé avec succès.")

    batches = [all_files[i:i + CHUNKS_PER_FILE] for i in range(0, total_files, CHUNKS_PER_FILE)]
    total_batches = len(batches)
    
    logger.info(f"Planification de {total_batches} fichiers maîtres. Analyse des fichiers déjà existants...")

    futures = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        for idx, batch_files in enumerate(batches):
            out_name = f"{prefix}_{idx:05d}.gz"
            out_path = os.path.join(dest_dir, out_name)
            
            # 🌟 REPRISE SÉCURISÉE : Si le fichier a déjà été généré lors d'un run précédent, on le saute
            if os.path.exists(out_path):
                continue
                
            f = executor.submit(pack_single_batch, batch_files, out_path)
            futures[f] = out_name

        total_to_process = len(futures)
        logger.info(f"Il reste {total_to_process} / {total_batches} fichiers à générer.")

        completed = 0
        for future in as_completed(futures):
            name = futures[future]
            success = future.result()
            completed += 1
            if completed % 50 == 0 or completed == total_to_process:
                logger.info(f"[PROGRESS] {prefix.upper()} : {completed}/{total_to_process} fichiers master générés.")

if __name__ == "__main__":
    SRC_TRAIN = "/lustre/fsn1/projects/rech/kwf/uzr96yg/leela/data/train"
    SRC_TEST = "/lustre/fsn1/projects/rech/kwf/uzr96yg/leela/data/test"
    
    DST_TRAIN = "/lustre/fsn1/projects/rech/kwf/uzr96yg/leela/data_packed/train"
    DST_TEST = "/lustre/fsn1/projects/rech/kwf/uzr96yg/leela/data_packed/test"
    
    logger.info("=== DÉMARRAGE DU PIPELINE DE CONSOLIDATION DU DATASET ===")
    process_partition(SRC_TEST, DST_TEST, "packed_test")
    process_partition(SRC_TRAIN, DST_TRAIN, "packed_train")
    logger.info("=== PIPELINE TERMINÉ AVEC SUCCÈS ===")