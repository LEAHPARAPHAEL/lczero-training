#!/usr/bin/env python3
import os
import sys
import gzip
import struct

RECORD_SIZE = 8396  # Taille d'une position binaire V7

def inspect_clean_chains(folder_path, window_seconds=300):
    print("=== Analyse de Chaînage Haute Performance (Sans Scan de Dossier) ===")
    print(f"Dossier cible : {folder_path}\n")
    
    # Étape 1 : Attraper quelques fichiers témoins INSTANTANÉMENT
    base_files = []
    with os.scandir(folder_path) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith('.gz') and entry.name.startswith('training.'):
                try:
                    parts = entry.name.split('.')
                    ts = int(parts[1])
                    base_files.append((entry.name, ts))
                    if len(base_files) >= 5: # 5 témoins suffisent pour valider la structure
                        break
                except ValueError:
                    continue

    if not base_files:
        print("❌ Erreur : Aucun fichier témoin trouvé.")
        return

    print(f"Trouvé {len(base_files)} fichiers témoins instantanément.")
    print(f"Lancement des recherches par sauts de métadonnées (Fenêtre: ±{window_seconds}s)...")

    for base_name, base_ts in base_files:
        print("\n" + "="*75)
        print(f"🔍 ANALYSE AUTOUR DE : {base_name} (Timestamp: {base_ts})")
        print("="*75)
        
        # Étape 2 : Vérification d'existence directe seconde par seconde (Zéro Scan)
        neighbor_files = []
        for t in range(base_ts - window_seconds, base_ts + window_seconds + 1):
            test_name = f"training.{t}.gz"
            test_path = os.path.join(folder_path, test_name)
            
            if os.path.exists(test_path): # Appel O(1) ultra-rapide sur Lustre
                neighbor_files.append((test_name, test_path, t))
                
        print(f"↳ {len(neighbor_files)} fichiers actifs découverts à proximité immédiate.")
        if len(neighbor_files) <= 1:
            print("↳ ❌ Aucun voisin temporel. Ce fragment est isolé.")
            continue
            
        # Tri chronologique des voisins trouvés
        neighbor_files.sort(key=lambda x: x[2])
        
        # Étape 3 : Lecture des métadonnées de plies des voisins
        chunk_meta = []
        for name, path, ts in neighbor_files:
            try:
                with gzip.open(path, "rb") as f:
                    data = f.read()
                if len(data) < RECORD_SIZE:
                    continue
                total_records = len(data) // RECORD_SIZE
                start_ply = struct.unpack("f", data[8304:8308])[0]
                end_ply = struct.unpack("f", data[(total_records - 1) * RECORD_SIZE + 8304 : (total_records - 1) * RECORD_SIZE + 8308])[0]
                
                chunk_meta.append({
                    'filename': name,
                    'total_records': total_records,
                    'start_ply': round(start_ply, 1),
                    'end_ply': round(end_ply, 1),
                    'timestamp': ts
                })
            except Exception:
                continue

        # Étape 4 : Détection des connexions de plies consécutives
        links_found = 0
        for idx in range(len(chunk_meta) - 1):
            a = chunk_meta[idx]
            target_ply = round(a['end_ply'] - 1.0, 1) # Le coup suivant doit avoir -1 pli restant
            
            for b in chunk_meta[idx+1:]:
                if abs(b['start_ply'] - target_ply) < 0.1:
                    dt = b['timestamp'] - a['timestamp']
                    print(f"➔ 🌟 LIAISON DE CONTINUITÉ DÉTECTÉE (Écart: {dt}s) :")
                    print(f"   ├─ Fragment Précédent : {a['filename']} ({a['total_records']} pos) | Plies: {a['start_ply']} ➔ {a['end_ply']}")
                    print(f"   └─ Fragment Suivant   : {b['filename']} ({b['total_records']} pos) | Plies: {b['start_ply']} ➔ {b['end_ply']}")
                    links_found += 1
                    
        if links_found == 0:
            print("➔ ❌ Aucune liaison de plies trouvée dans ce bloc. Les fragments sont indépendants.")

if __name__ == "__main__":
    inspect_clean_chains("/lustre/fsn1/projects/rech/kwf/uzr96yg/leela/data/train")