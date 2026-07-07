#!/usr/bin/env python3
import os
import sys
import glob
import gzip
import struct

RECORD_SIZE = 8396  # Taille d'une position V7

def analyze_chunks(folder_path, max_files=5000):
    print(f"=== ANALYSE DE CHAÎNAGE DES ENREGISTREMENTS ===")
    print(f"Scan des fichiers dans : {folder_path}")
    
    # Récupération et tri chronologique des fichiers d'origine
    files = sorted(glob.glob(os.path.join(folder_path, "training.*.gz")))
    if not files:
        print("Erreur : Aucun fichier trouvé dans ce répertoire.")
        return
        
    print(f"Trouvé {len(files)} fichiers au total. Échantillonnage des {max_files} premiers...")
    files = files[:max_files]
    
    chunk_meta = []
    
    for filepath in files:
        try:
            with gzip.open(filepath, "rb") as f:
                data = f.read()
            if len(data) < RECORD_SIZE:
                continue
                
            total_records = len(data) // RECORD_SIZE
            
            # Lecture du premier plies_left (index 0)
            start_ply = struct.unpack("f", data[8304:8308])[0]
            # Lecture du dernier plies_left (index total_records - 1)
            end_ply = struct.unpack("f", data[(total_records - 1) * RECORD_SIZE + 8304 : (total_records - 1) * RECORD_SIZE + 8308])[0]
            
            # Extraction du timestamp Unix depuis le nom du fichier
            base = os.path.basename(filepath)
            timestamp = int(base.split('.')[1])
            
            chunk_meta.append({
                'filename': base,
                'total_records': total_records,
                'start_ply': round(start_ply, 1),
                'end_ply': round(end_ply, 1),
                'timestamp': timestamp
            })
        except Exception:
            continue

    print(f"Analyse brute terminée ({len(chunk_meta)} fichiers valides en mémoire).")
    print("Recherche de connexions mathématiques intra-partie...")
    
    links_found = []
    
    # Double boucle pour chercher si la fin du fichier A se connecte au début du fichier B
    for a in chunk_meta:
        # Si le fichier A se termine à 184.9, le coup suivant (fichier B) DOIT commencer à 183.9
        target_start = round(a['end_ply'] - 1.0, 1)
        
        for b in chunk_meta:
            if a['filename'] == b['filename']:
                continue
                
            # Vérification de la continuité du pli
            if abs(b['start_ply'] - target_start) < 0.1:
                # Vérification de la cohérence de temps (le fichier B doit être contemporain ou futur, max 1 heure après)
                time_diff = b['timestamp'] - a['timestamp']
                if 0 <= time_diff <= 3600:
                    links_found.append((a, b, time_diff))
                    
    # --- Affichage des Résultats ---
    if not links_found:
        print("\n[RÉSULTAT] ❌ Aucune connexion trouvée entre les chunks.")
        print("Conclusion : Leela Chess Zero n'écrit PAS la suite des parties dans des chunks consécutifs identifiables.")
        print("Chaque fichier de 85 positions est un isolat indépendant, le reste de la partie est jeté ou dispersé de manière cryptique.")
    else:
        print(f"\n[RÉSULTAT] 🌟 Connexions trouvées ! {len(links_found)} liaisons détectées.")
        print("Cela prouve que les parties longues sont bien découpées et réparties sur plusieurs fichiers !")
        print("-" * 85)
        for a, b, dt in links_found[:15]:  # On affiche les 15 premières liaisons
            print(f"Liaison détectée (Écart temporel : {dt} secondes) :")
            print(f"  └─ Fichier Source A : {a['filename']} ({a['total_records']} pos) | Plies: {a['start_ply']} -> {a['end_ply']}")
            print(f"  └─ Fichier Suite B  : {b['filename']} ({b['total_records']} pos) | Plies: {b['start_ply']} -> {b['end_ply']}")
            print("-" * 85)

if __name__ == "__main__":
    # Par défaut, scanne le dossier d'entraînement d'origine
    target_folder = "/lustre/fsn1/projects/rech/kwf/uzr96yg/leela/data/train"
    if len(sys.argv) > 1:
        target_folder = sys.argv[1]
    analyze_chunks(target_folder)