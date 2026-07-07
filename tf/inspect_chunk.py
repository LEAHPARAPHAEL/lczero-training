#!/usr/bin/env python3
import os
import gzip
import struct

# Configuration
FOLDER_PATH = "/lustre/fsn1/projects/rech/kwf/uzr96yg/leela/data/train"
TARGET_SAMPLES = 1000
RECORD_SIZE = 8396  # Vos fichiers sont au format V7 (8396 octets)

def analyze_adjudication():
    print("=== VÉRIFICATION DE L'HYPOTHÈSE D'ADJUDICATION ===")
    print(f"Recherche de {TARGET_SAMPLES} fichiers témoins se terminant abruptement...\n")

    samples_collected = 0
    count_adjudicated = 0
    count_max_length = 0
    count_both = 0
    count_natural_or_split = 0

    with os.scandir(FOLDER_PATH) as it:
        for entry in it:
            if samples_collected >= TARGET_SAMPLES:
                break
                
            if entry.is_file() and entry.name.endswith('.gz') and entry.name.startswith('training.'):
                try:
                    with gzip.open(entry.path, "rb") as f:
                        data = f.read()
                        
                    if len(data) < RECORD_SIZE:
                        continue
                        
                    total_records = len(data) // RECORD_SIZE
                    last_offset = (total_records - 1) * RECORD_SIZE
                    
                    # 1. Extraction du plies_left de la TOUTE DERNIÈRE position du fichier (offset 8304)
                    last_plies_left = struct.unpack("f", data[last_offset + 8304 : last_offset + 8308])[0]
                    
                    # On ne s'intéresse qu'aux fichiers qui s'arrêtent brutalement (ex: plies_left > 2.0)
                    if last_plies_left > 2.0:
                        samples_collected += 1
                        
                        # 2. Extraction de l'invariance_info de cette dernière position (offset 8278, 1 octet)
                        invariance_info = data[last_offset + 8278]
                        
                        # 3. Vérification des masques de bits (Bit 5 et Bit 4)
                        is_adjudicated = bool(invariance_info & 0x20)  # Bit 5 (32)
                        is_max_length = bool(invariance_info & 0x10)   # Bit 4 (16)
                        
                        if is_adjudicated and is_max_length:
                            count_both += 1
                        elif is_adjudicated:
                            count_adjudicated += 1
                        elif is_max_length:
                            count_max_length += 1
                        else:
                            count_natural_or_split += 1
                            
                        # Petit affichage de progression tous les 200 échantillons
                        if samples_collected % 200 == 0:
                            print(f"[Progression] {samples_collected}/{TARGET_SAMPLES} fichiers analysés...")
                            
                except Exception:
                    continue

    # --- AFFICHAGE DES RÉSULTATS ---
    print("\n" + "="*60)
    print(f" VERDICT SUR {samples_collected} FRAGMENTS INTERROMPUS")
    print("="*60)
    print(f"1. Game Adjudicated uniquement (Bit 5)    : {count_adjudicated}")
    print(f"2. Max Length Exceeded uniquement (Bit 4) : {count_max_length}")
    print(f"3. Les deux flags simultanément            : {count_both}")
    print(f"4. AUCUN FLAG ACTIVÉ (Hypothèse du Split) : {count_natural_or_split}")
    print("="*60)
    
    total_flags = count_adjudicated + count_max_length + count_both
    percentage = (total_flags / samples_collected) * 100
    
    print(f"\nConclusion : {percentage:.2f}% des fichiers interrompus possèdent un flag de coupure forcée.")
    if count_natural_or_split == 0:
        print(" -> 🌟 VOTRE HYPOTHÈSE EST VRAIE ! Les parties ne sont pas splitées entre les fichiers.")
    else:
        print(f" -> 💡 HYPOTHÈSE MIXTE : {count_natural_or_split} fichiers n'ont pas de flag, le fractionnement existe donc minoritairement.")

if __name__ == "__main__":
    analyze_adjudication()