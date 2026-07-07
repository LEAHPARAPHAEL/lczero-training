import sys
import gzip
import struct

# Modifiez le chemin ici avec le fichier que vous voulez inspecter
FILENAME = "/lustre/fsn1/projects/rech/kwf/uzr96yg/leela/data/train/training.1597466789.gz"
RECORD_SIZE = 8396  # Taille stricte d'un enregistrement binaire V7

def inspect_file(filename):
    try:
        with gzip.open(filename, "rb") as f:
            data = f.read()
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
        return

    total_records = len(data) // RECORD_SIZE
    print(f"=== INSPECTION : {filename} ===")
    print(f"Nombre total de positions (records) dans le fichier : {total_records}")
    print("-" * 55)
    print(f"{'Index':<7} | {'Plies Restants (plies_left)':<28} | {'Règle des 50 coups':<12}")
    print("-" * 55)

    # Affichage des positions
    for i in range(total_records):
        offset = i * RECORD_SIZE
        
        # 1. Extraction de plies_left (Flottant stocké à l'offset 8304)
        plies_left = struct.unpack("f", data[offset + 8304 : offset + 8308])[0]
        
        # 2. Extraction du compteur des 50 coups (Octet stocké à l'offset 8277)
        rule50 = data[offset + 8277]
        
        # On n'affiche que les 15 premières et 15 dernières positions pour ne pas saturer l'écran
        if i < 15 or i >= total_records - 15:
            print(f"{i:<7} | {plies_left:<28.1f} | {rule50:<12d}")
        elif i == 15:
            print(f"{'...':<7} | {'... (les positions intermédiaires sont masquées) ...':<28} | {'...':<12}")

if __name__ == "__main__":
    # Permet de passer un fichier en argument système si désiré : python inspect_chunk.py fichier.gz
    target_file = sys.argv[1] if len(sys.argv) > 1 else FILENAME
    inspect_file(target_file)