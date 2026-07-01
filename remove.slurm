#!/bin/bash
#SBATCH --job-name=cleanup_param
#SBATCH --partition=prepost           # LA commande magique (Gratuit + Internet)
#SBATCH --account=kwf@v100            # Votre compte
#SBATCH --time=20:00:00               # Jusqu'à 20h autorisées sur prepost
#SBATCH --nodes=1                     # 1 nœud
#SBATCH --ntasks-per-node=1           # 1 tâche
#SBATCH --cpus-per-task=16            # On utilise ces 16 cœurs pour paralléliser !
#SBATCH --output=cleanup_%j.out
#SBATCH --error=cleanup_%j.err

# ==============================================================================
# RÉCUPÉRATION DU PARAMÈTRE
# ==============================================================================
TARGET_DIR="$1"

# ==============================================================================
# SÉCURITÉ ANTI-CATASTROPHE
# ==============================================================================
# Vérifie si l'utilisateur a bien fourni un argument
if [ -z "$TARGET_DIR" ]; then
    echo "ERREUR : Vous devez spécifier le dossier à supprimer." >&2
    echo "Usage : sbatch $0 <chemin_du_dossier>" >&2
    exit 1
fi

# Convertit en chemin absolu (au cas où un chemin relatif a été donné)
TARGET_DIR=$(realpath "$TARGET_DIR")

# Sécurité critique pour Jean-Zay : on s'assure qu'on est BIEN dans le SCRATCH 
# et qu'on n'essaie pas de supprimer la racine du SCRATCH, de HOME ou de WORK.
if [[ "$TARGET_DIR" != "$SCRATCH"* ]] || [ "$TARGET_DIR" == "$SCRATCH" ]; then
    echo "ERREUR SÉCURITÉ : Le dossier cible doit être un sous-dossier de $SCRATCH" >&2
    echo "Chemin fourni : $TARGET_DIR" >&2
    exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo "ERREUR : Le dossier '$TARGET_DIR' n'existe pas." >&2
    exit 1
fi

echo "=== DÉBUT DU NETTOYAGE PARAMÉTRÉ ==="
echo "Dossier cible : $TARGET_DIR"
echo "Nombre de cœurs : $SLURM_CPUS_PER_TASK"
echo "Heure de début : $(date)"
echo "-----------------------------------------"

# ==============================================================================
# NETTOYAGE EN PARALLÈLE
# ==============================================================================
echo "Étape 1 : Fichiers..."
find "$TARGET_DIR" -type f -print0 | xargs -0 -P "$SLURM_CPUS_PER_TASK" -n 2000 rm -f

echo "Étape 2 : Liens symboliques..."
find "$TARGET_DIR" -type l -print0 | xargs -0 -P "$SLURM_CPUS_PER_TASK" -n 2000 rm -f

echo "Étape 3 : Dossiers vides..."
find "$TARGET_DIR" -depth -type d -delete

echo "Étape 4 : Dossier parent..."
rm -rf "$TARGET_DIR"

echo "-----------------------------------------"
echo "=== NETTOYAGE TERMINÉ AVEC SUCCÈS ==="
echo "Heure de fin : $(date)"