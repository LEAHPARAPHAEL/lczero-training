#!/bin/bash
#SBATCH --job-name=lczero-training
#SBATCH --account=kwf@v100            # Votre compte d'heures
#SBATCH --time=20:00:00               # Temps alloué
#SBATCH --nodes=1                     # 1 nœud
#SBATCH --ntasks-per-node=1           # 1 tâche
#SBATCH --cpus-per-task=10            # 10 CPU par GPU
#SBATCH --gres=gpu:1                  # 1 GPU V100
#SBATCH -C v100-32g                   # Cartes avec 32 Go de VRAM
#SBATCH --output=log_train_%j.out     # Fichier de log standard
#SBATCH --error=err_train_%j.err      # Fichier d'erreurs

# --- SÉCURITÉ : Vérification de l'argument ---
if [ -z "$1" ]; then
    echo "Erreur : Aucun fichier de configuration n'a été fourni."
    echo "Utilisation : sbatch train.slurm <fichier_config.yaml>"
    exit 1
fi

CONFIG_FILE=$1

# 1. Environnement
module purge
module load tensorflow-gpu/py3/2.16.1
#export TF_USE_LEGACY_KERAS=1

# 2. On se place dans le dossier du code source (SUR LE WORK)
cd $WORK/lczero-training

echo "Début de l'entraînement avec la configuration : $CONFIG_FILE"

# 3. Lancement du code
python train.py --cfg $CONFIG_FILE

echo "Entraînement terminé !"