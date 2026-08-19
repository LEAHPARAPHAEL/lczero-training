#!/bin/bash
#SBATCH --job-name=lczero-saving
#SBATCH --account=kwf@v100
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -C v100-32g
#SBATCH --output=logs/default_job.out
#SBATCH --error=logs/default_job.err

# --- SÉCURITÉ : Vérification des arguments ---
if [ -z "$1" ]; then
    echo "Erreur : Aucun fichier de configuration n'a été fourni."
    echo "Utilisation : sbatch model_to_net.slurm <fichier_config.yaml> [step]"
    exit 1
fi

CONFIG_FILE=$1
CONFIG_NAME=$(basename "${CONFIG_FILE%.*}")

# Gestion de l'argument optionnel 'step'
STEP_ARG=""
if [ -n "$2" ]; then
    STEP_ARG="--step $2"
    echo "Cible configurée : Checkpoint step $2"
else
    echo "Aucun step spécifié : Sélection du dernier checkpoint par défaut"
fi

# 1. Préparation de l'environnement
module purge
module load tensorflow-gpu/py3/2.16.1
export TF_USE_LEGACY_KERAS=1

# 2. Création du dossier de logs spécifique
mkdir -p logs/$CONFIG_NAME

# 3. On se place dans le dossier du code source
cd $WORK/lczero-training/tf

echo "Saving .pb.gz for model : $CONFIG_NAME"

# 4. Compilation des protocoles et lancement
protoc -I=. --python_out=. proto/net.proto

# Lancement avec redirection dynamique des logs et passage du paramètre optionnel
python model_to_net.py --cfg "$CONFIG_FILE" $STEP_ARG > ../logs/"$CONFIG_NAME"/save.out 2> ../logs/"$CONFIG_NAME"/save.err

echo "Finished saving .pb.gz for model : $CONFIG_NAME !"