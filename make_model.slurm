#!/bin/bash
#SBATCH --job-name=lczero-training
#SBATCH --account=kwf@v100
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -C v100-32g
#SBATCH --output=logs/default_job.out
#SBATCH --error=logs/default_job.err

# --- SÉCURITÉ : Vérification de l'argument ---
if [ -z "$1" ]; then
    echo "Erreur : Aucun fichier de configuration n'a été fourni."
    echo "Utilisation : sbatch make_model.slurm <fichier_config.yaml>"
    exit 1
fi

CONFIG_FILE=$1
CONFIG_NAME=$(basename "${CONFIG_FILE%.*}")

# 1. Préparation de l'environnement
module purge
module load tensorflow-gpu/py3/2.16.1
export TF_USE_LEGACY_KERAS=1

mkdir -p logs/$CONFIG_NAME

cd $WORK/lczero-training/tf

echo "Creating empty model for : $CONFIG_NAME"

protoc -I=. --python_out=. proto/net.proto

python make_model.py --cfg $CONFIG_FILE > ../logs/$CONFIG_NAME/make.out 2> ../logs/$CONFIG_NAME/make.err

echo "Finished creating empty model for : $CONFIG_NAME !"