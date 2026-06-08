#!/bin/bash
#SBATCH --job-name=lczero-training
#SBATCH --account=kwf@v100
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -C v100-32g
#SBATCH --output=logs/default_job.out
#SBATCH --error=logs/default_job.err

if [ -z "$1" ]; then
    echo "Erreur : Aucun fichier de configuration n'a été fourni."
    echo "Utilisation : sbatch train.slurm <fichier_config.yaml>"
    exit 1
fi

CONFIG_FILE=$1
CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)

module purge
module load tensorflow-gpu/py3/2.16.1
export TF_USE_LEGACY_KERAS=1

mkdir -p logs/$CONFIG_NAME

cd $WORK/lczero-training/tf

echo "Début de l'entraînement avec la configuration : $CONFIG_NAME"

protoc -I=. --python_out=. proto/net.proto

python train.py --cfg $CONFIG_FILE > ../logs/$CONFIG_NAME/run.out 2> ../logs/$CONFIG_NAME/run.err

echo "Entraînement terminé pour $CONFIG_NAME !"
