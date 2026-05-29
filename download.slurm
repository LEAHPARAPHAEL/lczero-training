#!/bin/bash
#SBATCH --job-name=download
#SBATCH --partition=prepost           # LA commande magique (Gratuit + Internet)
#SBATCH --account=kwf@v100             # Votre compte (utilisez @cpu ou @v100 selon ce qui passe)
#SBATCH --time=10:00:00               # Jusqu'à 20h autorisées sur prepost
#SBATCH --nodes=1                     # 1 nœud
#SBATCH --ntasks-per-node=1           # 1 tâche
#SBATCH --cpus-per-task=4             # Demandez plusieurs cœurs pour accélérer le rescoring
#SBATCH --output=download.out
#SBATCH --error=download.err

# On charge l'environnement Python si nécessaire
module purge
module load tensorflow-gpu/py3/2.16.1

# On se place dans le code
cd $WORK/lczero-training/tf

echo "Begin downloading !"

# On lance le script de préparation (qui nécessite Internet)
python download.py

echo "Downloading over !"