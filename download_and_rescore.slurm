#!/bin/bash
#SBATCH --job-name=download_and_rescore
#SBATCH --partition=prepost           # LA commande magique (Gratuit + Internet)
#SBATCH --account=kwf@cpu             # Votre compte (utilisez @cpu ou @v100 selon ce qui passe)
#SBATCH --time=10:00:00               # Jusqu'à 20h autorisées sur prepost
#SBATCH --nodes=1                     # 1 nœud
#SBATCH --ntasks-per-node=1           # 1 tâche
#SBATCH --cpus-per-task=8             # Demandez plusieurs cœurs pour accélérer le rescoring
#SBATCH --output=download_and_rescore.out
#SBATCH --error=download_and_rescore.err

# On charge l'environnement Python si nécessaire
module purge
module load python/3.12.2

# On se place dans le code
cd $WORK/lczero-training

echo "Begin downloading and rescoring on the fly..."

# On lance le script de préparation (qui nécessite Internet)
python download_and_rescore.py

echo "Downloading and rescoring over !"