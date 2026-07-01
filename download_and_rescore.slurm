#!/bin/bash
#SBATCH --job-name=download
#SBATCH --partition=prepost           # LA commande magique (Gratuit + Internet)
#SBATCH --account=kwf@v100             # Votre compte (utilisez @cpu ou @v100 selon ce qui passe)
#SBATCH --time=20:00:00               # Jusqu'à 20h autorisées sur prepost
#SBATCH --nodes=1                     # 1 nœud
#SBATCH --ntasks-per-node=1           # 1 tâche
#SBATCH --cpus-per-task=16            # Demandez plusieurs cœurs pour accélérer le rescoring
#SBATCH --output=download.out
#SBATCH --error=download.err
#SBATCH --hint=nomultithread
# On charge l'environnement Python si nécessaire
module purge
module load tensorflow-gpu/py3/2.16.1

# On se place dans le code
cd $WORK/lczero-training/tf

echo "Begin downloading and rescoring on the fly !"

# On lance le script de préparation (qui nécessite Internet)
python download_and_rescore.py

echo "Downloading and rescoring over !"