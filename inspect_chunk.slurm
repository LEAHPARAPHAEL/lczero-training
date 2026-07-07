#!/bin/bash
#SBATCH --job-name=chain_leela
#SBATCH --partition=prepost           # 🌟 Identique à vos scripts de préparation
#SBATCH --account=kwf@v100            # Votre compte projet
#SBATCH --time=00:10:00               # ⏱️ Ajusté : 10 minutes maximum (S'exécutera très vite)
#SBATCH --nodes=1                     
#SBATCH --ntasks-per-node=1          
#SBATCH --cpus-per-task=1             # 1 seul cœur suffit largement pour ce scan linéaire
#SBATCH --hint=nomultithread          
#SBATCH --output=analyze_chains.out
#SBATCH --error=analyze_chains.err

# Nettoyage et chargement de Python (miroir de votre environnement standard)
module purge
module load python/3.12.7             # Aligné sur votre pack_dataset

# Déplacement dans votre répertoire de code
cd $WORK/lczero-training/tf

echo "Lancement de l'analyse de chaînage sur l'échantillon du dataset..."
python inspect_chunk.py
echo "Analyse terminée."