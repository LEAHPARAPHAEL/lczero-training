#!/bin/bash
#SBATCH --job-name=pack_leela
#SBATCH --partition=prepost      
#SBATCH --account=kwf@v100            
#SBATCH --time=20:00:00              
#SBATCH --nodes=1                     
#SBATCH --ntasks-per-node=1          
#SBATCH --cpus-per-task=40            
#SBATCH --hint=nomultithread          
#SBATCH --output=pack_dataset.out
#SBATCH --error=pack_dataset.err

# Nettoyage de l'environnement et chargement de Python
module purge
module load python/3.12.7

# Déplacement dans votre répertoire de code
cd $WORK/lczero-training/tf

echo "Démarrage du processus global de repaquettage..."
python pack_dataset.py
echo "Processus de repaquettage terminé !"