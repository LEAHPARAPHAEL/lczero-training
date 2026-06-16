#!/bin/bash
#SBATCH --job-name=compile
#SBATCH --partition=compil          
#SBATCH --account=kwf@v100             
#SBATCH --time=00:30:00               
#SBATCH --nodes=1                     
#SBATCH --ntasks-per-node=1          
#SBATCH --cpus-per-task=8             
#SBATCH --output=logs/compile.out
#SBATCH --error=logs/compile.err

echo "Starting compilation..."

# 1. On nettoie tout
module purge

# 2. ON CHARGE LES PRÉREQUIS EXACTS DEMANDÉS PAR TENSORFLOW (Dans le bon ordre)
module load gcc/12.2.0
module load cuda/12.2.0

# 3. On charge enfin TensorFlow
module load tensorflow-gpu/py3/2.16.1
export TF_USE_LEGACY_KERAS=1

# Sécurité : création du dossier de log s'il n'existe pas
mkdir -p logs

# On s'assure d'être au bon endroit
cd $WORK/lczero-training/tf/custom_ops 2>/dev/null || cd tf/custom_ops

LAYER_NAME=${1:-"depthwise"}
echo "Compiling custom TensorFlow layer: '${LAYER_NAME}'"

echo "Extracting TensorFlow and Eigen flags..."
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# L'ASTUCE : On récupère le chemin vers le dossier site-packages de TensorFlow
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

# On crée un alias d'inclusion pour Eigen (souvent situé dans external/eigen_archive)
EIGEN_INC="${TF_INC}/external/eigen_archive"

# 2. Compilation CUDA avec nvcc
echo "Running nvcc..."
nvcc -c -o "${LAYER_NAME}.cu.o" "${LAYER_NAME}.cu.cc" \
  "${TF_CFLAGS[@]}" \
  -I"${TF_INC}" -I"${EIGEN_INC}" \
  -D GOOGLE_CUDA=1 \
  -x cu \
  -Xcompiler -fPIC \
  --expt-relaxed-constexpr \
  -gencode arch=compute_70,code=sm_70 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90

# 3. Assemblage final avec g++
echo "Running g++ linker..."
g++ -shared "${LAYER_NAME}.cc" "${LAYER_NAME}.cu.o" -o "${LAYER_NAME}.so" \
  -fPIC \
  "${TF_CFLAGS[@]}" \
  -I"${TF_INC}" -I"${EIGEN_INC}" \
  "${TF_LFLAGS[@]}" \
  -O2
  
echo "Successfully generated ${LAYER_NAME}.so"
echo "Compilation over !"


