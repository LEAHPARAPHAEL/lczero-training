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

# 2. Interception des flags TensorFlow
TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# 3. Compilation CUDA
# Remplacement de -arch=native par sm_70 (V100) ou sm_80 (A100)
echo "Running nvcc..."
nvcc -c -o "${LAYER_NAME}.cu.o" "${LAYER_NAME}.cu.cc" \
  "${TF_CFLAGS[@]}" \
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

# 4. Assemblage final en bibliothèque partagée (.so)
echo "Running g++ linker..."
g++ -shared "${LAYER_NAME}.cc" "${LAYER_NAME}.cu.o" -o "${LAYER_NAME}.so" \
  -fPIC \
  "${TF_CFLAGS[@]}" \
  "${TF_LFLAGS[@]}" \
  -O2

echo "Successfully generated ${LAYER_NAME}.so"
echo "Compilation over !"


