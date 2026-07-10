#!/bin/bash
set -e

cd custom_ops
LAYER_NAME=${1:-"depthwise"}

echo "Compiling custom TensorFlow layer: '${LAYER_NAME}'"

# DYNAMICALLY LOCATE THE REAL SYSTEM CUDA RUNTIME HEADER
# This finds the file, filters out TensorFlow's local copies, and gets the directory name
CUDA_INCLUDE_DIR=$(find /usr/local/cuda* -name "cuda_runtime.h" 2>/dev/null | grep -v "tensorflow" | head -n 1)

if [ -z "$CUDA_INCLUDE_DIR" ]; then
    echo "Error: Could not find system cuda_runtime.h"
    exit 1
fi

CUDA_BASE_INCLUDE=$(dirname "$CUDA_INCLUDE_DIR")
echo "Targeting CUDA include directory: ${CUDA_BASE_INCLUDE}"

# 1. Intercept compiler flags cleanly into discrete array scopes
TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# 2. Compile the CUDA architecture code cleanly using nvcc (Targeting your RTX 5070 Laptop)
nvcc -c -o "${LAYER_NAME}.cu.o" "${LAYER_NAME}.cu.cc" \
  "${TF_CFLAGS[@]}" -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr \
  -gencode arch=compute_120,code=sm_120

# 3. Assemble and build the final module shared library via g++
g++ -shared "${LAYER_NAME}.cc" "${LAYER_NAME}.cu.o" -o "${LAYER_NAME}.so" \
  -fPIC "${TF_CFLAGS[@]}" "${TF_LFLAGS[@]}" -I"${CUDA_BASE_INCLUDE}" -O2

echo "Successfully generated ${LAYER_NAME}.so"