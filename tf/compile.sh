#!/bin/bash

cd custom_ops
# Capture the first argument passed to the script, defaulting to "depthwise" if left blank
LAYER_NAME=${1:-"depthwise"}

echo "Compiling custom TensorFlow layer: '${LAYER_NAME}'"

# 1. Intercept compiler flags cleanly into discrete array scopes
TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# 2. Compile the CUDA architecture code cleanly using nvcc
nvcc -c -o "${LAYER_NAME}.cu.o" "${LAYER_NAME}.cu.cc" \
  "${TF_CFLAGS[@]}" -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr \
  -gencode arch=compute_70,code=sm_70 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90

# 3. Assemble and build the final module shared library via g++
g++ -shared "${LAYER_NAME}.cc" "${LAYER_NAME}.cu.o" -o "${LAYER_NAME}.so" \
  -fPIC "${TF_CFLAGS[@]}" "${TF_LFLAGS[@]}" -O2

echo "Successfully generated ${LAYER_NAME}.so"