#ifndef GATED_DEPTHWISE_H_
#define GATED_DEPTHWISE_H_

#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct GatedDepthwiseFunctor {
  void operator()(const Device& d, int batch_size, int total_c_half2, int groups,
                  T* output, const T* input, const T* weights, const T* gating, const T* biases,
                  int activation_mode);
};

template <typename Device, typename T>
struct GatedDepthwiseGradFunctor {
  void operator()(OpKernelContext* context, const Device& d, int batch_size, int total_c_half2, int groups,
                  T* d_input, T* d_weights, T* d_gating, T* d_biases,
                  const T* d_out, const T* input, const T* weights, const T* gating, const T* biases,
                  int activation_mode);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // GATED_DEPTHWISE_H_