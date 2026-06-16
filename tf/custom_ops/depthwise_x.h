#ifndef DEPTHWISE_X_H_
#define DEPTHWISE_X_H_

#define EIGEN_USE_GPU
//#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct DepthwiseXFunctor {
  void operator()(const Device& d, int batch_size, int total_c_half2,
                  T* output, const T* input, const T* weights, const T* recomb, const T* biases,
                  int activation_mode);
};

template <typename Device, typename T>
struct DepthwiseXGradFunctor {
  void operator()(const Device& d, int batch_size, int total_c_half2,
                  T* d_input, T* d_weights, T* d_recomb, T* d_biases,
                  const T* d_out, const T* input, const T* weights, const T* recomb, const T* biases,
                  int activation_mode);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // DEPTHWISE_X_H_