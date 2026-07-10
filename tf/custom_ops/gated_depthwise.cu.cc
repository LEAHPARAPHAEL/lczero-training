#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "gated_depthwise.h"
#include <cuda_fp16.h>

namespace tensorflow {
namespace functor {

using GPUDevice = Eigen::GpuDevice;

// ============================================================================
// CUDA DEVICE UTILITIES
// ============================================================================
__device__ __forceinline__ float mishActivate(float el) {
    auto e = __expf(el);
    auto n = e * e + 2.0f * e;
    auto d = __fdividef(el, n + 2.0f);
    return (el <= -0.6f) ? (n * d) : (el - 2.0f * d);
}

__device__ __forceinline__ float mishGradient(float el, float d_out) {
    float sp = (el > 20.0f) ? el : __logf(1.0f + __expf(el));
    float exp_2sp = __expf(2.0f * sp);
    float t = (sp > 20.0f) ? 1.0f : __fdividef(exp_2sp - 1.0f, exp_2sp + 1.0f);
    float sig = __fdividef(1.0f, 1.0f + __expf(-el));
    
    float grad_factor = t + el * (1.0f - t * t) * sig;
    return d_out * grad_factor;
}

__device__ __forceinline__ half2 get_input_half2_nhwc_safe(const half2* input, int n, int h, int w, int c, int total_c) {
    if (h < 0 || h >= 8 || w < 0 || w >= 8) {
        return __float22half2_rn(make_float2(0.0f, 0.0f));
    }
    return input[(n * 64 * total_c) + (h * 8 * total_c) + (w * total_c) + c];
}

__device__ __forceinline__ void atomicAdd_input_safe(half2* d_input, int n, int h, int w, int c, int total_c, half2 val) {
    if (h >= 0 && h < 8 && w >= 0 && w < 8) {
        atomicAdd(&d_input[(n * 64 * total_c) + (h * 8 * total_c) + (w * total_c) + c], val);
    }
}

// ============================================================================
// FORWARD PASS GPU KERNEL
// ============================================================================
__global__ void GatedDepthwiseKernelNHWC_fp16(int total_c_half2, int groups, half2* output, 
                                              const half2* input, const half2* weights, 
                                              const half* gating, const half2* biases,
                                              int activation_mode) {
#if __CUDA_ARCH__ >= 700 
    int c_half2 = blockIdx.y * blockDim.x + threadIdx.x;
    if (c_half2 >= total_c_half2) return;

    int w = threadIdx.y;
    int h = blockIdx.z;
    int n = blockIdx.x;

    int total_channels = total_c_half2 * 2;
    int channels_per_group = total_channels / groups;
    
    int g1 = (c_half2 * 2) / channels_per_group;
    int g2 = (c_half2 * 2 + 1) / channels_per_group;

    half2 sum = __float22half2_rn(make_float2(0.0f, 0.0f));
    int gating_batch_offset = n * 25 * groups;

    int abs_h_input = h - 2;
    int abs_w_input = w - 2;

    #pragma unroll
    for(int i = 0; i < 25; ++i) {
        half2 base_w = weights[i * total_c_half2 + c_half2];
        
        half gate1 = gating[gating_batch_offset + i * groups + g1];
        half gate2 = gating[gating_batch_offset + i * groups + g2];
        half2 gate_h2; gate_h2.x = gate1; gate_h2.y = gate2;

        half2 eff_w = __hmul2(base_w, gate_h2);

        int dr = i / 5;
        int dc = i % 5;
        
        half2 in_val = get_input_half2_nhwc_safe(input, n, abs_h_input + dr, abs_w_input + dc, c_half2, total_c_half2);
        sum = __hfma2(eff_w, in_val, sum);
    }

    sum = __hadd2(sum, biases[c_half2]);
    
    if (activation_mode == 1) {
        float2 sum_f32 = __half22float2(sum);
        sum_f32.x = mishActivate(sum_f32.x);
        sum_f32.y = mishActivate(sum_f32.y);
        sum = __float22half2_rn(sum_f32);
    }

    output[(n * 64 * total_c_half2) + (h * 8 * total_c_half2) + (w * total_c_half2) + c_half2] = sum;
#endif
}

// ============================================================================
// SPLIT BACKWARD PASS KERNEL 1: DATA GRADIENTS & ACTIVATION EXTRACTION
// ============================================================================
__global__ void GatedDepthwiseBackwardDataKernel(int total_c_half2, int groups,
                                                 half2* d_input, half2* d_biases, half2* d_act_out,
                                                 const half2* d_out, const half2* input, const half2* weights, 
                                                 const half* gating, const half2* biases, int activation_mode) {
#if __CUDA_ARCH__ >= 700
    int c_half2 = blockIdx.y * blockDim.x + threadIdx.x;
    if (c_half2 >= total_c_half2) return;

    int w = threadIdx.y;
    int h = blockIdx.z;
    int n = blockIdx.x;

    int total_channels = total_c_half2 * 2;
    int channels_per_group = total_channels / groups;
    int g1 = (c_half2 * 2) / channels_per_group;
    int g2 = (c_half2 * 2 + 1) / channels_per_group;

    half2 sum = __float22half2_rn(make_float2(0.0f, 0.0f));
    int gating_batch_offset = n * 25 * groups;

    half2 base_w[25];
    half2 gate_h2[25];

    int abs_h_input = h - 2;
    int abs_w_input = w - 2;

    #pragma unroll
    for (int i = 0; i < 25; ++i) {
        base_w[i] = weights[i * total_c_half2 + c_half2];
        gate_h2[i].x = gating[gating_batch_offset + i * groups + g1];
        gate_h2[i].y = gating[gating_batch_offset + i * groups + g2];

        int dr = i / 5;
        int dc = i % 5;
        half2 feat = get_input_half2_nhwc_safe(input, n, abs_h_input + dr, abs_w_input + dc, c_half2, total_c_half2);
        sum = __hfma2(__hmul2(base_w[i], gate_h2[i]), feat, sum);
    }
    sum = __hadd2(sum, biases[c_half2]);

    int out_index = (n * 64 * total_c_half2) + (h * 8 * total_c_half2) + (w * total_c_half2) + c_half2;
    half2 d_out_val = d_out[out_index];
    half2 d_act = (activation_mode == 1) ? 
        __float22half2_rn(make_float2(mishGradient(__half22float2(sum).x, __half22float2(d_out_val).x),
                                      mishGradient(__half22float2(sum).y, __half22float2(d_out_val).y))) : d_out_val;

    atomicAdd(&(d_biases[c_half2]), d_act);
    d_act_out[out_index] = d_act;

    #pragma unroll
    for (int i = 0; i < 25; ++i) {
        half2 eff_w = __hmul2(base_w[i], gate_h2[i]);
        int dr = i / 5;
        int dc = i % 5;
        atomicAdd_input_safe(d_input, n, abs_h_input + dr, abs_w_input + dc, c_half2, total_c_half2, __hmul2(d_act, eff_w));
    }
#endif
}

// ============================================================================
// SPLIT BACKWARD PASS KERNEL 2: CONTENTION-FREE WEIGHTS & GATING REDUCTION
// ============================================================================
__global__ void GatedDepthwiseBackwardWeightsKernel(int total_c_half2, int groups,
                                                    half2* d_weights, half* d_gating,
                                                    const half2* d_act_in, const half2* input,
                                                    const half2* weights, const half* gating) {
#if __CUDA_ARCH__ >= 700
    // Shared structure mapping channels and board height structures 
    __shared__ half2 sh_sum[8][16];

    int c_half2 = blockIdx.y * blockDim.x + threadIdx.x;
    int h = threadIdx.y;
    int n = blockIdx.x;
    int i = blockIdx.z;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    half2 thread_sum = __float22half2_rn(make_float2(0.0f, 0.0f));

    if (c_half2 < total_c_half2) {
        int dr = i / 5;
        int dc = i % 5;
        int abs_h_input = h - 2;

        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            int out_index = (n * 64 * total_c_half2) + (h * 8 * total_c_half2) + (w * total_c_half2) + c_half2;
            half2 d_act = d_act_in[out_index];
            half2 feat = get_input_half2_nhwc_safe(input, n, abs_h_input + dr, w - 2 + dc, c_half2, total_c_half2);
            thread_sum = __hfma2(d_act, feat, thread_sum);
        }
    }

    sh_sum[ty][tx] = thread_sum;
    __syncthreads();

    // Sum row allocations inside block shared layout
    if (ty == 0 && c_half2 < total_c_half2) {
        half2 block_sum = __float22half2_rn(make_float2(0.0f, 0.0f));
        #pragma unroll
        for (int row = 0; row < 8; ++row) {
            block_sum = __hadd2(block_sum, sh_sum[row][tx]);
        }

        int total_channels = total_c_half2 * 2;
        int channels_per_group = total_channels / groups;
        int g1 = (c_half2 * 2) / channels_per_group;
        int g2 = (c_half2 * 2 + 1) / channels_per_group;

        int gating_batch_offset = n * 25 * groups;

        // 1. Gradients w.r.t Static Base Weights (Low-contention across batch size elements)
        half gate1 = gating[gating_batch_offset + i * groups + g1];
        half gate2 = gating[gating_batch_offset + i * groups + g2];
        half2 gate_h2; gate_h2.x = gate1; gate_h2.y = gate2;

        atomicAdd(&(d_weights[i * total_c_half2 + c_half2]), __hmul2(block_sum, gate_h2));

        // 2. Gradients w.r.t Channel Gating Elements
        half2 base_w = weights[i * total_c_half2 + c_half2];
        half2 d_gate_pair = __hmul2(block_sum, base_w);

        atomicAdd(&(d_gating[gating_batch_offset + i * groups + g1]), d_gate_pair.x);
        atomicAdd(&(d_gating[gating_batch_offset + i * groups + g2]), d_gate_pair.y);
    }
#endif
}

// ============================================================================
// FUNCTOR DISPATCH METHODS
// ============================================================================
template <typename T>
struct GatedDepthwiseFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int batch_size, int total_c_half2, int groups,
                  T* output, const T* input, const T* weights, const T* gating, const T* biases,
                  int activation_mode) {
    dim3 grid(batch_size, (total_c_half2 + 31) / 32, 8);
    dim3 block(32, 8, 1);

    GatedDepthwiseKernelNHWC_fp16<<<grid, block, 0, d.stream()>>>(
        total_c_half2, groups, reinterpret_cast<half2*>(output), 
        reinterpret_cast<const half2*>(input), reinterpret_cast<const half2*>(weights), 
        reinterpret_cast<const half*>(gating), reinterpret_cast<const half2*>(biases),
        activation_mode
    );
  }
};

template <typename T>
struct GatedDepthwiseGradFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context, const GPUDevice& d, int batch_size, int total_c_half2, int groups,
                  T* d_input, T* d_weights, T* d_gating, T* d_biases,
                  const T* d_out, const T* input, const T* weights, const T* gating, const T* biases,
                  int activation_mode) {
    
    // Allocate local scratchpad workspace to share d_act vectors between kernels without allocation penalties
    Tensor d_act_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(
        DataTypeToEnum<T>::value,
        TensorShape({batch_size, 8, 8, total_c_half2 * 2}),
        &d_act_tensor));

    T* d_act_ptr = d_act_tensor.flat<T>().data();

    // Launch Data Kernel Path 
    dim3 grid_data(batch_size, (total_c_half2 + 31) / 32, 8);
    dim3 block_data(32, 8, 1);

    GatedDepthwiseBackwardDataKernel<<<grid_data, block_data, 0, d.stream()>>>(
        total_c_half2, groups,
        reinterpret_cast<half2*>(d_input), reinterpret_cast<half2*>(d_biases), reinterpret_cast<half2*>(d_act_ptr),
        reinterpret_cast<const half2*>(d_out), reinterpret_cast<const half2*>(input), 
        reinterpret_cast<const half2*>(weights), reinterpret_cast<const half*>(gating), 
        reinterpret_cast<const half2*>(biases), activation_mode
    );

    // Launch Optimized Weights Kernel Path
    dim3 grid_weight(batch_size, (total_c_half2 + 15) / 16, 25);
    dim3 block_weight(16, 8, 1);

    GatedDepthwiseBackwardWeightsKernel<<<grid_weight, block_weight, 0, d.stream()>>>(
        total_c_half2, groups,
        reinterpret_cast<half2*>(d_weights), reinterpret_cast<half*>(d_gating),
        reinterpret_cast<const half2*>(d_act_ptr), reinterpret_cast<const half2*>(input),
        reinterpret_cast<const half2*>(weights), reinterpret_cast<const half*>(gating)
    );
  }
};

template struct GatedDepthwiseFunctor<GPUDevice, Eigen::half>;
template struct GatedDepthwiseGradFunctor<GPUDevice, Eigen::half>;

}  // namespace functor
}  // namespace tensorflow
#endif  // GOOGLE_CUDA