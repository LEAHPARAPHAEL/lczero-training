#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "depthwise.h"
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
// FORWARD PASS GPU KERNEL (PARAMETERIZED ACTIVATION)
// ============================================================================
__global__ void DepthwiseKernelNHWC_fp16(int total_c_half2, half2* output, const half2* input, 
                                         const half2* weights, const half2* biases,
                                         int rook_threshold, int bishop_threshold, int knight_threshold,
                                         int activation_mode) {
#if __CUDA_ARCH__ >= 700 
    int c_half2 = blockIdx.y * blockDim.x + threadIdx.x;
    if (c_half2 >= total_c_half2) return;

    int w = threadIdx.y;
    int h = blockIdx.z;
    int n = blockIdx.x;

    half2 w_h2[9];
    #pragma unroll
    for(int i = 0; i < 9; ++i) {
        w_h2[i] = weights[i * total_c_half2 + c_half2];
    }
    half2 b_h2 = biases[c_half2];

    int abs_h_input = h - 2;
    int abs_w_input = w - 2;

    half2 sum = __float22half2_rn(make_float2(0.0f, 0.0f));

    if (2 * c_half2 < rook_threshold) { 
        sum = __hfma2(w_h2[0], get_input_half2_nhwc_safe(input, n, abs_h_input,     abs_w_input + 2, c_half2, total_c_half2), sum);
        sum = __hfma2(w_h2[1], get_input_half2_nhwc_safe(input, n, abs_h_input + 1, abs_w_input + 2, c_half2, total_c_half2), sum);
        sum = __hfma2(w_h2[2], get_input_half2_nhwc_safe(input, n, abs_h_input + 2, abs_w_input,     c_half2, total_c_half2), sum);
        sum = __hfma2(w_h2[3], get_input_half2_nhwc_safe(input, n, abs_h_input + 2, abs_w_input + 1, c_half2, total_c_half2), sum);
        sum = __hfma2(w_h2[4], get_input_half2_nhwc_safe(input, n, abs_h_input + 2, abs_w_input + 2, c_half2, total_c_half2), sum);
        sum = __hfma2(w_h2[5], get_input_half2_nhwc_safe(input, n, abs_h_input + 2, abs_w_input + 3, c_half2, total_c_half2), sum);
        sum = __hfma2(w_h2[6], get_input_half2_nhwc_safe(input, n, abs_h_input + 2, abs_w_input + 4, c_half2, total_c_half2), sum);
        sum = __hfma2(w_h2[7], get_input_half2_nhwc_safe(input, n, abs_h_input + 3, abs_w_input + 2, c_half2, total_c_half2), sum);
        sum = __hfma2(w_h2[8], get_input_half2_nhwc_safe(input, n, abs_h_input + 4, abs_w_input + 2, c_half2, total_c_half2), sum);
    }
    else if (2 * c_half2 < bishop_threshold) { 
        sum = __hfma2(w_h2[0], get_input_half2_nhwc_safe(input, n, abs_h_input,     abs_w_input,     c_half2, total_c_half2), sum);
        sum = __hfma2(w_h2[1], get_input_half2_nhwc_safe(input, n, abs_h_input,     abs_w_input + 4, c_half2, total_c_half2), sum);
        sum = __hfma2(w_h2[2], get_input_half2_nhwc_safe(input, n, abs_h_input + 1, abs_w_input + 1, c_half2, total_c_half2), sum);
        sum = __hfma2(w_h2[3], get_input_half2_nhwc_safe(input, n, abs_h_input + 1, abs_w_input + 3, c_half2, total_c_half2), sum);
        sum = __hfma2(w_h2[4], get_input_half2_nhwc_safe(input, n, abs_h_input + 2, abs_w_input + 2, c_half2, total_c_half2), sum);
        sum = __hfma2(w_h2[5], get_input_half2_nhwc_safe(input, n, abs_h_input + 3, abs_w_input + 1, c_half2, total_c_half2), sum);
        sum = __hfma2(w_h2[6], get_input_half2_nhwc_safe(input, n, abs_h_input + 3, abs_w_input + 3, c_half2, total_c_half2), sum);
        sum = __hfma2(w_h2[7], get_input_half2_nhwc_safe(input, n, abs_h_input + 4, abs_w_input,     c_half2, total_c_half2), sum);
        sum = __hfma2(w_h2[8], get_input_half2_nhwc_safe(input, n, abs_h_input + 4, abs_w_input + 4, c_half2, total_c_half2), sum);
    }
    else { 
        sum = __hfma2(w_h2[0], get_input_half2_nhwc_safe(input, n, abs_h_input,     abs_w_input + 1, c_half2, total_c_half2), sum);
        sum = __hfma2(w_h2[1], get_input_half2_nhwc_safe(input, n, abs_h_input,     abs_w_input + 3, c_half2, total_c_half2), sum);
        sum = __hfma2(w_h2[2], get_input_half2_nhwc_safe(input, n, abs_h_input + 1, abs_w_input,     c_half2, total_c_half2), sum);
        sum = __hfma2(w_h2[3], get_input_half2_nhwc_safe(input, n, abs_h_input + 1, abs_w_input + 4, c_half2, total_c_half2), sum);
        sum = __hfma2(w_h2[4], get_input_half2_nhwc_safe(input, n, abs_h_input + 2, abs_w_input + 2, c_half2, total_c_half2), sum);
        sum = __hfma2(w_h2[5], get_input_half2_nhwc_safe(input, n, abs_h_input + 3, abs_w_input,     c_half2, total_c_half2), sum);
        sum = __hfma2(w_h2[6], get_input_half2_nhwc_safe(input, n, abs_h_input + 3, abs_w_input + 4, c_half2, total_c_half2), sum);
        sum = __hfma2(w_h2[7], get_input_half2_nhwc_safe(input, n, abs_h_input + 4, abs_w_input + 1, c_half2, total_c_half2), sum);
        sum = __hfma2(w_h2[8], get_input_half2_nhwc_safe(input, n, abs_h_input + 4, abs_w_input + 3, c_half2, total_c_half2), sum);
    }

    sum = __hadd2(sum, b_h2);
    
    // Conditional Activation routing
    if (activation_mode == 1) {
        float2 sum_f32 = __half22float2(sum);
        sum_f32.x = mishActivate(sum_f32.x);
        sum_f32.y = mishActivate(sum_f32.y);
        sum = __float22half2_rn(sum_f32);
    }

    int out_index = (n * 64 * total_c_half2) + (h * 8 * total_c_half2) + (w * total_c_half2) + c_half2;
    output[out_index] = sum;
#endif
}

// ============================================================================
// BACKWARD PASS GPU KERNEL (PARAMETERIZED GRADIENT)
// ============================================================================
__global__ void DepthwiseBackwardKernelNHWC_fp16(int total_c_half2, half2* d_input, half2* d_weights, half2* d_biases,
                                                 const half2* d_out, const half2* input, const half2* weights, const half2* biases,
                                                 int rook_threshold, int bishop_threshold, int knight_threshold,
                                                 int activation_mode) {
#if __CUDA_ARCH__ >= 700
    int c_half2 = blockIdx.y * blockDim.x + threadIdx.x;
    if (c_half2 >= total_c_half2) return;

    int w = threadIdx.y;
    int h = blockIdx.z;
    int n = blockIdx.x;

    half2 w_h2[9];
    #pragma unroll
    for(int i = 0; i < 9; ++i) {
        w_h2[i] = weights[i * total_c_half2 + c_half2];
    }
    half2 b_h2 = biases[c_half2];

    int abs_h_input = h - 2;
    int abs_w_input = w - 2;

    half2 feat[9];
    int coords_h[9];
    int coords_w[9];

    if (2 * c_half2 < rook_threshold) {
        coords_h[0] = abs_h_input;     coords_w[0] = abs_w_input + 2;
        coords_h[1] = abs_h_input + 1; coords_w[1] = abs_w_input + 2;
        coords_h[2] = abs_h_input + 2; coords_w[2] = abs_w_input;
        coords_h[3] = abs_h_input + 2; coords_w[3] = abs_w_input + 1;
        coords_h[4] = abs_h_input + 2; coords_w[4] = abs_w_input + 2;
        coords_h[5] = abs_h_input + 2; coords_w[5] = abs_w_input + 3;
        coords_h[6] = abs_h_input + 2; coords_w[6] = abs_w_input + 4;
        coords_h[7] = abs_h_input + 3; coords_w[7] = abs_w_input + 2;
        coords_h[8] = abs_h_input + 4; coords_w[8] = abs_w_input + 2;
    } else if (2 * c_half2 < bishop_threshold) {
        coords_h[0] = abs_h_input;     coords_w[0] = abs_w_input;
        coords_h[1] = abs_h_input;     coords_w[1] = abs_w_input + 4;
        coords_h[2] = abs_h_input + 1; coords_w[2] = abs_w_input + 1;
        coords_h[3] = abs_h_input + 1; coords_w[3] = abs_w_input + 3;
        coords_h[4] = abs_h_input + 2; coords_w[4] = abs_w_input + 2;
        coords_h[5] = abs_h_input + 3; coords_w[5] = abs_w_input + 1;
        coords_h[6] = abs_h_input + 3; coords_w[6] = abs_w_input + 3;
        coords_h[7] = abs_h_input + 4; coords_w[7] = abs_w_input;
        coords_h[8] = abs_h_input + 4; coords_w[8] = abs_w_input + 4;
    } else {
        coords_h[0] = abs_h_input;     coords_w[0] = abs_w_input + 1;
        coords_h[1] = abs_h_input;     coords_w[1] = abs_w_input + 3;
        coords_h[2] = abs_h_input + 1; coords_w[2] = abs_w_input;
        coords_h[3] = abs_h_input + 1; coords_w[3] = abs_w_input + 4;
        coords_h[4] = abs_h_input + 2; coords_w[4] = abs_w_input + 2;
        coords_h[5] = abs_h_input + 3; coords_w[5] = abs_w_input;
        coords_h[6] = abs_h_input + 3; coords_w[6] = abs_w_input + 4;
        coords_h[7] = abs_h_input + 4; coords_w[7] = abs_w_input + 1;
        coords_h[8] = abs_h_input + 4; coords_w[8] = abs_w_input + 3;
    }

    half2 sum = __float22half2_rn(make_float2(0.0f, 0.0f));
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        feat[i] = get_input_half2_nhwc_safe(input, n, coords_h[i], coords_w[i], c_half2, total_c_half2);
        sum = __hfma2(w_h2[i], feat[i], sum);
    }
    sum = __hadd2(sum, b_h2);

    int out_index = (n * 64 * total_c_half2) + (h * 8 * total_c_half2) + (w * total_c_half2) + c_half2;
    half2 d_out_val = d_out[out_index];
    half2 d_act;

    // Route backprop derivative based on activation selection
    if (activation_mode == 1) {
        float2 dout_f = __half22float2(d_out_val);
        float2 sum_f  = __half22float2(sum);
        dout_f.x = mishGradient(sum_f.x, dout_f.x);
        dout_f.y = mishGradient(sum_f.y, dout_f.y);
        d_act = __float22half2_rn(dout_f);
    } else {
        d_act = d_out_val; // Pass-through derivative line for ACTIVATION_NONE
    }

    atomicAdd(&(d_biases[c_half2]), d_act);
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        atomicAdd(&(d_weights[i * total_c_half2 + c_half2]), __hmul2(d_act, feat[i]));
        atomicAdd_input_safe(d_input, n, coords_h[i], coords_w[i], c_half2, total_c_half2, __hmul2(d_act, w_h2[i]));
    }
#endif
}

// ============================================================================
// FUNCTOR DISPATCH METHODS
// ============================================================================
template <typename T>
struct FusedChessDepthwiseFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int batch_size, int total_c_half2,
                  T* output, const T* input, const T* weights, const T* biases,
                  int rook_threshold, int bishop_threshold, int knight_threshold, int activation_mode) {
    dim3 grid(batch_size, (total_c_half2 + 31) / 32, 8);
    dim3 block(32, 8, 1);

    DepthwiseKernelNHWC_fp16<<<grid, block, 0, d.stream()>>>(
        total_c_half2, reinterpret_cast<half2*>(output), 
        reinterpret_cast<const half2*>(input), 
        reinterpret_cast<const half2*>(weights), 
        reinterpret_cast<const half2*>(biases),
        rook_threshold, bishop_threshold, knight_threshold, activation_mode
    );
  }
};

template <typename T>
struct FusedChessDepthwiseGradFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int batch_size, int total_c_half2,
                  T* d_input, T* d_weights, T* d_biases,
                  const T* d_out, const T* input, const T* weights, const T* biases,
                  int rook_threshold, int bishop_threshold, int knight_threshold, int activation_mode) {
    dim3 grid(batch_size, (total_c_half2 + 31) / 32, 8);
    dim3 block(32, 8, 1);

    DepthwiseBackwardKernelNHWC_fp16<<<grid, block, 0, d.stream()>>>(
        total_c_half2, reinterpret_cast<half2*>(d_input), 
        reinterpret_cast<half2*>(d_weights), 
        reinterpret_cast<half2*>(d_biases),
        reinterpret_cast<const half2*>(d_out), 
        reinterpret_cast<const half2*>(input), 
        reinterpret_cast<const half2*>(weights), 
        reinterpret_cast<const half2*>(biases),
        rook_threshold, bishop_threshold, knight_threshold, activation_mode
    );
  }
};


/*
// ============================================================================
// DIRECT 128-BIT SHAPE-ALIGNED VECTOR READ FOR TRAINING
// ============================================================================
__device__ __forceinline__ uint4 get_input_uint4_nhwc_safe(const uint4* input, int n, int h, int w, int c_uint4, int total_c_uint4) {
    if (h < 0 || h >= 8 || w < 0 || w >= 8) {
        uint4 zero; zero.x = 0; zero.y = 0; zero.z = 0; zero.w = 0;
        return zero;
    }
    return input[(n * 64 * total_c_uint4) + (h * 8 * total_c_uint4) + (w * total_c_uint4) + c_uint4];
}

// ============================================================================
// STREAMLINED 128-BIT VECTORIZED DEPTHWISE CONVOLUTION FORWARD KERNEL
// ============================================================================
__global__ void DepthwiseKernelNHWC_uint4(int total_c_half2, half2* output, const half2* input, 
                                          const half2* weights, const half2* biases,
                                          int rook_threshold, int bishop_threshold, int knight_threshold,
                                          int activation_mode) {
#if __CUDA_ARCH__ >= 700 
    // Since C is a multiple of 8, total_c_half2 is perfectly divisible by 4.
    int total_c_uint4 = total_c_half2 / 4;
    int c_uint4 = blockIdx.y * blockDim.x + threadIdx.x;
    
    // Warp-masking boundary check for unaligned thread tail blocks (e.g., 576 channels)
    if (c_uint4 >= total_c_uint4) return;

    int w = threadIdx.y;
    int h = blockIdx.z;
    int n = blockIdx.x;

    int base_c_half2 = c_uint4 * 4;

    // 1. Direct Coalesced Weight & Separated Bias Loads into Registers
    half2 w_h2[9][4];
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            w_h2[i][b] = weights[i * total_c_half2 + base_c_half2 + b];
        }
    }
    
    half2 b_h2[4];
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        b_h2[b] = biases[base_c_half2 + b];
    }

    int abs_h_input = h - 2;
    int abs_w_input = w - 2;

    // Absolute spatial stencil offsets matching your core network layout
    const int rook_dh[9] = {0, 1, 2, 2, 2, 2, 2, 3, 4};
    const int rook_dw[9] = {2, 2, 0, 1, 2, 3, 4, 2, 2};

    const int bishop_dh[9] = {0, 0, 1, 1, 2, 3, 3, 4, 4};
    const int bishop_dw[9] = {0, 4, 1, 3, 2, 1, 3, 0, 4};

    const int knight_dh[9] = {0, 0, 1, 1, 2, 3, 3, 4, 4};
    const int knight_dw[9] = {1, 3, 0, 4, 2, 0, 4, 1, 3};

    // 2. Initialize the 4 half2 accumulation structures (8 channels parallel)
    half2 sum0 = make_half2(0.0f, 0.0f);
    half2 sum1 = make_half2(0.0f, 0.0f);
    half2 sum2 = make_half2(0.0f, 0.0f);
    half2 sum3 = make_half2(0.0f, 0.0f);

    // 3. Select stencil paths. Guaranteed 100% homogeneous due to multiple-of-8 widths.
    int dh[9], dw[9];
    if (2 * base_c_half2 < rook_threshold) { 
        #pragma unroll
        for(int i = 0; i < 9; ++i) { dh[i] = rook_dh[i]; dw[i] = rook_dw[i]; }
    } else if (2 * base_c_half2 < bishop_threshold) { 
        #pragma unroll
        for(int i = 0; i < 9; ++i) { dh[i] = bishop_dh[i]; dw[i] = bishop_dw[i]; }
    } else { 
        #pragma unroll
        for(int i = 0; i < 9; ++i) { dh[i] = knight_dh[i]; dw[i] = knight_dw[i]; }
    }

    // 4. Core 128-Bit Streamed Spatial Math Loop
    const uint4* input_uint4 = reinterpret_cast<const uint4*>(input);

    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        uint4 raw_in = get_input_uint4_nhwc_safe(input_uint4, n, abs_h_input + dh[i], abs_w_input + dw[i], c_uint4, total_c_uint4);
        half2* in_h2 = reinterpret_cast<half2*>(&raw_in);

        sum0 = __hfma2(w_h2[i][0], in_h2[0], sum0);
        sum1 = __hfma2(w_h2[i][1], in_h2[1], sum1);
        sum2 = __hfma2(w_h2[i][2], in_h2[2], sum2);
        sum3 = __hfma2(w_h2[i][3], in_h2[3], sum3);
    }

    // 5. Apply Biases and Training Core Activation Function Routing
    sum0 = __hadd2(sum0, b_h2[0]); sum1 = __hadd2(sum1, b_h2[1]);
    sum2 = __hadd2(sum2, b_h2[2]); sum3 = __hadd2(sum3, b_h2[3]);

    if (activation_mode == 1) {
        half2* sums[4] = {&sum0, &sum1, &sum2, &sum3};
        #pragma unroll
        for(int s = 0; s < 4; ++s) {
            float2 sum_f32 = __half22float2(*(sums[s]));
            sum_f32.x = mishActivate(sum_f32.x);
            sum_f32.y = mishActivate(sum_f32.y);
            *(sums[s]) = __float22half2_rn(sum_f32);
        }
    }

    // 6. Pack and Commit 128 bits back to global VRAM in a single hardware cycle
    uint4 raw_out;
    half2* out_ptr = reinterpret_cast<half2*>(&raw_out);
    out_ptr[0] = sum0; out_ptr[1] = sum1; out_ptr[2] = sum2; out_ptr[3] = sum3;

    uint4* output_uint4 = reinterpret_cast<uint4*>(output);
    int out_index = (n * 64 * total_c_uint4) + (h * 8 * total_c_uint4) + (w * total_c_uint4) + c_uint4;
    output_uint4[out_index] = raw_out; 
#endif
}

// ============================================================================
// FUNCTOR DISPATCH MODULE WITH SCALE-ALIGNED GRID REDUCTION
// ============================================================================
template <typename T>
struct FusedChessDepthwiseFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int batch_size, int total_c_half2,
                  T* output, const T* input, const T* weights, const T* biases,
                  int rook_threshold, int bishop_threshold, int knight_threshold, int activation_mode) {
    
    int total_c_uint4 = total_c_half2 / 4;
    
    // Scale down Grid Y dimension to map the 4x workload concentration per thread
    dim3 grid(batch_size, (total_c_uint4 + 31) / 32, 8);
    dim3 block(32, 8, 1);

    DepthwiseKernelNHWC_uint4<<<grid, block, 0, d.stream()>>>(
        total_c_half2, reinterpret_cast<half2*>(output), 
        reinterpret_cast<const half2*>(input), 
        reinterpret_cast<const half2*>(weights), 
        reinterpret_cast<const half2*>(biases),
        rook_threshold, bishop_threshold, knight_threshold, activation_mode
    );
  }
};
*/

template struct FusedChessDepthwiseFunctor<GPUDevice, Eigen::half>;
template struct FusedChessDepthwiseGradFunctor<GPUDevice, Eigen::half>;

}  // namespace functor
}  // namespace tensorflow
#endif  // GOOGLE_CUDA
