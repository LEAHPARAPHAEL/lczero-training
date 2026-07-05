#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "depthwise_x.h"
#include <cuda_fp16.h>

namespace tensorflow {
namespace functor {

using GPUDevice = Eigen::GpuDevice;

__device__ __forceinline__ half2 get_input_half2_nhwc_safe(const half2* input, int n, int h, int w, int c, int total_c) {
    if (h < 0 || h >= 8 || w < 0 || w >= 8) return __float2half2_rn(0.0f);
    return input[(n * 64 * total_c) + (h * 8 * total_c) + (w * total_c) + c];
}

// ============================================================================
// FORWARD PASS KERNEL (SPACE-INVARIANT 1-HEAD GLU MIXING ENGINE)
// ============================================================================
__global__ void DepthwiseXKernelNHWC(int total_c_half2, half2* output, const half2* input, 
                                     const half2* weights, const half2* recomb, const half2* biases) {
#if __CUDA_ARCH__ >= 700
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= total_c_half2) return;

    int w = threadIdx.y;
    int h = blockIdx.z;
    int n = blockIdx.x;

    // Centered relative spatial stencil offsets (-2 to +2 grid layout)
    const int dh_r[9] = {-2, -1,  0,  0, 0, 0, 0, 1, 2}; const int dw_r[9] = { 0,  0, -2, -1, 0, 1, 2, 0, 0};
    const int dh_b[9] = {-2, -2, -1, -1, 0, 1, 1, 2, 2}; const int dw_b[9] = {-2,  2, -1,  1, 0, -1, 1, -2, 2};
    const int dh_k[9] = {-2, -2, -1, -1, 0, 1, 1, 2, 2}; const int dw_k[9] = {-1,  1, -2,  2, 0, -2, 2, -1, 1};

    half2 w_rook[9], w_bish[9], w_knig[9];
    #pragma unroll
    for(int i = 0; i < 9; ++i) {
        w_rook[i] = weights[(0 + i) * total_c_half2 + c];
        w_bish[i] = weights[(9 + i) * total_c_half2 + c];
        w_knig[i] = weights[(18 + i) * total_c_half2 + c];
    }
    
    // Cache the 6 space-invariant parameters per channel dimension
    half2 param[6];
    #pragma unroll
    for(int p = 0; p < 6; ++p) {
        param[p] = recomb[p * total_c_half2 + c];
    }
    half2 bias = biases[c];

    half2 R = __float2half2_rn(0.0f);
    half2 B = __float2half2_rn(0.0f);
    half2 K = __float2half2_rn(0.0f);

    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        R = __hfma2(w_rook[i], get_input_half2_nhwc_safe(input, n, h + dh_r[i], w + dw_r[i], c, total_c_half2), R);
        B = __hfma2(w_bish[i], get_input_half2_nhwc_safe(input, n, h + dh_b[i], w + dw_b[i], c, total_c_half2), B);
        K = __hfma2(w_knig[i], get_input_half2_nhwc_safe(input, n, h + dh_k[i], w + dw_k[i], c, total_c_half2), K);
    }

    // 1-Head GLU Setup
    half2 V     = __hfma2(param[0], R, __hfma2(param[1], B, __hmul2(param[2], K)));
    half2 G_raw = __hfma2(param[3], R, __hfma2(param[4], B, __hmul2(param[5], K)));

    float2 g_f = __half22float2(G_raw);
    g_f.x = 1.0f / (1.0f + __expf(-g_f.x));
    g_f.y = 1.0f / (1.0f + __expf(-g_f.y));
    half2 G = __float22half2_rn(g_f);

    half2 blended = __hfma2(V, G, bias);

    output[(n * 64 * total_c_half2) + (h * 8 * total_c_half2) + (w * total_c_half2) + c] = blended;
#endif
}

// ============================================================================
// BACKWARD PASS KERNEL (1-HEAD SPATIAL REDUCTION CALCULUS Engine)
// ============================================================================
__global__ void DepthwiseXGradKernelNHWC(int total_c_half2, half2* d_input, half2* d_weights, half2* d_recomb, half2* d_biases,
                                         const half2* d_out, const half2* input, const half2* weights, const half2* recomb, const half2* biases) {
#if __CUDA_ARCH__ >= 700
    // Shared memory allocated for block-level reduction of parameter gradients
    __shared__ half2 s_reduce[8][32];

    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= total_c_half2) return;

    int w = threadIdx.y;
    int h = blockIdx.z;
    int n = blockIdx.x;

    const int dh_r[9] = {-2, -1,  0,  0, 0, 0, 0, 1, 2}; const int dw_r[9] = { 0,  0, -2, -1, 0, 1, 2, 0, 0};
    const int dh_b[9] = {-2, -2, -1, -1, 0, 1, 1, 2, 2}; const int dw_b[9] = {-2,  2, -1,  1, 0, -1, 1, -2, 2};
    const int dh_k[9] = {-2, -2, -1, -1, 0, 1, 1, 2, 2}; const int dw_k[9] = {-1,  1, -2,  2, 0, -2, 2, -1, 1};

    half2 w_rook[9], w_bish[9], w_knig[9];
    #pragma unroll
    for(int i = 0; i < 9; ++i) {
        w_rook[i] = weights[(0 + i) * total_c_half2 + c];
        w_bish[i] = weights[(9 + i) * total_c_half2 + c];
        w_knig[i] = weights[(18 + i) * total_c_half2 + c];
    }

    half2 param[6];
    #pragma unroll
    for(int p = 0; p < 6; ++p) {
        param[p] = recomb[p * total_c_half2 + c];
    }

    half2 x_r[9], x_b[9], x_k[9];
    half2 R = __float2half2_rn(0.0f);
    half2 B = __float2half2_rn(0.0f);
    half2 K = __float2half2_rn(0.0f);

    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        x_r[i] = get_input_half2_nhwc_safe(input, n, h + dh_r[i], w + dw_r[i], c, total_c_half2);
        x_b[i] = get_input_half2_nhwc_safe(input, n, h + dh_b[i], w + dw_b[i], c, total_c_half2);
        x_k[i] = get_input_half2_nhwc_safe(input, n, h + dh_k[i], w + dw_k[i], c, total_c_half2);

        R = __hfma2(w_rook[i], x_r[i], R);
        B = __hfma2(w_bish[i], x_b[i], B);
        K = __hfma2(w_knig[i], x_k[i], K);
    }

    int flat_idx = (n * 64 * total_c_half2) + (h * 8 * total_c_half2) + (w * total_c_half2) + c;
    half2 d_blended = d_out[flat_idx];

    atomicAdd(&(d_biases[c]), d_blended);

    // Exact Backpropagation Calculus for 1-Head GLU
    half2 V     = __hfma2(param[0], R, __hfma2(param[1], B, __hmul2(param[2], K)));
    half2 G_raw = __hfma2(param[3], R, __hfma2(param[4], B, __hmul2(param[5], K)));
    
    float2 g_f = __half22float2(G_raw);
    g_f.x = 1.0f / (1.0f + __expf(-g_f.x));
    g_f.y = 1.0f / (1.0f + __expf(-g_f.y));
    half2 G = __float22half2_rn(g_f);

    half2 d_V = __hmul2(d_blended, G);
    half2 d_G = __hmul2(d_blended, V);
    
    half2 one = __float2half2_rn(1.0f);
    half2 d_G_raw = __hmul2(d_G, __hmul2(G, __hsub2(one, G)));

    half2 d_p_local[6];
    d_p_local[0] = __hmul2(d_V, R);
    d_p_local[1] = __hmul2(d_V, B);
    d_p_local[2] = __hmul2(d_V, K);
    d_p_local[3] = __hmul2(d_G_raw, R);
    d_p_local[4] = __hmul2(d_G_raw, B);
    d_p_local[5] = __hmul2(d_G_raw, K);

    half2 d_R = __hfma2(d_V, param[0], __hmul2(d_G_raw, param[3]));
    half2 d_B = __hfma2(d_V, param[1], __hmul2(d_G_raw, param[4]));
    half2 d_K = __hfma2(d_V, param[2], __hmul2(d_G_raw, param[5]));

    // Parallel Shared Memory Reduction for the 6 Space-Invariant Parameters
    #pragma unroll
    for (int p = 0; p < 6; ++p) {
        s_reduce[threadIdx.y][threadIdx.x] = d_p_local[p];
        __syncthreads();
        if (threadIdx.y == 0) {
            half2 sum = s_reduce[0][threadIdx.x];
            #pragma unroll
            for (int step = 1; step < 8; ++step) {
                sum = __hadd2(sum, s_reduce[step][threadIdx.x]);
            }
            atomicAdd(&(d_recomb[p * total_c_half2 + c]), sum);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        atomicAdd(&(d_weights[(0 + i) * total_c_half2 + c]), __hmul2(d_R, x_r[i]));
        atomicAdd(&(d_weights[(9 + i) * total_c_half2 + c]), __hmul2(d_B, x_b[i]));
        atomicAdd(&(d_weights[(18 + i) * total_c_half2 + c]), __hmul2(d_K, x_k[i]));

        int h_r = h + dh_r[i]; int w_r = w + dw_r[i];
        if (h_r >= 0 && h_r < 8 && w_r >= 0 && w_r < 8) {
            int in_idx_r = (n * 64 * total_c_half2) + (h_r * 8 * total_c_half2) + (w_r * total_c_half2) + c;
            atomicAdd(&(d_input[in_idx_r]), __hmul2(d_R, w_rook[i]));
        }
        
        int h_b = h + dh_b[i]; int w_b = w + dw_b[i];
        if (h_b >= 0 && h_b < 8 && w_b >= 0 && w_b < 8) {
            int in_idx_b = (n * 64 * total_c_half2) + (h_b * 8 * total_c_half2) + (w_b * total_c_half2) + c;
            atomicAdd(&(d_input[in_idx_b]), __hmul2(d_B, w_bish[i]));
        }
        
        int h_k = h + dh_k[i]; int w_k = w + dw_k[i];
        if (h_k >= 0 && h_k < 8 && w_k >= 0 && w_k < 8) {
            int in_idx_k = (n * 64 * total_c_half2) + (h_k * 8 * total_c_half2) + (w_k * total_c_half2) + c;
            atomicAdd(&(d_input[in_idx_k]), __hmul2(d_K, w_knig[i]));
        }
    }
#endif
}

template <typename T>
struct DepthwiseXFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int batch_size, int total_c_half2,
                  T* output, const T* input, const T* weights, const T* recomb, const T* biases) {
    dim3 grid(batch_size, (total_c_half2 + 31) / 32, 8);
    dim3 block(32, 8, 1);

    DepthwiseXKernelNHWC<<<grid, block, 0, d.stream()>>>(
        total_c_half2, reinterpret_cast<half2*>(output), reinterpret_cast<const half2*>(input), 
        reinterpret_cast<const half2*>(weights), reinterpret_cast<const half2*>(recomb), 
        reinterpret_cast<const half2*>(biases)
    );
  }
};

template <typename T>
struct DepthwiseXGradFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int batch_size, int total_c_half2,
                  T* d_input, T* d_weights, T* d_recomb, T* d_biases,
                  const T* d_out, const T* input, const T* weights, const T* recomb, const T* biases) {
    dim3 grid(batch_size, (total_c_half2 + 31) / 32, 8);
    dim3 block(32, 8, 1);

    DepthwiseXGradKernelNHWC<<<grid, block, 0, d.stream()>>>(
        total_c_half2, reinterpret_cast<half2*>(d_input), reinterpret_cast<half2*>(d_weights), 
        reinterpret_cast<half2*>(d_recomb), reinterpret_cast<half2*>(d_biases),
        reinterpret_cast<const half2*>(d_out), reinterpret_cast<const half2*>(input), 
        reinterpret_cast<const half2*>(weights), reinterpret_cast<const half2*>(recomb), 
        reinterpret_cast<const half2*>(biases)
    );
  }
};

template struct DepthwiseXFunctor<GPUDevice, Eigen::half>;
template struct DepthwiseXGradFunctor<GPUDevice, Eigen::half>;

}  // namespace functor
}  // namespace tensorflow
#endif

/*
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "depthwise_x.h"
#include <cuda_fp16.h>

namespace tensorflow {
namespace functor {

using GPUDevice = Eigen::GpuDevice;

// ============================================================================
// CUDA GEOMETRIC MATH PRIMITIVES
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
    return d_out * (t + el * (1.0f - t * t) * sig);
}

__device__ __forceinline__ half2 get_input_half2_nhwc_safe(const half2* input, int n, int h, int w, int c, int total_c) {
    if (h < 0 || h >= 8 || w < 0 || w >= 8) return __float2half2_rn(0.0f);
    return input[(n * 64 * total_c) + (h * 8 * total_c) + (w * total_c) + c];
}

// ============================================================================
// FORWARD PASS KERNEL (GEOMETRIC CROSS-INTERACTION WITH COORDINATE MODULATION)
// ============================================================================
__global__ void DepthwiseXKernelNHWC(int total_c_half2, half2* output, const half2* input, 
                                     const half2* weights, const half2* recomb, const half2* biases,
                                     int activation_mode) {
#if __CUDA_ARCH__ >= 700
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= total_c_half2) return;

    int w = threadIdx.y;
    int h = blockIdx.z;
    int n = blockIdx.x;

    // Centered relative spatial stencil offsets (-2 to +2 grid layout)
    const int dh_r[9] = {-2, -1,  0,  0, 0, 0, 0, 1, 2}; const int dw_r[9] = { 0,  0, -2, -1, 0, 1, 2, 0, 0};
    const int dh_b[9] = {-2, -2, -1, -1, 0, 1, 1, 2, 2}; const int dw_b[9] = {-2,  2, -1,  1, 0, -1, 1, -2, 2};
    const int dh_k[9] = {-2, -2, -1, -1, 0, 1, 1, 2, 2}; const int dw_k[9] = {-1,  1, -2,  2, 0, -2, 2, -1, 1};

    half2 w_rook[9], w_bish[9], w_knig[9];
    #pragma unroll
    for(int i = 0; i < 9; ++i) {
        w_rook[i] = weights[(0 + i) * total_c_half2 + c];
        w_bish[i] = weights[(9 + i) * total_c_half2 + c];
        w_knig[i] = weights[(18 + i) * total_c_half2 + c];
    }
    
    // Space-conditional coordinate-absolute prior indexing
    int recomb_base = (h * 8 * 6 * total_c_half2) + (w * 6 * total_c_half2) + c;
    half2 param_r  = recomb[recomb_base + 0 * total_c_half2];
    half2 param_b  = recomb[recomb_base + 1 * total_c_half2];
    half2 param_k  = recomb[recomb_base + 2 * total_c_half2];
    half2 param_rb = recomb[recomb_base + 3 * total_c_half2];
    half2 param_rk = recomb[recomb_base + 4 * total_c_half2];
    half2 param_bk = recomb[recomb_base + 5 * total_c_half2];
    half2 bias     = biases[c];

    half2 r_sum = __float2half2_rn(0.0f);
    half2 b_sum = __float2half2_rn(0.0f);
    half2 k_sum = __float2half2_rn(0.0f);

    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        // FIX: Add offsets directly to absolute thread square coordinates (h, w)
        r_sum = __hfma2(w_rook[i], get_input_half2_nhwc_safe(input, n, h + dh_r[i], w + dw_r[i], c, total_c_half2), r_sum);
        b_sum = __hfma2(w_bish[i], get_input_half2_nhwc_safe(input, n, h + dh_b[i], w + dw_b[i], c, total_c_half2), b_sum);
        k_sum = __hfma2(w_knig[i], get_input_half2_nhwc_safe(input, n, h + dh_k[i], w + dw_k[i], c, total_c_half2), k_sum);
    }

    half2 R = r_sum; half2 B = b_sum; half2 K = k_sum;

    if (activation_mode == 1) {
        float2 r_f = __half22float2(r_sum); r_f.x = mishActivate(r_f.x); r_f.y = mishActivate(r_f.y); R = __float22half2_rn(r_f);
        float2 b_f = __half22float2(b_sum); b_f.x = mishActivate(b_f.x); b_f.y = mishActivate(b_f.y); B = __float22half2_rn(b_f);
        float2 k_f = __half22float2(k_sum); k_f.x = mishActivate(k_f.x); k_f.y = mishActivate(k_f.y); K = __float22half2_rn(k_f);
    }

    half2 RB = __hmul2(R, B);
    half2 RK = __hmul2(R, K);
    half2 BK = __hmul2(B, K);

    half2 blended = bias + __hmul2(param_r, R) + __hmul2(param_b, B) + __hmul2(param_k, K)
                         + __hmul2(param_rb, RB) + __hmul2(param_rk, RK) + __hmul2(param_bk, BK);

    output[(n * 64 * total_c_half2) + (h * 8 * total_c_half2) + (w * total_c_half2) + c] = blended;
#endif
}

// ============================================================================
// BACKWARD PASS KERNEL (RIGOROUS MULTI-VARIATE CHAIN RULE SOLVER)
// ============================================================================
__global__ void DepthwiseXGradKernelNHWC(int total_c_half2, half2* d_input, half2* d_weights, half2* d_recomb, half2* d_biases,
                                         const half2* d_out, const half2* input, const half2* weights, const half2* recomb, const half2* biases,
                                         int activation_mode) {
#if __CUDA_ARCH__ >= 700
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= total_c_half2) return;

    int w = threadIdx.y;
    int h = blockIdx.z;
    int n = blockIdx.x;

    const int dh_r[9] = {-2, -1,  0,  0, 0, 0, 0, 1, 2}; const int dw_r[9] = { 0,  0, -2, -1, 0, 1, 2, 0, 0};
    const int dh_b[9] = {-2, -2, -1, -1, 0, 1, 1, 2, 2}; const int dw_b[9] = {-2,  2, -1,  1, 0, -1, 1, -2, 2};
    const int dh_k[9] = {-2, -2, -1, -1, 0, 1, 1, 2, 2}; const int dw_k[9] = {-1,  1, -2,  2, 0, -2, 2, -1, 1};

    half2 w_rook[9], w_bish[9], w_knig[9];
    #pragma unroll
    for(int i = 0; i < 9; ++i) {
        w_rook[i] = weights[(0 + i) * total_c_half2 + c];
        w_bish[i] = weights[(9 + i) * total_c_half2 + c];
        w_knig[i] = weights[(18 + i) * total_c_half2 + c];
    }

    int recomb_base = (h * 8 * 6 * total_c_half2) + (w * 6 * total_c_half2) + c;
    half2 param_r  = recomb[recomb_base + 0 * total_c_half2];
    half2 param_b  = recomb[recomb_base + 1 * total_c_half2];
    half2 param_k  = recomb[recomb_base + 2 * total_c_half2];
    half2 param_rb = recomb[recomb_base + 3 * total_c_half2];
    half2 param_rk = recomb[recomb_base + 4 * total_c_half2];
    half2 param_bk = recomb[recomb_base + 5 * total_c_half2];

    half2 x_r[9], x_b[9], x_k[9];
    half2 r_sum = __float2half2_rn(0.0f);
    half2 b_sum = __float2half2_rn(0.0f);
    half2 k_sum = __float2half2_rn(0.0f);

    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        // FIX: Add offsets directly to absolute thread square coordinates (h, w)
        x_r[i] = get_input_half2_nhwc_safe(input, n, h + dh_r[i], w + dw_r[i], c, total_c_half2);
        x_b[i] = get_input_half2_nhwc_safe(input, n, h + dh_b[i], w + dw_b[i], c, total_c_half2);
        x_k[i] = get_input_half2_nhwc_safe(input, n, h + dh_k[i], w + dw_k[i], c, total_c_half2);

        r_sum = __hfma2(w_rook[i], x_r[i], r_sum);
        b_sum = __hfma2(w_bish[i], x_b[i], b_sum);
        k_sum = __hfma2(w_knig[i], x_k[i], k_sum);
    }

    half2 R = r_sum; half2 B = b_sum; half2 K = k_sum;
    if (activation_mode == 1) {
        float2 r_f = __half22float2(r_sum); r_f.x = mishActivate(r_f.x); r_f.y = mishActivate(r_f.y); R = __float22half2_rn(r_f);
        float2 b_f = __half22float2(b_sum); b_f.x = mishActivate(b_f.x); b_f.y = mishActivate(b_f.y); B = __float22half2_rn(b_f);
        float2 k_f = __half22float2(k_sum); k_f.x = mishActivate(k_f.x); k_f.y = mishActivate(k_f.y); K = __float22half2_rn(k_f);
    }

    half2 RB = __hmul2(R, B); half2 RK = __hmul2(R, K); half2 BK = __hmul2(B, K);

    int flat_idx = (n * 64 * total_c_half2) + (h * 8 * total_c_half2) + (w * total_c_half2) + c;
    half2 d_blended = d_out[flat_idx];

    // Gradients for coordinate-absolute blending priors
    atomicAdd(&(d_biases[c]), d_blended);
    atomicAdd(&(d_recomb[recomb_base + 0 * total_c_half2]), __hmul2(d_blended, R));
    atomicAdd(&(d_recomb[recomb_base + 1 * total_c_half2]), __hmul2(d_blended, B));
    atomicAdd(&(d_recomb[recomb_base + 2 * total_c_half2]), __hmul2(d_blended, K));
    atomicAdd(&(d_recomb[recomb_base + 3 * total_c_half2]), __hmul2(d_blended, RB));
    atomicAdd(&(d_recomb[recomb_base + 4 * total_c_half2]), __hmul2(d_blended, RK));
    atomicAdd(&(d_recomb[recomb_base + 5 * total_c_half2]), __hmul2(d_blended, BK));

    // FIX: Highly optimized 3-instruction factored pipeline for multivariate backprop signals
    half2 d_R = __hmul2(d_blended, __hadd2(param_r, __hfma2(param_rb, B, __hmul2(param_rk, K))));
    half2 d_B = __hmul2(d_blended, __hadd2(param_b, __hfma2(param_rb, R, __hmul2(param_bk, K))));
    half2 d_K = __hmul2(d_blended, __hadd2(param_k, __hfma2(param_rk, R, __hmul2(param_bk, B))));

    half2 d_r_sum = d_R; half2 d_b_sum = d_B; half2 d_k_sum = d_K;

    if (activation_mode == 1) {
        float2 dr_f = __half22float2(d_R); float2 r_f = __half22float2(r_sum); dr_f.x = mishGradient(r_f.x, dr_f.x); dr_f.y = mishGradient(r_f.y, dr_f.y); d_r_sum = __float22half2_rn(dr_f);
        float2 db_f = __half22float2(d_B); float2 b_f = __half22float2(b_sum); db_f.x = mishGradient(b_f.x, db_f.x); db_f.y = mishGradient(b_f.y, db_f.y); d_b_sum = __float22half2_rn(db_f);
        float2 dk_f = __half22float2(d_K); float2 k_f = __half22float2(k_sum); dk_f.x = mishGradient(k_f.x, dk_f.x); dk_f.y = mishGradient(k_f.y, dk_f.y); d_k_sum = __float22half2_rn(dk_f);
    }

    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        atomicAdd(&(d_weights[(0 + i) * total_c_half2 + c]), __hmul2(d_r_sum, x_r[i]));
        atomicAdd(&(d_weights[(9 + i) * total_c_half2 + c]), __hmul2(d_b_sum, x_b[i]));
        atomicAdd(&(d_weights[(18 + i) * total_c_half2 + c]), __hmul2(d_k_sum, x_k[i]));

        // FIX: Bounds-check and index map with identical absolute thread positions
        int h_r = h + dh_r[i]; int w_r = w + dw_r[i];
        if (h_r >= 0 && h_r < 8 && w_r >= 0 && w_r < 8) {
            int in_idx_r = (n * 64 * total_c_half2) + (h_r * 8 * total_c_half2) + (w_r * total_c_half2) + c;
            atomicAdd(&(d_input[in_idx_r]), __hmul2(d_r_sum, w_rook[i]));
        }
        
        int h_b = h + dh_b[i]; int w_b = w + dw_b[i];
        if (h_b >= 0 && h_b < 8 && w_b >= 0 && w_b < 8) {
            int in_idx_b = (n * 64 * total_c_half2) + (h_b * 8 * total_c_half2) + (w_b * total_c_half2) + c;
            atomicAdd(&(d_input[in_idx_b]), __hmul2(d_b_sum, w_bish[i]));
        }
        
        int h_k = h + dh_k[i]; int w_k = w + dw_k[i];
        if (h_k >= 0 && h_k < 8 && w_k >= 0 && w_k < 8) {
            int in_idx_k = (n * 64 * total_c_half2) + (h_k * 8 * total_c_half2) + (w_k * total_c_half2) + c;
            atomicAdd(&(d_input[in_idx_k]), __hmul2(d_k_sum, w_knig[i]));
        }
    }
#endif
}

template <typename T>
struct DepthwiseXFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int batch_size, int total_c_half2,
                  T* output, const T* input, const T* weights, const T* recomb, const T* biases,
                  int activation_mode) {
    dim3 grid(batch_size, (total_c_half2 + 31) / 32, 8);
    dim3 block(32, 8, 1);

    DepthwiseXKernelNHWC<<<grid, block, 0, d.stream()>>>(
        total_c_half2, reinterpret_cast<half2*>(output), reinterpret_cast<const half2*>(input), 
        reinterpret_cast<const half2*>(weights), reinterpret_cast<const half2*>(recomb), 
        reinterpret_cast<const half2*>(biases), activation_mode
    );
  }
};

template <typename T>
struct DepthwiseXGradFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int batch_size, int total_c_half2,
                  T* d_input, T* d_weights, T* d_recomb, T* d_biases,
                  const T* d_out, const T* input, const T* weights, const T* recomb, const T* biases,
                  int activation_mode) {
    dim3 grid(batch_size, (total_c_half2 + 31) / 32, 8);
    dim3 block(32, 8, 1);

    DepthwiseXGradKernelNHWC<<<grid, block, 0, d.stream()>>>(\
        total_c_half2, reinterpret_cast<half2*>(d_input), reinterpret_cast<half2*>(d_weights), 
        reinterpret_cast<half2*>(d_recomb), reinterpret_cast<half2*>(d_biases),
        reinterpret_cast<const half2*>(d_out), reinterpret_cast<const half2*>(input), 
        reinterpret_cast<const half2*>(weights), reinterpret_cast<const half2*>(recomb), 
        reinterpret_cast<const half2*>(biases), activation_mode
    );
  }
};

template struct DepthwiseXFunctor<GPUDevice, Eigen::half>;
template struct DepthwiseXGradFunctor<GPUDevice, Eigen::half>;

}  // namespace functor
}  // namespace tensorflow
#endif
*/