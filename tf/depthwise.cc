#define EIGEN_USE_GPU
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <cuda_fp16.h>

using namespace tensorflow;

// ============================================================================
// 1. CUDA DEVICE HELPERS & KERNELS
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

__global__ void DepthwiseKernelNHWC_fp16(int total_c_half2, half2* output, const half2* input, 
                                         const half2* weights, const half2* biases,
                                         int rook_channels, int bishop_channels, int knight_channels) {
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

    if (2 * c_half2 < rook_channels) { 
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
    else if (2 * c_half2 < bishop_channels) { 
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
    
    float2 sum_f32 = __half22float2(sum);
    sum_f32.x = mishActivate(sum_f32.x);
    sum_f32.y = mishActivate(sum_f32.y);
    sum = __float22half2_rn(sum_f32);

    int out_index = (n * 64 * total_c_half2) + (h * 8 * total_c_half2) + (w * total_c_half2) + c_half2;
    output[out_index] = sum;
#endif
}

__global__ void DepthwiseBackwardKernelNHWC_fp16(int total_c_half2, half2* d_input, half2* d_weights, half2* d_biases,
                                                 const half2* d_out, const half2* input, const half2* weights, const half2* biases,
                                                 int rook_channels, int bishop_channels, int knight_channels) {
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

    if (2 * c_half2 < rook_channels) {
        coords_h[0] = abs_h_input;     coords_w[0] = abs_w_input + 2;
        coords_h[1] = abs_h_input + 1; coords_w[1] = abs_w_input + 2;
        coords_h[2] = abs_h_input + 2; coords_w[2] = abs_w_input;
        coords_h[3] = abs_h_input + 2; coords_w[3] = abs_w_input + 1;
        coords_h[4] = abs_h_input + 2; coords_w[4] = abs_w_input + 2;
        coords_h[5] = abs_h_input + 2; coords_w[5] = abs_w_input + 3;
        coords_h[6] = abs_h_input + 2; coords_w[6] = abs_w_input + 4;
        coords_h[7] = abs_h_input + 3; coords_w[7] = abs_w_input + 2;
        coords_h[8] = abs_h_input + 4; coords_w[8] = abs_w_input + 2;
    } else if (2 * c_half2 < bishop_channels) {
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
    half2 d_act = d_out_val;

    float2 dout_f = __half22float2(d_out_val);
    float2 sum_f  = __half22float2(sum);
    dout_f.x = mishGradient(sum_f.x, dout_f.x);
    dout_f.y = mishGradient(sum_f.y, dout_f.y);
    d_act = __float22half2_rn(dout_f);

    atomicAdd(&(d_biases[c_half2]), d_act);
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        atomicAdd(&(d_weights[i * total_c_half2 + c_half2]), __hmul2(d_act, feat[i]));
        atomicAdd_input_safe(d_input, n, coords_h[i], coords_w[i], c_half2, total_c_half2, __hmul2(d_act, w_h2[i]));
    }
#endif
}

// ============================================================================
// 2. TENSORFLOW OP METADATA REGISTRATION
// ============================================================================

REGISTER_OP("FusedChessDepthwise")
    .Input("input: half")
    .Input("weights: half")
    .Input("biases: half")
    .Output("output: half")
    .Attr("rook_c: int")
    .Attr("bishop_c: int")
    .Attr("knight_c: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        // Native placeholder graph parsing safety constraint
        c->set_output(0, c->input(0));
        return Status::OK();
    });

REGISTER_OP("FusedChessDepthwiseGrad")
    .Input("d_out: half")
    .Input("input: half")
    .Input("weights: half")
    .Input("biases: half")
    .Output("d_input: half")
    .Output("d_weights: half")
    .Output("d_biases: half")
    .Attr("rook_c: int")
    .Attr("bishop_c: int")
    .Attr("knight_c: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1)); // d_input matches input shape [N, 64, C]
        c->set_output(1, c->input(2)); // d_weights matches weights shape [9, C]
        c->set_output(2, c->input(3)); // d_biases matches biases shape [C]
        return Status::OK();
    });

// ============================================================================
// 3. TENSORFLOW OP KERNEL LAUNCHERS
// ============================================================================

class FusedChessDepthwiseOp : public OpKernel {
public:
    explicit FusedChessDepthwiseOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("rook_c", &rook_c_));
        OP_REQUIRES_OK(context, context->GetAttr("bishop_c", &bishop_c_));
        OP_REQUIRES_OK(context, context->GetAttr("knight_c", &knight_c_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& input_tensor = context->input(0);
        const Tensor& weights_tensor = context->input(1);
        const Tensor& biases_tensor = context->input(2);

        OP_REQUIRES(context, input_tensor.dims() == 3, errors::InvalidArgument("Input must be 3D [N, 64, C]"));

        int batch_size = input_tensor.dim_size(0);
        int channels = input_tensor.dim_size(2);
        int total_c_half2 = channels / 2;

        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));

        // Reinterpret native Eigen::half pointers into packed hardware vector arrays
        const half2* input_ptr = reinterpret_cast<const half2*>(input_tensor.flat<Eigen::half>().data());
        const half2* weights_ptr = reinterpret_cast<const half2*>(weights_tensor.flat<Eigen::half>().data());
        const half2* biases_ptr = reinterpret_cast<const half2*>(biases_tensor.flat<Eigen::half>().data());
        half2* output_ptr = reinterpret_cast<half2*>(output_tensor->flat<Eigen::half>().data());

        // Extract the active CUDA stream bound to TensorFlow's device runner context
        cudaStream_t stream = context->eigen_gpu_device().stream();

        dim3 grid(batch_size, (total_c_half2 + 31) / 32, 8);
        dim3 block(32, 8, 1);

        DepthwiseKernelNHWC_fp16<<<grid, block, 0, stream>>>(
            total_c_half2, output_ptr, input_ptr, weights_ptr, biases_ptr,
            rook_c_, bishop_c_, knight_c_
        );
    }
private:
    int rook_c_;
    int bishop_c_;
    int knight_c_;
};

class FusedChessDepthwiseGradOp : public OpKernel {
public:
    explicit FusedChessDepthwiseGradOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("rook_c", &rook_c_));
        OP_REQUIRES_OK(context, context->GetAttr("bishop_c", &bishop_c_));
        OP_REQUIRES_OK(context, context->GetAttr("knight_c", &knight_c_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& d_out_tensor = context->input(0);
        const Tensor& input_tensor = context->input(1);
        const Tensor& weights_tensor = context->input(2);
        const Tensor& biases_tensor = context->input(3);

        Tensor* d_input_tensor = nullptr;
        Tensor* d_weights_tensor = nullptr;
        Tensor* d_biases_tensor = nullptr;

        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &d_input_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(1, weights_tensor.shape(), &d_weights_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(2, biases_tensor.shape(), &d_biases_tensor));

        cudaStream_t stream = context->eigen_gpu_device().stream();

        // Essential: Zero-initialize accumulation arrays before initiating atomic additions
        cudaMemsetAsync(d_input_tensor->flat<Eigen::half>().data(), 0, d_input_tensor->AllocatedBytes(), stream);
        cudaMemsetAsync(d_weights_tensor->flat<Eigen::half>().data(), 0, d_weights_tensor->AllocatedBytes(), stream);
        cudaMemsetAsync(d_biases_tensor->flat<Eigen::half>().data(), 0, d_biases_tensor->AllocatedBytes(), stream);

        int batch_size = input_tensor.dim_size(0);
        int channels = input_tensor.dim_size(2);
        int total_c_half2 = channels / 2;

        half2* d_input_ptr = reinterpret_cast<half2*>(d_input_tensor->flat<Eigen::half>().data());
        half2* d_weights_ptr = reinterpret_cast<half2*>(d_weights_tensor->flat<Eigen::half>().data());
        half2* d_biases_ptr = reinterpret_cast<half2*>(d_biases_tensor->flat<Eigen::half>().data());

        const half2* d_out_ptr = reinterpret_cast<const half2*>(d_out_tensor.flat<Eigen::half>().data());
        const half2* input_ptr = reinterpret_cast<const half2*>(input_tensor.flat<Eigen::half>().data());
        const half2* weights_ptr = reinterpret_cast<const half2*>(weights_tensor.flat<Eigen::half>().data());
        const half2* biases_ptr = reinterpret_cast<const half2*>(biases_tensor.flat<Eigen::half>().data());

        dim3 grid(batch_size, (total_c_half2 + 31) / 32, 8);
        dim3 block(32, 8, 1);

        DepthwiseBackwardKernelNHWC_fp16<<<grid, block, 0, stream>>>(
            total_c_half2, d_input_ptr, d_weights_ptr, d_biases_ptr,
            d_out_ptr, input_ptr, weights_ptr, biases_ptr,
            rook_c_, bishop_c_, knight_c_
        );
    }
private:
    int rook_c_;
    int bishop_c_;
    int knight_c_;
};

REGISTER_KERNEL_BUILDER(Name("FusedChessDepthwise").Device(DEVICE_GPU), FusedChessDepthwiseOp);
REGISTER_KERNEL_BUILDER(Name("FusedChessDepthwiseGrad").Device(DEVICE_GPU), FusedChessDepthwiseGradOp);