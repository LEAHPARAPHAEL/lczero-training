#define EIGEN_USE_GPU
#include <cuda_runtime.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "depthwise.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

// ============================================================================
// OP METADATA STRUCTURE REGISTRATION
// ============================================================================
REGISTER_OP("FusedChessDepthwise")
    .Input("input: half")
    .Input("weights: half")
    .Input("biases: half")
    .Output("output: half")
    .Attr("rook_threshold: int")
    .Attr("bishop_threshold: int")
    .Attr("knight_threshold: int")
    .Attr("activation_mode: int") 
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return tensorflow::OkStatus();
    });

REGISTER_OP("FusedChessDepthwiseGrad")
    .Input("d_out: half")
    .Input("input: half")
    .Input("weights: half")
    .Input("biases: half")
    .Output("d_input: half")
    .Output("d_weights: half")
    .Output("d_biases: half")
    .Attr("rook_threshold: int")
    .Attr("bishop_threshold: int")
    .Attr("knight_threshold: int")
    .Attr("activation_mode: int") 
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1)); 
        c->set_output(1, c->input(2)); 
        c->set_output(2, c->input(3)); 
        return tensorflow::OkStatus();
    });

// ============================================================================
// OP KERNEL RUNTIMES
// ============================================================================
class FusedChessDepthwiseOp : public OpKernel {
public:
    explicit FusedChessDepthwiseOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("rook_threshold", &rook_threshold_));
        OP_REQUIRES_OK(context, context->GetAttr("bishop_threshold", &bishop_threshold_));
        OP_REQUIRES_OK(context, context->GetAttr("knight_threshold", &knight_threshold_));
        OP_REQUIRES_OK(context, context->GetAttr("activation_mode", &activation_mode_)); 
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

        functor::FusedChessDepthwiseFunctor<GPUDevice, Eigen::half>()(
            context->eigen_gpu_device(), batch_size, total_c_half2,
            output_tensor->flat<Eigen::half>().data(),
            input_tensor.flat<Eigen::half>().data(),
            weights_tensor.flat<Eigen::half>().data(),
            biases_tensor.flat<Eigen::half>().data(),
            rook_threshold_, bishop_threshold_, knight_threshold_,
            activation_mode_ 
        );
    }
private:
    int rook_threshold_;
    int bishop_threshold_;
    int knight_threshold_;
    int activation_mode_; 
};

class FusedChessDepthwiseGradOp : public OpKernel {
public:
    explicit FusedChessDepthwiseGradOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("rook_threshold", &rook_threshold_));
        OP_REQUIRES_OK(context, context->GetAttr("bishop_threshold", &bishop_threshold_));
        OP_REQUIRES_OK(context, context->GetAttr("knight_threshold", &knight_threshold_));
        OP_REQUIRES_OK(context, context->GetAttr("activation_mode", &activation_mode_)); 
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

        auto device = context->eigen_gpu_device();
        auto stream = device.stream();

        cudaMemsetAsync(d_input_tensor->flat<Eigen::half>().data(), 0, d_input_tensor->AllocatedBytes(), stream);
        cudaMemsetAsync(d_weights_tensor->flat<Eigen::half>().data(), 0, d_weights_tensor->AllocatedBytes(), stream);
        cudaMemsetAsync(d_biases_tensor->flat<Eigen::half>().data(), 0, d_biases_tensor->AllocatedBytes(), stream);

        int batch_size = input_tensor.dim_size(0);
        int channels = input_tensor.dim_size(2); // FIX: Declared cleanly as lower-case int channels
        int total_c_half2 = channels / 2;

        functor::FusedChessDepthwiseGradFunctor<GPUDevice, Eigen::half>()(
            device, batch_size, total_c_half2,
            d_input_tensor->flat<Eigen::half>().data(),
            d_weights_tensor->flat<Eigen::half>().data(),
            d_biases_tensor->flat<Eigen::half>().data(),
            d_out_tensor.flat<Eigen::half>().data(),
            input_tensor.flat<Eigen::half>().data(),
            weights_tensor.flat<Eigen::half>().data(),
            biases_tensor.flat<Eigen::half>().data(),
            rook_threshold_, bishop_threshold_, knight_threshold_,
            activation_mode_ 
        );
    }
private:
    int rook_threshold_;
    int bishop_threshold_;
    int knight_threshold_;
    int activation_mode_; 
};

REGISTER_KERNEL_BUILDER(Name("FusedChessDepthwise").Device(DEVICE_GPU), FusedChessDepthwiseOp);
REGISTER_KERNEL_BUILDER(Name("FusedChessDepthwiseGrad").Device(DEVICE_GPU), FusedChessDepthwiseGradOp);

} // namespace tensorflow