#define EIGEN_USE_GPU
#include <cuda_runtime.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "gated_depthwise.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

// ============================================================================
// OP METADATA STRUCTURE REGISTRATION
// ============================================================================
REGISTER_OP("GatedDepthwise")
    .Input("input: half")
    .Input("weights: half")
    .Input("gating: half")
    .Input("biases: half")
    .Output("output: half")
    .Attr("groups: int")
    .Attr("activation_mode: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0)); 
        return tensorflow::OkStatus();
    });

REGISTER_OP("GatedDepthwiseGrad")
    .Input("d_out: half")
    .Input("input: half")
    .Input("weights: half")
    .Input("gating: half")
    .Input("biases: half")
    .Output("d_input: half")
    .Output("d_weights: half")
    .Output("d_gating: half")
    .Output("d_biases: half")
    .Attr("groups: int")
    .Attr("activation_mode: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1)); 
        c->set_output(1, c->input(2)); 
        c->set_output(2, c->input(3)); 
        c->set_output(3, c->input(4)); 
        return tensorflow::OkStatus();
    });

// ============================================================================
// OP KERNEL RUNTIMES
// ============================================================================
class GatedDepthwiseOp : public OpKernel {
public:
    explicit GatedDepthwiseOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("groups", &groups_));
        OP_REQUIRES_OK(context, context->GetAttr("activation_mode", &activation_mode_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& input_tensor = context->input(0);
        const Tensor& weights_tensor = context->input(1);
        const Tensor& gating_tensor = context->input(2);
        const Tensor& biases_tensor = context->input(3);

        OP_REQUIRES(context, input_tensor.dims() == 3, errors::InvalidArgument("Input must be 3D [N, 64, C]"));

        int batch_size = input_tensor.dim_size(0);
        int channels = input_tensor.dim_size(2);
        int total_c_half2 = channels / 2;

        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));

        functor::GatedDepthwiseFunctor<GPUDevice, Eigen::half>()(
            context->eigen_gpu_device(), batch_size, total_c_half2, groups_,
            output_tensor->flat<Eigen::half>().data(),
            input_tensor.flat<Eigen::half>().data(),
            weights_tensor.flat<Eigen::half>().data(),
            gating_tensor.flat<Eigen::half>().data(),
            biases_tensor.flat<Eigen::half>().data(),
            activation_mode_
        );
    }
private:
    int groups_;
    int activation_mode_;
};

class GatedDepthwiseGradOp : public OpKernel {
public:
    explicit GatedDepthwiseGradOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("groups", &groups_));
        OP_REQUIRES_OK(context, context->GetAttr("activation_mode", &activation_mode_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& d_out_tensor = context->input(0);
        const Tensor& input_tensor = context->input(1);
        const Tensor& weights_tensor = context->input(2);
        const Tensor& gating_tensor = context->input(3);
        const Tensor& biases_tensor = context->input(4);

        Tensor* d_input_tensor = nullptr;
        Tensor* d_weights_tensor = nullptr;
        Tensor* d_gating_tensor = nullptr;
        Tensor* d_biases_tensor = nullptr;

        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &d_input_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(1, weights_tensor.shape(), &d_weights_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(2, gating_tensor.shape(), &d_gating_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(3, biases_tensor.shape(), &d_biases_tensor));

        auto device = context->eigen_gpu_device();
        auto stream = device.stream();

        cudaMemsetAsync(d_input_tensor->flat<Eigen::half>().data(), 0, d_input_tensor->AllocatedBytes(), stream);
        cudaMemsetAsync(d_weights_tensor->flat<Eigen::half>().data(), 0, d_weights_tensor->AllocatedBytes(), stream);
        cudaMemsetAsync(d_gating_tensor->flat<Eigen::half>().data(), 0, d_gating_tensor->AllocatedBytes(), stream);
        cudaMemsetAsync(d_biases_tensor->flat<Eigen::half>().data(), 0, d_biases_tensor->AllocatedBytes(), stream);

        int batch_size = input_tensor.dim_size(0);
        int channels = input_tensor.dim_size(2);
        int total_c_half2 = channels / 2;

        functor::GatedDepthwiseGradFunctor<GPUDevice, Eigen::half>()(
            context, device, batch_size, total_c_half2, groups_,
            d_input_tensor->flat<Eigen::half>().data(),
            d_weights_tensor->flat<Eigen::half>().data(),
            d_gating_tensor->flat<Eigen::half>().data(),
            d_biases_tensor->flat<Eigen::half>().data(),
            d_out_tensor.flat<Eigen::half>().data(),
            input_tensor.flat<Eigen::half>().data(),
            weights_tensor.flat<Eigen::half>().data(),
            gating_tensor.flat<Eigen::half>().data(),
            biases_tensor.flat<Eigen::half>().data(),
            activation_mode_
        );
    }
private:
    int groups_;
    int activation_mode_;
};

REGISTER_KERNEL_BUILDER(Name("GatedDepthwise").Device(DEVICE_GPU), GatedDepthwiseOp);
REGISTER_KERNEL_BUILDER(Name("GatedDepthwiseGrad").Device(DEVICE_GPU), GatedDepthwiseGradOp);

} // namespace tensorflow