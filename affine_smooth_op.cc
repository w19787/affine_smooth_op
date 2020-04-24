#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "affine_smooth_op.h"

using namespace tensorflow;

REGISTER_OP("AffineSmooth")
  .Input("input1: T")
  .Input("input2: T")
  .Input("epsilon: float32")
  .Input("patch: int32")
  .Input("h: int32")
  .Input("w: int32")
  .Input("f_r: float32")
  .Input("f_e: float32")
  .Output("smooth_output: T")
  .Attr("T: numbertype")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  c->set_output(0, c->input(0));
  return Status::OK();
});

using GPUDevice = Eigen::GpuDevice;
// using CPUDevice = Eigen::ThreadPoolDevice;

template <typename Device, typename T>
class AffineSmoothOp : public OpKernel {
  public:
    explicit AffineSmoothOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
      // Grab the input tensor

      const Tensor& output_tensor = context->input(0);
      const Tensor& input_tensor = context->input(1);
      const Tensor& epsilon_tensor = context->input(2);
      const Tensor& patch_tensor = context->input(3);
      const Tensor& height_tensor = context->input(4);
      const Tensor& width_tensor = context->input(5);
      const Tensor& fr_tensor = context->input(6);
      const Tensor& fe_tensor = context->input(7);

 
      auto output = output_tensor.flat<T>().data();
      auto input = input_tensor.flat<T>().data();
      auto epsilon = epsilon_tensor.flat<float>().data();
      auto patch = patch_tensor.flat<int>().data();
      auto h = height_tensor.flat<int>().data();
      auto w = width_tensor.flat<int>().data();
      auto f_r = fr_tensor.flat<float>().data();
      auto f_e = fe_tensor.flat<float>().data();


      const int64 number_of_elements = input_tensor.NumElements();

      // Create an output tensor
      Tensor* output_affine_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                       &output_affine_tensor));
      auto output_affine_flat = output_affine_tensor->flat<T>().data();

      functor_(context->eigen_device<Device>(), number_of_elements,
        output, input, epsilon, patch, h, w, f_r, f_e, output_affine_flat);
    }

  private:
    functor::AffineSmoothOpFunctor<T> functor_;
};


#define REGISTER_GPU(T) \
  REGISTER_KERNEL_BUILDER( \
      Name("AffineSmooth")      \
      .Device(DEVICE_GPU)   \
      .TypeConstraint<T>("T"),  \
    AffineSmoothOp<GPUDevice, T>);
REGISTER_GPU(float);

// Register the GPU kernels.
// #ifdef GOOGLE_CUDA
// #define REGISTER_GPU(T) \
//   REGISTER_KERNEL_BUILDER( \
//       Name("AffineSmooth")      \
//       .Device(DEVICE_GPU)   \
//       .TypeConstraint<T>("T"),  \
//     AffineSmoothOp<GPUDevice, T>);
// REGISTER_GPU(float);

// #endif // GOOGLE_CUDA