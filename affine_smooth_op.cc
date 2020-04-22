#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "affine_smooth_op.h"

using namespace tensorflow;

REGISTER_OP("AffineSmooth")
  .Attr("T: type")
  .Input("output: T")
  .Input("input: T")
  .Input("epsilon: float32")
  .Input("patch: int32")
  .Input("h: int32")
  .Input("w: int32")
  .Input("f_r: int32")
  .Input("f_e: int32")
  .Output("smooth_output: T");

using GPUDevice = Eigen::GpuDevice;

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

      const auto data_ptr = [](const Tensor& tensor) -> const T* {
        return reinterpret_cast<const T*>(tensor.tensor_data().data());
      };

      auto output = output_tensor.flat<T>().data();
      auto input = input_tensor.flat<T>().data();
      auto epsilon = *data_ptr(epsilon_tensor);
      auto patch = *data_ptr(patch_tensor);
      auto h = *data_ptr(height_tensor);
      auto w = *data_ptr(width_tensor);
      auto f_r = *data_ptr(fr_tensor);
      auto f_e = *data_ptr(fe_tensor);

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

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T) \
  REGISTER_KERNEL_BUILDER( \
      Name("AffineSmooth")      \
      .Device(DEVICE_GPU)   \
      .TypeConstraint<T>("T"),  \
    AffineSmoothOp<GPUDevice, T>);
REGISTER_GPU(float);

#endif // GOOGLE_CUDA