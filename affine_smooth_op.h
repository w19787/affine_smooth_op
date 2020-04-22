#ifndef _AFFINE_SMOOTH_OP_H__
#define _AFFINE_SMOOTH_OP_H__

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T>
struct AffineSmoothOpFunctor {
  void operator()(const GPUDevice& d, const int64 number_of_elements,
	    const T* output, const T* input, float epsilon, int patch, int h, int w, int f_r, int f_e, T* output_affine);
};


}  // namespace functor
}  // namespace tensorflow

#endif
#endif