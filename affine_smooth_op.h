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
	    const T* output, const T* input, const float* epsilon, const int* patch, 
	    const int* h, const int* w, const float* f_r, const float* f_e, T* output_affine);
};


}  // namespace functor
}  // namespace tensorflow

#endif
#endif