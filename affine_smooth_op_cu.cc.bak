#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/util/gpu_launch_config.h"
#include "tensorflow/core/framework/op.h"
#include "affine_smooth_kernel.h"
#include "affine_smooth_op.h"

namespace tensorflow{

namespace functor{
using GPUDevice = Eigen::GpuDevice;

template<typename T>
void AffineSmoothOpFunctor<T>::operator()(const GPUDevice& d, const int64 number_of_elements, 
	const T* output, const T* input, const float* epsilon, 
	const int* patch, const int* h, const int* w, const float* f_r, const float* f_e, T* output_affine)
{
	// const GpuLaunchConfig config =
	//   GetGpuLaunchConfig(number_of_elements, d);
	// const int threads_per_block = config.thread_per_block;
	const int threads_per_block = 256;
	const int block_count =
	  (number_of_elements + threads_per_block - 1) / threads_per_block;

	AffineSmoothKernalLauncher(output, input, epsilon, patch, h, w, f_r, f_e, 
		output_affine, block_count, threads_per_block, d.stream());

}

template struct AffineSmoothOpFunctor<float>;

} // name space functor

} //namespace tensorflow

#endif //GOOGLE_CUDA