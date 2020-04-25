#ifndef _AFFINE_SMOOTH_KERNEL_H_
#define _AFFINE_SMOOTH_KERNEL_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
// #include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

void AffineSmoothKernalLauncher(const float* output, const float* input, const float* epsilon, const int* patch, 
	const int* h, const int* w, const float* f_r, const float* f_e, float* output_affine, int block_count, int threads_per_block, cudaStream_t stream);

}
#endif