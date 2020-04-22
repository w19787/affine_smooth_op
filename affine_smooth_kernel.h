#ifndef _AFFINE_SMOOTH_KERNEL_H_
#define _AFFINE_SMOOTH_KERNEL_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
// #include "tensorflow/core/util/cuda_kernel_helper.h"

void AffineSmoothKernalLauncher(const float* output, const float* input, float epsilon, int patch, 
	int h, int w, int f_r, int f_e, float* output_affine, int block_count, int threads_per_block, cudaStream_t stream);

#endif