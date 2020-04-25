#Get location of Tensorflow headers and library files
TF_INC=$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_LFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CC        = gcc -O2 -pthread
CXX       = g++
GPUCC     = nvcc
CFLAGS    = -std=c++11 -I$(TF_INC) -D_GLIBCXX_USE_CXX11_ABI=0
GPUCFLAGS = -c
LFLAGS    =  -shared -fPIC -I$(TF_INC) -I$(TF_INC)/external/nsync/public -L$(TF_LIB) -L/usr/local/cuda/lib64 -I/usr/local/cuda/include
GPULFLAGS = -x cu -shared -Xcompiler -fPIC -I$(TF_INC) -I$(TF_INC)/external/nsync/public -L$(TF_LIB)
DEBUG = -g -G
GPUDEF    = -D GOOGLE_CUDA=1
CGPUFLAGS = -lcudart -ltensorflow_framework


SRC       = affine_smooth_op.cc
GPUSRC    = affine_smooth_op_cu.cc
PROD      = affine_smooth.so
GPUPROD = affine_smooth_cu.so

default: gpu

# cpu:
# 	$(CXX) $(CFLAGS) $(SRC) $(LFLAGS) -o $(PROD)

gpu:
	$(GPUCC) $(CFLAGS) $(GPUCFLAGS) $(GPUSRC) $(GPULFLAGS) -o $(GPUPROD) ${TF_LFLAGS[@]} $(GPUDEF) -I/usr/local/ --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED
	$(CXX) $(CFLAGS)  $(SRC) $(GPUPROD) $(LFLAGS) $(CGPUFLAGS) -o $(PROD) $(GPUDEF)

clean:
	rm -f $(PROD) $(GPUPROD)