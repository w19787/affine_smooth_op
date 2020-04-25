CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python

OP_SRCS = kernels/affine_smooth_op.cc
CUDA_SRCS = kernels/affine_smooth_op_cu.cc

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}

GPU_ONLY_TARGET_LIB = python/affine_smooth.cu.o
TARGET_LIB = python/_affine_smooth.so


all: affine_smooth_op

$(GPU_ONLY_TARGET_LIB): $(CUDA_SRCS)
	$(NVCC) -std=c++11 -c -o $@ $^ ${TF_CFLAGS} $(TF_LFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr

affine_smooth_op: $(TARGET_LIB) 
$(TARGET_LIB): $(OP_SRCS) $(GPU_ONLY_TARGET_LIB)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}  -D GOOGLE_CUDA=1  -I/usr/local/cuda/targets/x86_64-linux/include -L/usr/local/cuda-10.1/targets/x86_64-linux/lib -lcudart

clean:
	rm -f $(ZERO_OUT_TARGET_LIB) $(GPU_ONLY_TARGET_LIB) $(TARGET_LIB)