# Affine Smooth GPU OPs for tensorflow 2.1+

Since pycuda seems to be not longer supported and the 10.0+ cuda version is not worked well. 

Affine smooth is a nice tool to optimize the generated image. This project is to provide your simply function to do the affine smooth. and this project should be a good reference for other engineers to create gpu ops for tensorflow kernel.

# Usage

## Use pre-build custom op
The prebuild .so file is built on tensorflow 2.1 with cuda 10.1. The example.py shows the usage. 
```
affine_output = affine_smooth(output_img, input_img, patch_size=3, f_r=15.0, f_e=0.01, epsilon=1e-7)
```
*Note: the image is required in channel last format and be normalized between 0, 1 as float32 type*

## Build your own version affine smooth op
You can build your own version:

```
make
```
*Note: your might face cuda_fp16.h not found issue. You can create a gpus directory under third_party in your virtual python environment. and make a simple link to your cuda include directory*

``` 
cd gpus
ln -sf /usr/local/cuda-10.1/targets/x86_64-linux/ cuda
```


# Affine Result
<p align="center">
    <img src='examples/gen101.png' height='140' width='210'/>
    <img src='examples/in101.png' height='140' width='210'/>
    <img src='examples/affine_smooth101.png' height='140' width='210'/>
</p>


## Acknowledgement
The affine cuda code is from https://github.com/luanfujun/deep-photo-styletransfer/blob/master/cuda_utils.cu