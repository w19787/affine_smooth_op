from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

import tensorflow as tf

affine_module = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_affine_smooth.so'))
affine_smooth_ops = affine_module.affine_smooth

#img need to be: channel last image h x w x c, and normalized (0, 1)
def affine_smooth(output_img, input_img, patch_size=3, f_r=15.0, f_e=0.01, epsilon=1e-7):
	out_img_shape = output_img.shape
	in_img_shape = input_img.shape

	assert (out_img_shape == in_img_shape), "output_img shape should be same as input img"
	assert (out_img_shape[-1] == 3), "please make sure input image is in channel last format"

	## affine smooth kernel accept channel first format
	output_img_reshape = tf.transpose(output_img, (2, 0, 1))
	input_img_reshape = tf.transpose(input_img, (2, 0, 1))

	affine_smooth_output =  affine_smooth_ops(output_img_reshape, input_img_reshape, epsilon, patch_size, 
		in_img_shape[0], in_img_shape[1], f_r, f_e)

	return tf.transpose(affine_smooth_output, (1, 2, 0))