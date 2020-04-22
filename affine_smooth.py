import tensorflow as tf

affine_module = tf.load_op_library('./affine_smooth.so')
affine_smooth = affine_module.affine_smooth