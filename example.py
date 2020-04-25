import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from python.affine_smooth import affine_smooth

def main():
	output_img = np.array(Image.open("examples/gen101.png").convert("RGB"), dtype=np.float32)/225.
	input_img = np.array(Image.open("examples/in101.png").convert("RGB"), dtype=np.float32)/255.

	affine_smooth_output = affine_smooth(output_img, input_img)
	result = np.uint8(np.clip(affine_smooth_output * 255., 0, 255.))
	# plt.imshow(result)
	Image.fromarray(result).save("examples/affine_smooth101.png")

if __name__ == '__main__':
  main()