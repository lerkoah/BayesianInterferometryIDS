import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import math
import cmath
from scipy.optimize import minimize
from PIL import Image
from masking import *

def main():

	ImageName = '/home/lerkoah-lab/Dropbox/Interferometry/Imagenes_interf/vla1_256.tiff'
	img = cv2.imread(ImageName,0)
	imgFFT = np.fft.fft2(img)
	imgFFT = np.fft.fftshift(imgFFT)
	logImgFFT = 20*np.log(np.abs(imgFFT))

	color = 'viridis'

	fig = plt.imshow(logImgFFT)
	fig.set_cmap(color)
	plt.axis('off')
	plt.show()


if __name__ == "__main__":
	main()