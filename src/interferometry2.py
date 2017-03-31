import numpy as np
from pymc3 import *
import matplotlib.pyplot as plt
from modelfunctions import *

def main():
    size = 100
    MAX_VALUE = 255
    MIN_VALUE = 0
    IMAGE_SIZE_X = 256
    IMAGE_SIZE_Y = 256
    SIGMA = 40

    I = np.ceil((MAX_VALUE - MIN_VALUE) * np.random.rand(size) + MIN_VALUE)
    Cx = np.ceil((IMAGE_SIZE_X)/2 + SIGMA*np.random.randn(size)).astype('int')
    Cx = np.clip(Cx,0,255)
    Cy = np.ceil((IMAGE_SIZE_Y)/2 + SIGMA*np.random.randn(size)).astype('int')
    Cy = np.clip(Cy,0,255)

    minI = np.min(I)
    argminI = np.argmin(I)
    maxI = np.max(I)
    argmaxI = np.argmax(I)

    print 'min I: %s at (%s,%s)\n' \
          'max I: %s at (%s,%s)\n' % (minI, Cx[argminI], Cy[argminI],
                                      maxI, Cx[argmaxI], Cy[argmaxI])

    Im_sinthc = sintheticImage(I, Cx, Cy)
    plt.imshow(Im_sinthc)
    Im_sinthcSprectrum, Im_sinthcSprectrumAbs = spectrum(Im_sinthc)
    plt.imshow(np.log(Im_sinthcSprectrumAbs))
    plt.show()


if __name__ == "__main__":
	main()