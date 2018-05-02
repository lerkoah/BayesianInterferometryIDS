import numpy as np
import scipy.io as sio

import matplotlib.pyplot as plt

import imageio
import ot

mycmap = 'seismic'

I = imageio.imread('name_HD142527.jpg')
Imod = imageio.imread('name_HD142527_reconstruction_1500basis.jpg')

#%% sinkhorn

# reg term
lambd = 1e-3
xs = np.zeros((I.shape[0]*I.shape[1],2))

i = 0
j = 0
for k in range(xs.shape[1]):
    xs[k,:] = np.array([i,j])
    j += 1
    if j%I.shape[1] == 0:
        j = 0
        i += 1
xt = xs
M = ot.dist(xs,xt)
M /= M.max()

Gs = ot.sinkhorn(I.reshape(-1)/np.sum(I), Imod.reshape(-1)/np.sum(Imod), M, lambd)

plt.imshow(Gs)
plt.show()
