import pymc3 as pm
import numpy as np

import matplotlib.pyplot as plt
from modelfunctions import *

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from masking import *

import theano.tensor as tt

PI = np.pi
def Vobs_function(U,l,alpha,Cx,Cy, factor = None, IMAGE_SIZE_X = 20, IMAGE_SIZE_Y = 20):
    # We have implemented the model defined by:
    # V_obs = sum_{i=0}^{N_BASIS} alpha_i phi_i(u,v)
    # where phi_i(u,v) = exp(-2pi(u^2+v^2)l^2)exp(-j2pi(Cx^i u + Cv^i v))

    if factor == None:
        factor = (IMAGE_SIZE_X*IMAGE_SIZE_Y/IMAGE_SIZE_X)

    # Rescale u,v space
    u_aux = U[:,1] - IMAGE_SIZE_X/2
    u_aux = u_aux/factor
    v_aux = U[:,0] - IMAGE_SIZE_Y/2
    v_aux = v_aux/factor
#     print(u_aux)

    # Amplitud (Mantle o carrier)
    A = tt.exp(-2*np.pi**2*(u_aux*u_aux+v_aux*v_aux)*l**2)
#     A   = tt.outer(A_u,A_v)
#     print(A.eval().shape)
#     print(A.shape)

    # Imaginary Exponential (Mudolator)
#     cosPart = tt.cos(-2*np.pi*(Cx[:,np.newaxis]*u_aux + Cy[:,np.newaxis]*v_aux))
#     sinPart = tt.sin(-2*np.pi*(Cx[:,np.newaxis]*u_aux + Cy[:,np.newaxis]*v_aux))
    imagExp = tt.exp(-2*np.pi*1j*(Cx[:,np.newaxis]*u_aux + Cy[:,np.newaxis]*v_aux))
#     ## This is a outer multiplication between column's matrix
#     imagExp   = imagExp_u.dimshuffle(0, 1, 'x') * imagExp_v.dimshuffle(0, 'x', 1)
#     print(imagExp.eval().shape)

#     print((alpha*A*imagExp).eval().shape)
    ## Finally, alpha_i * phi_i
#     out_real = tt.sum(alpha*A*cosPart, axis = 0)
#     out_imag = tt.sum(alpha*A*sinPart, axis = 0)
    out_real = tt.real(tt.sum(alpha[:,np.newaxis]*A*imagExp, axis = 0))
    out_imag = tt.imag(tt.sum(alpha[:,np.newaxis]*A*imagExp, axis = 0))

#     print(imagExp.shape)

#     print(cosPart.shape, sinPart.shape)
    return out_real, out_imag

size = 3
MAX_VALUE = 255
MIN_VALUE = 0
IMAGE_SIZE_X = 50
IMAGE_SIZE_Y = 50
SIGMA = 3

#I = np.ceil((MAX_VALUE - MIN_VALUE) * np.random.rand(size) + MIN_VALUE)
#Cx = np.ceil((IMAGE_SIZE_X)/2 + SIGMA*np.random.randn(size)).astype('int')
#Cx = np.clip(Cx,0,IMAGE_SIZE_X)
#Cy = np.ceil((IMAGE_SIZE_Y)/2 +  SIGMA*np.random.randn(size)).astype('int')
#Cy = np.clip(Cy,0,IMAGE_SIZE_Y)

I = np.array([1, 1, 1])
Cx = np.array([25, 40, 10])
Cy = np.array([25, 30, 10])

minI = np.min(I)
argminI = np.argmin(I)
maxI = np.max(I)
argmaxI = np.argmax(I)

print('min I: %s at (%s,%s)\n' \
      'max I: %s at (%s,%s)\n' % (minI, Cx[argminI], Cy[argminI],
                                  maxI, Cx[argmaxI], Cy[argmaxI]))

Im_sinthc = sintheticImage(I,Cx,Cy, l=SIGMA, IMAGE_SIZE_X = IMAGE_SIZE_X, IMAGE_SIZE_Y = IMAGE_SIZE_Y)
img = Im_sinthc #rename variable

img_fft, img_fftabs = spectrum(img)

## Parameters
B_max = 1
antennas = 60
typeArray = 'ALL'
sigma = B_max/6.0

lambda_phy=3*10**(-6); #(km)
H0=10; #(deg)
delta0=-30; #(deg)
Lat=34.05;  #(deg) Lat VLA

## Masking function returns the mask
mask = createSpectrumMasking(B_max,antennas,typeArray,sigma, lambda_phy, H0, delta0, Lat, N1 = IMAGE_SIZE_X, N2 = IMAGE_SIZE_Y)

measurements,U,V = UVCreator(np.fft.fftshift(img_fft),mask)
measurements_abs = np.log(np.abs(measurements)+1e-12)

print(V[1,:])
#print(U)
print(measurements[0,1])

allIndex = np.arange(V.shape[0])
numberOfSamplings = np.ceil(V.shape[0]*0.7).astype(int)
sampledIndex = np.random.choice(allIndex, numberOfSamplings)
allV = V
allU = U

cov = np.array([[0.02, 0.0002],[0.0002, 0.02]])
m = np.zeros(2)

V = allV[sampledIndex,:] + np.random.multivariate_normal(m,cov)
U = allU[sampledIndex,:] + np.random.multivariate_normal(m,cov)

#from scipy import optimize
numberOfSamplings = U.shape[0]
numberOfBasis = 20
rbf_model = pm.Model()

l = SIGMA
init_alpha = np.random.rand(numberOfBasis)
init_Cx = 50*np.random.rand(numberOfBasis)
init_Cy = 50*np.random.rand(numberOfBasis)
init_sigma = 10*np.array([[1,1],[1,2]])
init_U = U.astype(int)

#l = np.random.rand()
#init_alpha = 2*np.ones(numberOfBasis)
#init_Cx = 2*np.ones(numberOfBasis)
#init_Cy = 2*np.ones(numberOfBasis)
#init_sigma = 2*np.ones(numberOfBasis)

beta_0 = 0.5

print('Initial Conditions:')
print('Number of Basis: %s; Number of Samplings: %s' % (numberOfBasis, numberOfSamplings))
print('Cx,Cy: %s,%s; ' % (init_Cx,init_Cy)),
print('alpha: %s; ' % (init_alpha)),
print('l: %s' % l)
with rbf_model:
    alpha_model = pm.Gamma('alpha', alpha=beta_0*init_alpha+1, beta=beta_0, shape = numberOfBasis)

#     Cx_model = pm.Gamma('Cx', alpha=beta_0*(init_Cx)+1, beta=beta_0, shape = numberOfBasis)
#     Cy_model = pm.Gamma('Cy', alpha=beta_0*(init_Cy)+1, beta=beta_0, shape = numberOfBasis)
#     Cx_model = pm.Normal('Cx', mu=init_Cx, sd=20, shape = numberOfBasis)
    Cx_model = pm.Uniform('Cx', lower=0, upper=IMAGE_SIZE_X, shape = numberOfBasis)
#     Cy_model = pm.Normal('Cy', mu=init_Cy, sd=20, shape = numberOfBasis)
    Cy_model = pm.Uniform('Cy', lower=0, upper=IMAGE_SIZE_Y, shape = numberOfBasis)

    l_model = pm.Gamma('l', alpha = beta_0*l+2, beta=beta_0, shape = 1)
#     l_model = pm.Normal('l',  mu=l, sd=200, shape = 1)

#     nu = pm.Uniform('nu', 0, 1000)
#     C_triu = pm.LKJCorr('C_triu', nu, 2, testval=init_corr)
#     C = pm.Deterministic('C', tt.fill_diagonal(C_triu[np.zeros((2, 2), dtype=np.int64)], 1.))

#     sigma_diag = pm.Deterministic('sigma_mat', tt.nlinalg.diag(sigma))
#     sigma_model = pm.Deterministic('cov', tt.nlinalg.matrix_dot(sigma_diag, C, sigma_diag))

    sd_dist = pm.Normal.dist(0,0.1)
    packed_chol = pm.LKJCholeskyCov('chol_cov', eta=2, n=2, sd_dist=sd_dist)
    chol = pm.expand_packed_triangular(2, packed_chol, lower=True)
    sigma_model = tt.dot(chol, chol.T)

#     sigma_model = pm.Lognormal('sigma', init_sigma, np.ones(2), shape=(2,2), testval=init_sigma)

    PHI_Re, PHI_Im = Vobs_function(init_U, l_model, alpha_model, Cx_model, Cy_model,
                                   IMAGE_SIZE_X=IMAGE_SIZE_X, IMAGE_SIZE_Y=IMAGE_SIZE_Y)
    V_model = tt.stack([PHI_Re, PHI_Im], axis = 1)

    V_obs = pm.MvNormal('V_obs', mu=V_model, cov=sigma_model, observed= V)

#     db = pm.backends.Text('test_log_normal')
    n_samples = 1000

#     step = pm.Metropolis()
    step = pm.Slice()
    trace = pm.sample(n_samples, step)
#     estimation = pm.find_MAP()
#     print(estimation)
