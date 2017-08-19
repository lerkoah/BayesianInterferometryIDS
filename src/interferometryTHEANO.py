import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import math
import cmath
import theano
import theano.tensor as T
from PIL import Image
from masking import *

def myImageSave(strI, I):
	I = (I-I.min())/(I.max()-I.min())
	I = I*255

	im = Image.fromarray(I.astype(np.uint8)).convert('RGB')
	im.save(strI)

def phi(u,v,Cx,Cy,l):

	exponent = -(0.5)*( (u*u+v*v) - 1j*(Cx*u+Cy*v) )*l*l
	# buf = "(u,v) = (%d,%d) l = %d, exponent = %d+j%d" % (u,v,l,exponent.real,exponent.imag)
	# print buf
	
	phiOut = cmath.exp(-(0.5)*( (u*u+v*v) - 1j*(Cx*u+Cy*v) )*l*l)
	return phiOut

def Vhat(u,v,alpha,Cu,Cv,l,nBases):

	Vout = 0
	for i in range(nBases):
		Vout += alpha[i]*phi(u,v,Cu[i],Cv[i],l)
	
	return Vout


def NLL(p,alpha,Cu,Cv,l,U,V,nRow,nCol,nBases):
	
	uCenter = nCol/2
	vCenter = nRow/2

	Krr = p[0]*p[0]+p[1]*p[1]
	Kii = p[2]*p[2]+p[3]*p[3]
	Kir = p[1]*p[2]
	
	C = Krr+Kii
	P = Krr-Kii+2j*Kir

	L = np.array( [ [C,P] , [np.conjugate(P), np.conjugate(C)] ])
	Sigma = np.dot(L,L.T.conj())
	# Sigma = L
	# print Sigma
	SigmaDet = T.nlinalg.Det(Sigma)
	SigmaInv = T.nlinalg.MatrixInverse(Sigma)

	NLLout = T.log(math.pi*SigmaDet)
	for i in range(nBases):
		u = U[k,0]
		v = U[k,1]

		V_k    = V[k]
		Vhat_k = Vhat(u-uCenter,v-vCenter,alpha,Cu,Cv,l, nBases)
		x      = np.array([[V_k-Vhat_k],[np.conjugate(V_k-Vhat_k)]])

		NLLout += np.real(( 1/(2*nSamples)*np.dot( x.T,np.dot(SigmaInv,x) ) )/nSamples)

	return NLLout

def sitentic(nRow, nCol, alpha, Cu, Cv, l):

	uCenter = nCol/2
	vCenter = nRow/2
	nBases  = alpha.shape[0]

	sinteticImage = np.zeros((nRow,nCol), dtype=np.complex)
	for u in range(nRow):
		for v in range(nCol):
			sinteticImage[u,v] = Vhat(u -uCenter,v - vCenter,alpha,Cu,Cv,l,nBases)
			# print sinteticImage[u,v]

	sinteticImageLog = np.log(np.abs(sinteticImage)+1e-6)

	return sinteticImage
	# plt.imshow(sinteticImageLog)
	# plt.show()
	# buf = "sintetic%d_%d_%d_%d.jpg" %(alpha[0],uCenter[0],vCenter[0],l)
	# myImageSave(buf,sinteticImageLog)

def main():

	#Image Size
	nRow = 101;
	nCol = 101;

	#Image Centers
	uCenter = np.array([nRow/2]);
	vCenter = np.array([nCol/2]);
	Cu = np.array([100,10])
	Cv = np.array([10,100])

	#Parameters
	alpha = np.array([255, 100]);
	l = 0.5

	sinteticImage = sitentic(nRow, nCol,alpha,Cu,Cv,l)
	logSinteticImage = 20*np.log(np.abs(sinteticImage) + 1e-6)
	img = sinteticImage

	## Position of antennas
	B_max = 1
	antennas = 90
	typeArray = 'NRA'
	sigma = B_max/6.0

	lambda_phy=3*10**(-6); #(km)
	H0=10; #(deg)
	delta0=-30; #(deg)
	Lat=34.05;  #(deg) Lat VLA


	SpectrumMask = createSpectrumMasking(img, B_max,antennas,typeArray,sigma, lambda_phy, H0, delta0, Lat)
	(measures,U,V) = UVCreator(img,SpectrumMask)




	# print U.shape

	# Optimization Initialization
	p = T.dvector('p')
	alpha = T.dvector('alpha')
	Cu = T.dvector('Cu')
	Cv = T.dvector('Cv')
	l = T.dscalar('l')

	nBases = 1

	NLLfun = NLL(p,alpha,Cu,Cv,l,U,V,nRow,nCol,nBases)

	# x0 = np.random.rand(3*nBases+5)
	# x0[4:nBases+4] = 255*x0[4:nBases+4] # alpha initialization
	# x0[nBases+4:2*nBases+4]   = 50*x0[nBases+4:2*nBases+4] #Cu Intialization
	# x0[2*nBases+4:3*nBases+4] = 50*x0[2*nBases+4:3*nBases+4] #Cv Initialization


	# result =  minimize(NLL, x0, args=(U,V,nRow,nCol,), method='SLSQP')
	# # result = minimize(NLL, x0, args=(U,V,), method='nelder-mead')
	# print result.x

	# alpha2 = result.x[4:nBases+4]
	# Cu2 = result.x[nBases+4:2*nBases+4]
	# Cv2 = result.x[2*nBases+4:3*nBases+4]
	# l2 = result.x[3*nBases+4]

	# recontructImage = sitentic(nRow, nCol,alpha2,Cu2, Cv2,l2)
	# logReconstructImage = 20*np.log(np.abs(recontructImage) + 1e-6)

	# plt.subplot(1,2,1)
	# plt.imshow(logSinteticImage)
	# plt.title("Imagen Sintetica")

	# plt.subplot(1,2,2)
	# plt.imshow(logReconstructImage)
	# plt.title("Imagen Reconstruida")
	# # plot_image = np.concatenate((logSinteticImage, logReconstructImage), axis=1)
	# # plt.imshow(plot_image)
	# plt.show()




	# print minimize(NLL, x0, args=(U,V,), method='nelder-mead')



if __name__ == "__main__":
	main()