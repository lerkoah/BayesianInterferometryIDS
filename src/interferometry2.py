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

def myImageSave(strI, I):
	I = (I-I.min())/(I.max()-I.min())
	I = I*255

	im = Image.fromarray(I.astype(np.uint8)).convert('RGB')
	im.save(strI)

def phi(u,v,Cx,Cy,l):
	# buf = "(u,v) = (%d,%d) l = %d, exponent = %d+j%d" % (u,v,l,exponent.real,exponent.imag)
	# print buf
	
	phiOut = cmath.exp(-(1.0)*( 2*math.pi*math.pi*(u*u+v*v)*l*l - 2j*math.pi*(Cx*u+Cy*v) ))
	# buf = "  - phi(u,v) = %d+j%d" % (phiOut.real,phiOut.imag)
	# print buf

	return phiOut

def Vhat(u,v,alpha,Cu,Cv,l):

	nBases = alpha.shape[0]

	Vout = 0
	for i in range(nBases):
		Vout += alpha[i]*phi(u,v,Cu[i],Cv[i],l)
	
	return Vout

def Ihat(u,v,alpha,Cu,Cv,l):

	nBases = alpha.shape[0]

	Iout = 0
	for i in range(nBases):
		uu = (u-Cu[i])
		vv = (v-Cv[i])
		g = (1.0/(math.sqrt(2*math.pi)*l))*math.exp(-(0.5)*( (uu*uu+vv*vv)/(l*l) ))
		# print uu,',',vv,',',g,',',l
		Iout += alpha[i]*g
	
	return Iout


def NLL(x,U,V,nRow,nCol):
	
	uCenter = nCol/2
	vCenter = nRow/2

	nVar = x.shape[0]
	nBases = (nVar-5)/3

	# buff = " - Number of Bases: %d" % nBases
	# print buff

	p = x[0:4]
	alpha = x[4:nBases+4]
	Cu = x[nBases+4:2*nBases+4]
	Cv = x[2*nBases+4:3*nBases+4]
	l = x[3*nBases+4]

	# U = m[0]
	# v = m[1]

	nSamples = U.shape[0]

	# Creating Sigma Matrix
	Krr = p[0]*p[0]+p[1]*p[1]
	Kii = p[2]*p[2]+p[3]*p[3]
	Kir = p[1]*p[2]
	
	C = Krr+Kii
	P = Krr-Kii+2j*Kir

	L = np.array( [ [C,P] , [np.conjugate(P), np.conjugate(C)] ])
	Sigma = np.dot(L,L.T.conj())
	# Sigma = L
	# print Sigma
	SigmaDet = np.real(np.linalg.det(Sigma))

	# buff = " - Sigma Determinant = %f" % SigmaDet
	# print buff
	SigmaInv = np.linalg.inv(Sigma)

	NLLout = math.log(math.pi*SigmaDet)
	for k in range(nSamples):

		u = U[k,0]
		v = U[k,1]

		V_k    = V[k]
		Vhat_k = Vhat(u-uCenter,v-vCenter,alpha,Cu,Cv,l)

		x = np.array([[V_k-Vhat_k],[np.conjugate(V_k-Vhat_k)]])

		r = np.real(( 1/(2*nSamples)*np.dot( x.T,np.dot(SigmaInv,x) ) )/nSamples)
		# buff = "   - Element k = %f" % r
		# print r

		NLLout += r

	return NLLout

def sitentic(nRow, nCol, alpha, Cu, Cv, l):

	uCenter = nCol/2
	vCenter = nRow/2

	sinteticImage = np.zeros((nRow,nCol), dtype=np.complex)
	for u in range(nRow):
		for v in range(nCol):
			sinteticImage[u,v] = Vhat(u -uCenter,v - vCenter,alpha,Cu,Cv,l)
			# print sinteticImage[u,v]

	sinteticImageLog = np.log(np.abs(sinteticImage)+1e-6)

	return sinteticImage
	# plt.imshow(sinteticImageLog)
	# plt.show()
	# buf = "sintetic%d_%d_%d_%d.jpg" %(alpha[0],uCenter[0],vCenter[0],l)
	# myImageSave(buf,sinteticImageLog)
def sinteticPixImage(nRow,nCol,alpha,Cu,Cv,l):
	uCenter = nCol/2
	vCenter = nRow/2

	sinteticImage = np.zeros((nRow,nCol))
	for u in range(nRow):
		for v in range(nCol):
			sinteticImage[u,v] = Ihat(u -uCenter,v - vCenter,alpha,Cu,Cv,l)
			# print sinteticImage[u,v]

	return sinteticImage


def main():

	#Image Size
	nRow = 101;
	nCol = 101;

	#Image Centers
	uCenter = np.array([nRow/2])
	vCenter = np.array([nCol/2])
	Cu = np.array([0,10]) #2 Gaussian
	Cv = np.array([0,10]) #2 Gaussian
	# Cu = np.array([10])
	# Cv = np.array([10])

	#Parameters
	alpha = np.array([255, 250]) #2 Gaussian
	# alpha = np.array([255]);
	l = 0.1

	sinteticImage = sitentic(nRow, nCol,alpha,Cu,Cv,l)
	logSinteticImage = 20*np.log(np.abs(sinteticImage) + 1e-6)
	img = sinteticImage

	sinteticImagePix = sinteticPixImage(nRow, nCol,alpha/l,Cu,Cv,l)

	## Position of antennas
	B_max = 1
	antennas = 30
	typeArray = 'NRA'
	sigma = B_max/6.0

	lambda_phy=3*10**(-6); #(km)
	H0=10; #(deg)
	delta0=-30; #(deg)
	Lat=34.05;  #(deg) Lat VLA


	SpectrumMask = createSpectrumMasking(img, B_max,antennas,typeArray,sigma, lambda_phy, H0, delta0, Lat)
	SpectrumMask = np.fft.fftshift(SpectrumMask)
	(measures,U,V) = UVCreator(img,SpectrumMask)
	logMeasures = 20*np.log(np.abs(measures) + 1e-6)


	## Optimization Initialization
	# p = np.random.rand(4)
	# alpha = np.random.rand()
	# uCenter = 100*np.random.rand()
	# vCenter = 100*np.random.rand()
	# l = np.random.rand()

	nBases = 2
	# nBases = 1


	x0 = 0.1*np.random.rand(3*nBases+5)
	x0[4:nBases+4] = alpha # alpha initialization
	x0[nBases+4:2*nBases+4]   = 50*x0[nBases+4:2*nBases+4] #Cu Intialization
	x0[2*nBases+4:3*nBases+4] = 50*x0[2*nBases+4:3*nBases+4] #Cv Initialization


	result =  minimize(NLL, x0, args=(U,V,nRow,nCol,), method='SLSQP')
	# result = minimize(NLL, x0, args=(U,V,), method='nelder-mead')
	print result.x

	alpha2 = result.x[4:nBases+4]
	Cu2 = result.x[nBases+4:2*nBases+4]
	Cv2 = result.x[2*nBases+4:3*nBases+4]
	l2 = result.x[3*nBases+4]

	recontructImage = sitentic(nRow, nCol,alpha2,Cu2, Cv2,l2)
	logReconstructImage = 20*np.log(np.abs(recontructImage) + 1e-6)
	recontructImagePix = sinteticPixImage(nRow, nCol,alpha2/l2,Cu2, Cv2,l2)

	color = 'viridis'

	plt.subplot(5,1,1)
	fig = plt.imshow(sinteticImagePix)
	fig.set_cmap(color)
	plt.title("Synthetic image")
	plt.xlim([35,65])
	plt.ylim([35,65])
	plt.axis('off')

	plt.subplot(5,1,2)
	fig = plt.imshow(logSinteticImage)
	fig.set_cmap(color)
	plt.title("Synthetic image spectrum")
	plt.xlim([40,60])
	plt.ylim([40,60])
	plt.axis('off')

	plt.subplot(5,1,3)
	fig = plt.imshow(logMeasures)
	fig.set_cmap(color)
	plt.title("Measurement")
	plt.xlim([40,60])
	plt.ylim([40,60])
	plt.axis('off')


	plt.subplot(5,1,4)
	fig = plt.imshow(logReconstructImage)
	fig.set_cmap(color)
	plt.title("Reconstructed image spectrum")
	plt.xlim([40,60])
	plt.ylim([40,60])
	plt.axis('off')


	plt.subplot(5,1,5)
	fig = plt.imshow(recontructImagePix)
	fig.set_cmap(color)
	plt.title("Reconstructed image")
	plt.xlim([35,65])
	plt.ylim([35,65])
	plt.axis('off')

	# plot_image = np.concatenate((logSinteticImage, logReconstructImage), axis=1)
	# plt.imshow(plot_image)
	plt.show()

	diff = np.abs(logSinteticImage-logReconstructImage)>1e-6
	e = 1.0*np.sum(diff.astype(int))/(nRow*nCol)

	buff = 'Error rate: %f' % e
	print buff



	# print minimize(NLL, x0, args=(U,V,), method='nelder-mead')



if __name__ == "__main__":
	main()