import numpy as np
# import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import math


def antennas_position2(antennas, B_max,typeArray,sigma):
	R = B_max/2.0
	BMax = B_max

	if (typeArray=='VLA') & (antennas%3==0):
		antennas_per_arm=antennas/3
		arm1_y=np.linspace(0.05,R,antennas_per_arm)
		arm1_x=np.zeros(antennas_per_arm)
		arm1  = np.array( [arm1_x.transpose(),arm1_y.transpose()] ).transpose()

		theta1 = 120 #in degree
		theta2 = 240 #in degre
		Rot_matrix1 = np.array( [[math.cos(theta1*math.pi/180), -math.sin(theta1*math.pi/180)],[math.sin(theta1*math.pi/180), math.cos(theta1*math.pi/180)]] )
		Rot_matrix2 = np.array( [[math.cos(theta2*math.pi/180), -math.sin(theta2*math.pi/180)],[math.sin(theta2*math.pi/180), math.cos(theta2*math.pi/180)]] )

		# print Rot_matrix1
		# print Rot_matrix2
		# print arm1


		arm2 = np.dot(arm1,Rot_matrix1)
		# print arm2
		arm3 = np.dot(arm1,Rot_matrix2)
		# print arm3

		VLA = np.concatenate((arm1,arm2,arm3))

		z = 2.124*np.ones(antennas)

		# print z
		# print VLA
		# print np.shape(VLA)
		# print np.shape(z)
		# positions = 0

		positions = np.column_stack((VLA,z)).transpose() #column_stack for concatenate multidimensional and unidimensional array

	elif (typeArray=='NRA'):
		x = np.zeros(antennas)
		y = np.zeros(antennas)

		R = B_max/2.0

		counter = 0
		while counter<antennas-1: #-1 because the index start in 0...

			vector_i = sigma*np.random.randn(2)
			if (math.fabs(vector_i[0]) < R) & (math.fabs(vector_i[1] )< R): #square
			#if (np.linalg.norm(vector_i)<R):								 #circunference
				x[counter+1] = vector_i[0]
				y[counter+1] = vector_i[1]
				counter += 1

		z = 5.058*np.ones(antennas)
		positions = np.column_stack((x,y,z)).transpose()

	elif typeArray=='URA':
		x = np.zeros(antennas)
		y = np.zeros(antennas)

		counter = 0
		while counter < antennas-1:
			vector_i = -R+2*R*np.random.randn(2)
			if (math.fabs(vector_i[0]) < R) & (math.fabs(vector_i[1] )< R): #square
			#if (np.linalg.norm(vector_i)<R):								 #circunference
				x[counter+1] = vector_i[0]
				y[counter+1] = vector_i[1]
				counter += 1

		z = 5.058*np.ones(antennas)
		positions = np.column_stack((x,y,z)).transpose()


	else:
		positions = np.zeros(2)

	return positions, BMax
def baselines(positions,lambdaPhy,Lat,H0,delta0):
	#Calculate coordinates u,v,w of baselines given positions
	#of antennas.

	## Change coordinate system : (N,E,U) to (X,Y,Z).
	# Z perpendicular to North pole, Y parallel to Equator.
	antennas = positions.shape[1]
	Rot_matrix = np.array( [[0,-math.sin((math.pi/180)*Lat),math.cos((math.pi/180)*Lat)],[1,0,0],[0,math.cos((math.pi/180)*Lat),math.sin((math.pi/180)*Lat)]] )

	XYZ_coordinates = np.zeros((3,antennas))

	for i in range(antennas):
		XYZ_coordinates[:,i] = np.dot(Rot_matrix,positions[:,i])

	#print XYZ_coordinates

	#Calculating Baselines (Bx,By,Bz)
	numberOfBaselines = antennas*(antennas-1)/2
	Baselines = np.zeros((3,numberOfBaselines))

	counter = 0
	for i in range(antennas-1):
		for j in range(i+1,antennas):
			Baselines[:,counter] = XYZ_coordinates[:,i]-XYZ_coordinates[:,j]
			counter += 1
	# print Baselines
	#Calculate Baselines projected (in units of wavelength) b/lambda (u,v,w)


	uvw_matrix=np.array( [[math.sin((math.pi/180)*H0),math.cos((math.pi/180)*H0),0],[-math.sin((math.pi/180)*delta0)*math.cos((math.pi/180)*H0),math.sin((math.pi/180)*delta0)*math.sin((math.pi/180)*H0),math.cos((math.pi/180)*delta0)],[math.cos((math.pi/180)*delta0)*math.cos((math.pi/180)*H0),-math.cos((math.pi/180)*delta0)*math.sin((math.pi/180)*H0),math.sin((math.pi/180)*delta0)]] )
	# print uvw_matrix
	uvw_coordinates=np.zeros((3,numberOfBaselines))

	for i in range(numberOfBaselines):
		uvw_coordinates[:,i]=(1/lambdaPhy)*np.dot(uvw_matrix,Baselines[:,i])

	return uvw_coordinates
	# print uvw_coordinates.shape
def discrete_freq(M,N,U_freq,V_freq):

	u_max = max(abs(U_freq))
	v_max = max(abs(V_freq))
	# print u_max," , ", v_max

	L = len(U_freq)
	# print L

	uv_matrix_notFlipped = np.zeros((M,N))
	row_indices = np.zeros(L,dtype=np.int)
	column_indices = np.zeros(L, dtype=np.int)

	for i in range(L):

		row_indices[i] = int( math.floor((M/2.0)*(V_freq[i]/v_max)) )
		if V_freq[i] == v_max:
			row_indices[i] = -1


		column_indices[i] = int( math.floor((N/2.0)*(U_freq[i]/u_max)) )
		if U_freq[i] == u_max:
			column_indices[i] = -1

		row_indices[i] = row_indices[i]+1
		column_indices[i] = column_indices[i]+1

		uv_matrix_notFlipped[row_indices[i],column_indices[i]] = uv_matrix_notFlipped[row_indices[i],column_indices[i]] +1

	return uv_matrix_notFlipped,row_indices,column_indices
def Gaussian2D(mu_i,mu_j,sigma,xx,yy):
	x = (xx-mu_i)*(xx-mu_i)
	y = (yy-mu_i)*(yy-mu_i)
	return math.exp(-(x*x+y*y)/(2*sigma))

def completeSpectrumBaseFunction(baseFunction,SpectrumMask,SpectrumMatrix):

	(nRows,nCols) = SpectrumMatrix.shape
	print '('+str(nRows)+' , '+str(nCols)+')'
	newSpectrum = SpectrumMatrix
	newSpectrum.real = np.zeros((nRows,nCols))
	newSpectrum.imag = np.zeros((nRows,nCols))

	sigma = 1;
	for mu_i in range(nRows):
		for mu_j in range(nCols):

			if (SpectrumMask[mu_i,mu_j] == 1):
				for x in range(nRows):
					for y in range(nCols):
						newSpectrum.real[x,y] += SpectrumMatrix.real[x,y]*baseFunction(mu_i,mu_j,sigma,x,y)
						newSpectrum.imag[x,y] += SpectrumMatrix.imag[x,y]*baseFunction(mu_i,mu_j,sigma,x,y)
				print 'Spectrum (',mu_i,',',mu_j,') added.'

	return newSpectrum

def UVCreator(imgFFT,uv_matrix_bin):
	logImgFFT = 20*np.log(np.abs(imgFFT) +1e-6)

	(N,M) = imgFFT.shape
	measures = imgFFT

	numberOfMeasures = np.sum(np.sum(uv_matrix_bin))

	U = np.zeros((numberOfMeasures, 2))
	V = np.zeros((numberOfMeasures, 2))

	k = 0
	for u in range(N):
		for v in range(M):
			if (uv_matrix_bin[u,v] == 1):
				logMeasures = logImgFFT[u,v]

				U[k,0] = u
				U[k,1] = v
				V[k,0]= np.real(measures[u,v])
				V[k,1] = np.imag(measures[u,v])
				k+=1
	M = uv_matrix_bin*imgFFT;
	return M,U,V

def createSpectrumMasking(B_max,antennas,typeArray,sigma, lambda_phy, H0, delta0, Lat, N1 = 256, N2 = 256):
	(positions,BMax) = antennas_position2(antennas,B_max,typeArray,sigma)
	# print "positions = \n"+ str(positions)
	# print "BMax = "+str(BMax)

	assert (positions != 0).any(), "Wrong number of antennas"
	#print positions[0,:]
	#Plotting
	# fig = plt.figure(1)
	# h1 = plt.subplot(121)
	# plt.plot(positions[0,:],positions[1,:],'.')
	# plt.grid()
	# plt.title("Array of antennas")
	# plt.xlabel("Esat [km]")
	# plt.ylabel("North [km]")
	# R = B_max/2
	# ArrayR = np.array( [-R, -R, 2*R, 2*R] )
	# h1.add_patch(
    # patches.Rectangle(
	#         (0,0),   # (x,y)
	#         R,          # width
	#         R,          # height
	#     )
	# )

	# baselines

	lambda_phy=3*10**(-6); #(km)
	H0=10; #(deg)
	delta0=-30; #(deg)
	Lat=34.05;  #(deg) Lat VLA
	Baselines_uvw_oneHalf=baselines(positions,lambda_phy,Lat,H0,delta0)
	Baselines_uvw=np.column_stack( (Baselines_uvw_oneHalf,-1*Baselines_uvw_oneHalf) )

	# Display Baselines
	# h2 = plt.subplot(122)
	# plt.plot(Baselines_uvw[0,:],Baselines_uvw[1,:],'r.')
	# plt.grid()
	# plt.title('Projected Baselines (Snapshot)')
	# plt.xlabel('u [rad^{-1}]')
	# plt.ylabel('v [rad^{-1}]')
	# plt.show(1)

	# Display one Half Baselines
	# plt.figure(2)
	# plt.plot(Baselines_uvw_oneHalf[0,:],Baselines_uvw_oneHalf[1,:],'r.')
	# plt.grid()
	# plt.title('Projected one Half Baselines (Snapshot)')
	# plt.xlabel('u [rad^{-1}]')
	# plt.ylabel('v [rad^{-1}]')
	# plt.show(2)

	## FFT2 indices (discrete uv plane; considering 'shift' o swapping)
	U_freq = Baselines_uvw[0,:];
	V_freq = Baselines_uvw[1,:];

	(uv_matrix,row_indices,column_indices) = discrete_freq( N1,N2,U_freq,V_freq )
	uv_mask = uv_matrix>0
	uv_matrix_bin = uv_mask.astype(int)
	return uv_matrix_bin

	# measures = measuresMatrix(imgFFT,uv_matrix_bin)

	# reconstructedSpectrum = completeSpectrumBaseFunction(Gaussian2D,uv_matrix_bin,measures)



	# plt.figure(3)
	# plt.imshow(uv_matrix)
	# plt.show(3)

	# plt.figure(4)
	# plt.imshow(measures)
	# plt.show(4)

	# cv2.destroyAllWindows()

def main():
	ImageName = 'example.jpg'
	img = cv2.imread(ImageName,0)

	B_max = 1
	antennas = 60
	typeArray = 'VLA'
	sigma = B_max/6

	lambda_phy=3*10**(-6); #(km)
	H0=10; #(deg)
	delta0=-30; #(deg)
	Lat=34.05;  #(deg) Lat VLA

	createSpectrumMasking(img, B_max,antennas,typeArray,sigma, lambda_phy, H0, delta0, Lat)



if __name__ == "__main__":
	main()
