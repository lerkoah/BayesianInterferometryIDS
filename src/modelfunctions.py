import numpy as np

PI = np.pi


# Basis function in pixel domain
def psi(x, y, l, Cx, Cy):
    return (1.0 / (np.sqrt(2.0 * PI) * l)) * np.exp(-(1.0 / 2) * ((x - Cx) * (x - Cx) + (y - Cy) * (y - Cy)) / (l * l))


# Real part of basis function
def phiRe(u, v, l, Cx, Cy):
    return np.exp(-2.0 * PI * (u * u + v * v) * l * l) * np.cos(2.0 * PI * (Cx * u + Cy * v))


# Imaginary part of basis function
def phiIm(u, v, l, Cx, Cy):
    return np.exp(-2.0 * PI * (u * u + v * v) * l * l) * np.sin(2.0 * PI * (Cx * u + Cy * v))


def sintheticImage(I, Cx, Cy, l=10, IMAGE_SIZE_X=256, IMAGE_SIZE_Y=256):
    n_Cx = Cx.shape[0]
    n_Cy = Cy.shape[0]
    n_I = I.shape[0]
    assert n_Cx == n_Cy and n_Cx == n_I and n_Cy == n_I, 'Error dimension'
    Ns = n_I
    Im_out = np.zeros((IMAGE_SIZE_X, IMAGE_SIZE_Y))

    for i in range(Ns):
        for x in range(IMAGE_SIZE_X):
            for y in range(IMAGE_SIZE_Y):
                # print 'I(%i,%i) = %s' %(Cx[i],Cy[i],I[i]*psi(x,y,l,Cx[i],Cy[i]))
                Im_out[x][y] = Im_out[x][y] + I[i] * psi(x, y, l, Cx[i], Cy[i])
                # print Im_out[x][y]
    return Im_out


def spectrumMeasurement(V, u, v, IMAGE_SIZE_X=256, IMAGE_SIZE_Y=256):
    n_u = u.shape[0]
    n_v = v.shape[0]
    n_V = V.shape[1]
    assert n_u == n_v and n_u == n_V and n_v == n_V, 'Error dimension'
    Ns = n_u
    Vout = np.zeros((IMAGE_SIZE_X, IMAGE_SIZE_Y))
    for i in range(Ns):
        # print 'I(%i,%i) = %s' %(u[i],v[i],np.sqrt(V[0,i]*V[0,i]+V[1,i]*V[1,i]))
        Vout[u[i]][v[i]] = np.sqrt(V[0, i] * V[0, i] + V[1, i] * V[1, i])

    return Vout

def spectrum(I):
    I_fft = np.fft.fft2(I)
    I_fftAbs = np.abs(I_fft)

    return I_fft, I_fftAbs

