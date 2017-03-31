from scipy.optimize import minimize
import numpy as np
def f_to_min (s, h):
	x = s[0:2]
	p = s[2:]
	return (p[0]*x[0]*x[0]+p[1]*x[1]*x[1]+p[2] + h)

f_to_min([1,2,1,1,1],1) # test function to minimize

p=[] # define additional args to be passed to objective function
f_to_min_cons=({'type': 'ineq', 'fun': lambda x,p: x[0]+p[0], 'args': (p,)}) # define constraint

p0=np.array([1,1,1])
h0 = 1
print minimize(f_to_min, [1,2,1,1,1], args=(h0,), method='SLSQP')