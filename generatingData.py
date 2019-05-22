# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from statsmodels.tsa.vector_ar.var_model import VARProcess
import matplotlib.pyplot as plt

p = 6
k = 3

timesteps = 20

coefs = np.zeros((p, k, k))
coefs[0] = 0.8 * np.eye(k)
coefs[1,0,1] = 0.1
coefs[2,0,1] = 0.2
coefs[3,0,1] = 0.4
coefs[4,0,1] = 0.2
coefs[5,0,1] = 0.1

coefs_exog = np.array([1])
sigma_u = np.eye(k)

VAR = VARProcess(coefs=coefs, coefs_exog=coefs_exog, sigma_u=sigma_u)

coefMatrices = VAR.ma_rep(timesteps)

#VAR.plotsim(100)

arr = [[] for x in range(k*k)]

for matrix in coefMatrices:
    for i in range(k):
        for j in range(k):
            arr[3*i+j].append(matrix[i,j])
        
fig = plt.figure()

for i in range(k*k):
    ax = fig.add_subplot(k,k,i+1)
    ax.plot([x for x in range(timesteps+1)], arr[i])

plt.tight_layout()
plt.show

VAR.plotsim(100)
simulation = VAR.simulate_var(400)

a = np.asarray(simulation)
np.savetxt("foo.csv", a, delimiter=",")