from math import sin, pi
import numpy as np
from SVD import SVD



n = 12
A = 2*np.diag(np.ones(n))-np.diag(np.ones(n-1),-1)-np.diag(np.ones(n-1),1)

U, Sigma, V = SVD(A)

sing_vals = []
exact_vals = []

for i in range(n):
    sing_vals.append(Sigma[i][i])

    k = n - i
    exact_vals.append(4 * sin(k * pi / (2 * (n + 1))) ** 2)

print('\nEigenvalues for general n:')
print('lambda_k = 4*sin(k*pi/(2*(n+1)))**2, k = 1, ..., n')
print(f'\nSingular values for n = {n} (descending order):')

for i in range(n):
    print(f'{i + 1}: Computed = {sing_vals[i]}, Theoretical = {exact_vals[i]}')
