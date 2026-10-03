import numpy as np
from numpy import linalg

def mgs(A): # computes a reduced QR factorization with A in R^(mxn) where m >= n using modified Gram-Schmidt
    m = len(A)
    n = len(A[0])
    Q = np.zeros((m,n))
    R = np.zeros((n,n))
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i][j] = np.dot(Q[:, i], v)
            v -= R[i][j]*Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R

def cgs(A):
    m = len(A)
    n = len(A[0])
    Q = np.zeros((m,n))
    R = np.zeros((n,n))
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i][j] = np.dot(Q[:, i], A[:, j])
            v -= R[i][j]*Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return Q, R