#!/usr/bin/env python
""" This module is a library of functions for completing linear algebra and related routines """

import numpy as np
import numpy.linalg as LA

class LinAlgError(Exception):
    """Base class for other exceptions"""
    pass

class DimensionMismatchError(LinAlgError):
    """For errors involving incorrect dimensions"""
    pass

def forwardSubstitution_LowerTri(L, b):
    """
    Completes forward Substitution to solve the linear system
         Lz = b
    where L is lower triangular and b is n x 1 np.array
    """
    # Dimension Check
    if L.shape[0] != L.shape[1]:
        raise DimensionMismatchError("L must be square")
    if L.shape[0] != b.shape[0]:
        raise DimensionMismatchError("Rows L ({L.shape[0]} != Rows b ({b.shape[0]}))")

    n = L.shape[0]
    z = np.zeros((n, 1))
    z[0, 0] = b[0, 0]/L[0, 0]
    for i in range(1, z.shape[0]):
        s = 0
        for j in range(i):
            s += L[i, j]*z[j, 0]
        z[i, 0] = (b[i, 0] - s)/L[i, i]

    return z

def backSubstitution_UpperTri(U, b):
    """
    Completes backward Substitution to solve the linear system
         Ux = b
    where U is upper triangular and b is a n x 1 np.array
    """
    # Dimension Check
    if U.shape[0] != U.shape[1]:
        raise DimensionMismatchError("Q must be square")
    if U.shape[0] != b.shape[0]:
        raise DimensionMismatchError("Rows Q ({Q.shape[0]} != Rows b ({b.shape[0]}))")

    n = U.shape[0]
    x = np.zeros((n, 1))
    x[n-1, 0] = b[n-1, 0]/U[n-1, n-1]

    for i in range(n-2, -1, -1):
        s = 0
        for j in range(i+1, n):
            s += U[i, j]*x[j, 0]
        x[i, 0] = (b[i, 0] - s)/U[i, i]
    return x

def householderQR(A):
    """
    Completes a QR factorization using householder matrices
    Modifies A in place to store the R matrix in the upper triangular portion
    and the householder vectors in the lower triangular portion.

    Returns a vector containing the beta values corresponding to each householder vector
    """
    m, n = A.shape
    beta = np.empty(n)
    for j in range(n):
        [v, beta[j]] = _house(A[j:, j])
        A[j:, j:] = np.matmul(np.eye(m-j) - beta[j]*np.matmul(v, v.T), A[j:, j:])
        if j < m-1:
            A[j+1:, j] = v[1:m-j]
        # print(A)
    return beta

def LSQR(A, b):
    """Completes a least squares solve using the householderQR decomp"""
    m, n = A.shape
    A_copy = A.copy()
    beta = householderQR(A_copy)
    b_copy = b.copy()

    for j in range(n):
        v = np.hstack(([1.0], A_copy[j+1:, j])).T
        b_copy[j:] = np.matmul(np.eye(m-j) - beta[j]*np.matmul(v, v.T), b_copy[j:])

    return backSubstitution_UpperTri(A_copy[:n, :n], b_copy[:n].reshape((-1, 1)))

def _house(x):
    sigma = np.matmul(x[1:].T, x[1:])
    v = x.copy()
    if sigma == 0:
        beta = 0
    else:
        mu = np.sqrt(x[0]*x[0] + sigma)
        if x[0] <= 0:
            v[0] = x[0] - mu
        else:
            v[0] = -sigma/(x[0] + mu)
        beta = 2*v[0]*v[0]/(sigma + v[0]*v[0])
        v = v/v[0]
    return v, beta

def lowRank_MinNormLS(A, b):
    """Find the minimum norm least squares solution to a low rank matrix system."""
    # TODO: Write own qr code
    # Compute the QR factorization
    Q, R = LA.qr(A, mode="reduced")

    # TODO: Reduce the dimension of Q to rank p when diagonal elements get too small

    # Solve RR^T z = Q^T b with a Cholesky Decomposition
    L = LA.cholesky(np.matmul(R, R.T))
    lamb = forwardSubstitution_LowerTri(L, np.matmul(Q.T, b))
    z = backSubstitution_UpperTri(L.T.conj(), lamb)

    return np.matmul(R.T, z)

if __name__ == "__main__":
    print("Nothing here")
