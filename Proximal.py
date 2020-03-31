#!/usr/bin/python
"""Module for Proximal Methods"""
import numpy as np
import numpy.linalg as LA

# Define Exceptions
class IPError(Exception):
    """Base class for other exceptions"""
    pass

class NonConvergenceError(IPError):
    """For Errors where convergence is not complete"""
    pass

class DimensionMismatchError(IPError):
    """For errors involving incorrect dimensions"""
    pass

def Lasso(A, y, x0, tol, step_size, kmax=1000):
    """
    Implement the LASSO method
    TODO Docs
    TODO Consider moving general proximal stuff into separate method
    """

    # Put these far enough apart to make sure the tolerance is exceeded
    xnew = x0 + 2*tol
    xold = x0

    k = 1

    # Precalc A^T*A since it's needed often
    ATA = np.matmul(A.T, A)

    while LA.norm(xnew-xold) > tol and k < kmax:
        xold = xnew
        yk = xold - step_size*(np.matmul(ATA, xold) - np.matmul(A.T, y))
        xnew = _Prox_1Norm(step_size, yk)

    if k >= kmax:
        raise NonConvergenceError("kmax exceeded, consider raising it")

    return xnew, k


def _Prox_1Norm(lamb, v):
    ret = np.zeros(v.shape)
    for i in range(ret.shape[0]):
        if v[i, 0] <= -lamb:
            ret[i, 0] = v[i, 0] + lamb
        elif v[i, 0] >= lamb:
            ret[i, 0] = v[i, 0] - lamb
    return ret

if __name__ == "__main__":
    np.random.seed(2020)
    # Select random matrix size
    m = np.random.randint(8, 15)
    n = m+1
    while n >= m:
        n = np.random.randint(5, 15)

    print("m = ", m)
    print("n = ", n)

    # Select random A, y, x0
    A = np.random.normal(size=(m, n))
    xtrue = np.random.normal(size=(n, 1))
    y = np.matmul(A, xtrue)
    x0 = np.random.normal(size=(n, 1))

    print("A = ", A)
    print("y = ", y)
    print("x0 = ", x0)

    # Problem Params
    tol = 1e-8
    L = max(LA.eigvalsh(np.matmul(A.T,A)))
    step_size = 1.0/L
    print("step_size = ", step_size)

    # Run LASSO
    x, k = Lasso(A, y, x0, tol, step_size, kmax=1000)
    print("||xtrue - x|| = ", LA.norm(xtrue - x))
