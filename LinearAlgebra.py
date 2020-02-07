import numpy as np

def forwardSubstitution_LowerTri(L, b):
    """
    Completes forward Substitution to solve the linear system
         Lz = b
    where L is lower triangular and b is a vector
    """
    z = np.zeros(b.shape)
    z[0] = b[0]/L[1,1]
    for i in range(1,z.shape[0]):
        z[i,0] = (b[i] - np.matmul(L[i,0:i], z[0:i,0]))/L[i,i]

    return z

def backSubstitution_UpperTri(U, b):
    """
    Completes backward Substitution to solve the linear system
         Ux = b
    where U is upper triangular and b is a vector
    """
    z = np.zeros(b.shape)
    z[-1] = b[-1]/U[-1,-1]
    for i in range(z.shape[0]-1,-1,-1):
        z[i,0] = (b[i] - np.matmul(U[i,i+1:], z[i+1:,0]))/U[i,i]

    return z

def householderQR(A):
    """
    Completes a QR factorization using householder matrices
    Modifies A in place to store the R matrix in the upper triangular portion
    and the householder vectors in the lower triangular portion.
    """
    for j in range(A.shape[1]):
        [v, beta] = _house(A[j:,j])
        A[j:,j:] = np.matmul(np.eye(m-j) - beta[j]*np.matmul(v,v.transpose()),A[j:,j:])
        if j < A.shape[0]:
            A[j+1:,j] = v[2:m-j]
    return A, beta

def LSQR(A,b):
    [H, beta] = 


def _house(x):
    sigma = np.matmul(x[1:].transpose(),x[1:])
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


if __name__ == "__main__":
    A = np.random.rand()
    [v, beta] = _house(x)
    print(3*v)
