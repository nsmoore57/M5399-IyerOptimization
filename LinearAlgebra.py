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

    Returns a vector containing the beta values corresponding to each householder vector
    """
    m,n = A.shape
    beta = np.empty(n)
    for j in range(n):
        [v, beta[j]] = _house(A[j:,j])
        A[j:,j:] = np.matmul(np.eye(m-j) - beta[j]*np.matmul(v,v.transpose()),A[j:,j:])
        if j < m-1:
            A[j+1:,j] = v[1:m-j]
        # print(A)
    return beta

def LSQR(A,b):
    m,n = A.shape
    beta = householderQR(A)

    for j in range(n):
         v = np.hstack(([1.0],A[j+1:,j])).transpose()
         b[j:] = np.matmul(np.eye(m-j) - beta[j]*np.matmul(v,v.transpose()),b[j:])

    return backSubstitution_UpperTri(A[1:n,1:n], b[1:n])

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
    A = np.array([[3, 10, 2, 3],[0, 0, 5, 7], [1, 4, 4, 7], [9, 4, 7, 1], [7, 8, 7, 1], [3, 8, 8, 5]], dtype="float")
    b = np.array([8, 2, 5, 7, 9, 10]).transpose()

