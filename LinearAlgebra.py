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
