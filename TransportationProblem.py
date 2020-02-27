import numpy as np
import InteriorPoint as IP

def TransportationProblem(supplyVec, demandVec, costMatrix):
    """
    Solves the standard Transportation problem where the supply amounts per location are given in the
    supplyVec, the amounts demanded per location are given in the demandVec, and the cost to route
    a good from supply i to demand j is in costMatrix(i,j)

    Input Arguments:
    supplyVec  -- Vector representing the amount each supply location is able to supply
    demandVec  -- Vector representing the amount each demand location demands
    costMatrix -- Matrix describing the cost of shipping a good from location i to location j

    Returns:
    If the optimal combination is found
    x        -- Coordinates of the optimal value
    k        -- The number of total iterations required
                - Includes the iterations needed to find the first feasible point

    Errors:
    Raises a DimensionMismatchError if the dimensions of the matrices are not compatible
    Raises a NonConvergenceError if the optimum cannot be found within tolerance

    Example:
    supplyVec = np.array([[150, 175, 275]]).T
    demandVec = np.array([[200, 100, 300]]).T
    costMatrix = np.array([[6, 8, 10],[7, 11, 11],[4, 5, 12]])

    # print regular style numbers with a single decimal place
    np.set_printoptions(suppress=True,precision=1)
    x, k = TransportationProblem(supplyVec, demandVec, costMatrix)
    print(x)
    """
    m = supplyVec.shape[0]
    n = demandVec.shape[0]

    A = np.zeros((m+n,m*n))
    onevec = np.ones((1,n))
    for i in range(m):
        A[i,i*m:(i+1)*m] = onevec
        A[m:,i*m:(i+1)*m] = np.eye(n)

    c = costMatrix.reshape((-1,1))
    b = np.vstack((supplyVec,demandVec))
    Q = np.zeros((m*n,m*n))
    tol = 1e-5
    kmax = 10000
    rho = .9
    mu0 = 1e4
    mumin = 1e-8

    x,k = InteriorPointBarrier(Q, c, A, b, tol, kmax, rho, mu0, mumin)
    return x.reshape(costMatrix.shape),k

if __name__ == "__main__":
    supplyVec = np.array([[150, 175, 275]]).T
    demandVec = np.array([[200, 100, 300]]).T
    costMatrix = np.array([[6, 8, 10],[7, 11, 11],[4, 5, 12]])

    # print regular style numbers with a single decimal place
    np.set_printoptions(suppress=True,precision=1)
    x, k = TransportationProblem(supplyVec, demandVec, costMatrix)
    print(x)
