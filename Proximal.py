#!/usr/bin/python
"""Module for Proximal Methods"""
import numpy as np
import numpy.linalg as LA

# Define Exceptions
class ProxError(Exception):
    """Base class for other exceptions"""
    pass

class NonConvergenceError(ProxError):
    """For Errors where convergence is not complete"""
    pass

class DimensionMismatchError(ProxError):
    """For errors involving incorrect dimensions"""
    pass

class UnrecognizedArgumentError(ProxError):
    """For errors involving incorrectly given function arguments"""
    pass

def Lasso(A, y, x0, lamb, tol, step_size, cost_or_pos="cost", kmax=1000):
    """
    Implement the LASSO method
    Run Lasso Method to solve:
    min (1/2)norm2(Ax-b)**2 + lamb*norm1(x)

    Input Arguments:
    A           -- Matrix of coefficients - typically data
    y           -- Data to match Ax to
    x0          -- Initial guess for weights
    lamb        -- Strength of the norm1(x) term
    tol         -- Error tolerance for stopping condition - see cost_or_pos
    step_size   -- Step size to take
                   - typically 1/(largest eigenvalue of A) if available
    cost_or_pos -- Whether the stopping condition is based on cost or position
                   - "cost" will terminate if cost(x_{k+1}) - cost(x_{k}) < tol
                   - "pos" will terminate if norm2(x_{k+1} - x_k) < tol
    kmax        -- Maximum steps allowed, used for stopping condition

    Returns:
    If the optimal is found within tolerance
    x        -- Coordinates of the optimal value
    k        -- The number of total iterations required
                - Includes the iterations needed to find the first feasible point

    Errors:
    Raises a DimensionMismatchError if the dimensions of the matrices are not compatible
    Raises a NonConvergenceError if the optimum cannot be found within tolerance
    Raises a UnrecognizedArgumentError if cost_or_pos takes a value other than "cost" or "pos"

    Example:
    m = np.random.randint(8, 15)
    n = m+1
    while n >= m:
        n = np.random.randint(5, 15)

    # Select random A, y, x0
    A = np.random.normal(size=(m, n))
    y = np.random.normal(size=(m, 1))
    x0 = np.random.normal(size=(n, 1))

    # Problem Params
    tol = 1e-8
    lamb = 0.2

    # Calculate max eigenvalue of A and use it to set the step size
    L = max(LA.eigvalsh(np.matmul(A.T, A)))
    step_size = 1.0/L

    # Run LASSO
    x_Lasso, k_Lasso = Lasso(A, y, x0, lamb, tol, step_size, cost_or_pos="cost", kmax=100000)

    TODOs:
    TODO Consider moving general proximal stuff into separate method
    """

    # Make sure the cost_or_pos is recognized
    if cost_or_pos not in ('cost', 'pos'):
        raise UnrecognizedArgumentError("cost_or_pos must either be cost or pos")
    # Check if the dimensions of A and b are compatible
    compat, error = _DimensionsCompatible_Lasso(A, y, x0)
    if not compat:
        raise DimensionMismatchError(error)

    # Put these far enough apart to make sure the tolerance is exceeded
    xnew = x0 + 2*tol
    xold = x0

    k = 1

    # Precalc A^T*A since it's needed often
    ATA = np.matmul(A.T, A)

    # If the cost function is our stopping criteria, then we need
    #   variables to track it
    if cost_or_pos == "cost":
        cost_curr = 0.5*LA.norm(np.matmul(A, xnew) - y)**2 + lamb*LA.norm(xnew, 1)
        cost_old = cost_curr

    # Used for stopping criteria
    diff = 2*tol

    while diff > tol and k < kmax:
        xold = xnew
        yk = xold - step_size*(np.matmul(ATA, xold) - np.matmul(A.T, y))
        xnew = _Prox_1Norm(yk, lamb*step_size)

        # Update diff based on preference
        if cost_or_pos == "cost":
            cost_old = cost_curr
            cost_curr = 0.5*LA.norm(np.matmul(A, xnew) - y)**2 + lamb*LA.norm(xnew, 1)
            diff = cost_old - cost_curr
        else:
            diff = LA.norm(xnew - xold)

        k += 1

    if k >= kmax:
        raise NonConvergenceError("kmax exceeded, consider raising it")

    return xnew, k

def _Prox_1Norm(v, theta):
    """Calculate the Prox operator for the 1 norm - used for Lasso"""
    ret = np.zeros(v.shape)
    for i in range(ret.shape[0]):
        if abs(v[i, 0]) >= theta:
            ret[i, 0] = v[i, 0] - theta*np.sign(v[i, 0])
    return ret

def _DimensionsCompatible_Lasso(A, y, x0):
    """Check to make sure the dimensions of a quadratic programming problem are compatible"""
    if A.shape[0] != y.shape[0]: return False, f"Rows A ({A.shape[0]}) != Rows y ({y.shape[0]})"
    if A.shape[1] != x0.shape[0]: return False, f"Cols A ({A.shape[1]}) != Rows x0 ({x0.shape[0]})"
    return True, "No error"

if __name__ == "__main__":
    # Test _Prox1Norm
    # import matplotlib
    # import matplotlib.pyplot as plt
    # s = np.linspace(-1,1,100).reshape((-1,1))
    # theta = 0.1
    # y = _Prox_1Norm_VerTwo(s, theta)

    # plt.plot(s,y)
    # plt.show()

    from Newton import GradDescent_ILS
    import time
    # Test seeds - need seeds where ILS will converge
    seeds = range(2020, 2025)

    for i in seeds:
        np.random.seed(i)

        # Select random matrix size
        m = np.random.randint(8, 15)
        n = m+1
        while n >= m:
            n = np.random.randint(5, 15)

        # Select random A, y, x0
        A = np.random.normal(size=(m, n))
        y = np.random.normal(size=(m, 1))
        x0 = np.random.normal(size=(n, 1))

        # Problem Params
        tol = 1e-8
        lamb = 0.2

        # Run LASSO and time it - including Eigvalue calculation since it could be expensive
        start = time.time()
        L = max(LA.eigvalsh(np.matmul(A.T, A)))
        step_size = 1.0/L
        x_Lasso, k_Lasso = Lasso(A, y, x0, lamb, tol, step_size, cost_or_pos="pos", kmax=100000)
        end = time.time()
        Lasso_time = end - start

        # Now we need code to check our results, we'll use GradDescent_BB
        Atilde = np.hstack((A, -A))
        z = np.vstack((np.maximum(0, x0), -np.minimum(0, x0)))
        Q = np.matmul(Atilde.T, Atilde)
        c = -np.matmul(Atilde.T, y) + lamb*np.sum(z)
        tol = 1e-4
        CD_tao = 1e-8

        # Cost function
        q = (lambda z: 0.5*np.matmul(z.T, np.matmul(Q, z)) + np.matmul(c.T, z))

        # Run ILS and time it
        start = time.time()
        x_ILS, _ = GradDescent_ILS(q, "CD", z, tol, 1000000, CD_tao=CD_tao)
        x_ILS = x_ILS[:n] - x_ILS[n:]
        end = time.time()
        ILS_time = end - start

        Lasso_cost = 0.5*LA.norm(np.matmul(A, x_Lasso) - y)**2 + lamb*LA.norm(x_Lasso, 1)
        ILS_cost = 0.5*LA.norm(np.matmul(A, x_ILS) - y)**2 + lamb*LA.norm(x_ILS, 1)

        print("============")
        print(f"Seed                 : {i}", i)
        print(f"m x n                : {m} x {n}")
        print(f"Cost Lasso           : {Lasso_cost}")
        print(f"Cost ILS             : {ILS_cost}")
        print(f"Time Lasso (sec)     : {Lasso_time}")
        print(f"Time ILS   (sec)     : {ILS_time}")
