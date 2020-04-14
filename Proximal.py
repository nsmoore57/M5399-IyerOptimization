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

class InvalidArgumentError(ProxError):
    """For errors involving incorrectly given function arguments"""
    pass

def Lasso(A, y, x0, lamb, tol, step_size=None, cost_or_pos="cost", kmax=100000):
    """
    Implement the Proximal Method to solve the Lasso Problem:
    min (1/2)norm2(Ax-b)**2 + lamb*norm1(x)

    Input Arguments:
    A           -- Matrix of coefficients - typically data
    y           -- Data to match Ax to
    x0          -- Initial guess for weights
    lamb        -- Regularization Parameter
    tol         -- Error tolerance for stopping condition - see cost_or_pos
    step_size   -- Step size to take
                   - If None - will use 1/(largest eigenvalue of A)
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
    Raises a InvalidArgumentError if cost_or_pos takes a value other than "cost" or "pos"

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
    lamb = 0.2
    tol = 1e-8

    # Run LASSO
    x_Lasso, k_Lasso = Lasso(A, y, x0, lamb, tol)
    """

    # This is used a couple of places so calculate it once
    ATA = np.matmul(A.T, A)

    # If the step_size is not explicitly specified - we calculate it here
    if step_size is None:
        L = max(LA.eigvalsh(ATA))
        step_size = 1.0/L

    # Make sure the cost_or_pos is recognized
    if cost_or_pos not in ('cost', 'pos'):
        raise InvalidArgumentError("cost_or_pos must either be cost or pos")

    # Check if the dimensions of A and b are compatible
    compat, error = _DimensionsCompatible_Lasso(A, y, x0)
    if not compat:
        raise DimensionMismatchError(error)

    gradf = (lambda x: np.matmul(ATA, x) - np.matmul(A.T, y))
    if cost_or_pos == "cost":
        cost = (lambda x: 0.5*LA.norm(np.matmul(A, x) - y)**2 + lamb*LA.norm(x, 1))
    else:
        cost = (lambda x: x)

    return ProximalMethod(x0, gradf, _Prox_1Norm, lamb, tol, step_size, cost, kmax)

def RidgeRegression(A, y, x0, lamb, tol, step_size=None, cost_or_pos="cost", kmax=100000):
    """
    Implement the Proximal Method to solve the Ridge Regression problem:
    min (1/2)norm2(Ax-b)**2 + (lamb/2)*norm2(x)**2

    Input Arguments:
    A           -- Matrix of coefficients - typically data
    y           -- Data to match Ax to
    x0          -- Initial guess for weights
    lamb        -- Regularization Parameter
    tol         -- Error tolerance for stopping condition - see cost_or_pos
    step_size   -- Step size to take
                   - If None - will use 1/(largest eigenvalue of A)
    cost_or_pos -- Whether the stopping condition is based on cost or position
                   - "cost" will terminate if cost(x_{k+1}) - cost(x_{k}) < tol
                   - "pos" will terminate if norm2(x_{k+1} - x_k) < tol
    kmax        -- Maximum steps allowed, used for stopping condition

    Returns:
    If the optimal is found within tolerance
    x        -- Coordinates of the optimal value
    k        -- The number of total iterations required

    Errors:
    Raises a DimensionMismatchError if the dimensions of the matrices are not compatible
    Raises a NonConvergenceError if the optimum cannot be found within tolerance
    Raises a InvalidArgumentError if cost_or_pos takes a value other than "cost" or "pos"

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
    lamb = 0.2
    tol = 1e-8

    # Run Ridge Regression
    x_RR, k_RR = RidgeRegression(A, y, x0, lamb, tol)
    """

    # This is used a couple of times so calculate it once
    ATA = np.matmul(A.T, A)

    # If step_size is not explicitly given - calculate it from A
    if step_size is None:
        L = max(LA.eigvalsh(ATA))
        step_size = 1.0/L

    # Make sure the cost_or_pos is recognized
    if cost_or_pos not in ('cost', 'pos'):
        raise InvalidArgumentError("cost_or_pos must either be cost or pos")

    # Check if the dimensions of A and b are compatible
    compat, error = _DimensionsCompatible_Lasso(A, y, x0)
    if not compat:
        raise DimensionMismatchError(error)

    # Prox operator for 2 norm
    proxg = (lambda v, theta: v.copy()/(theta + 1))

    gradf = (lambda x: np.matmul(ATA, x) - np.matmul(A.T, y))
    if cost_or_pos == "cost":
        cost = (lambda x: 0.5*LA.norm(np.matmul(A, x) - y)**2 + (lamb/2.0)*LA.norm(x)**2)
    else:
        cost = (lambda x: x)

    return ProximalMethod(x0, gradf, proxg, lamb, tol, step_size, cost, kmax)

def ElasticNet(A, y, x0, lamb, alpha, tol, step_size=None, cost_or_pos="cost", kmax=100000):
    """
    Implement the Proximal Method to solve the Elastic Net Problem:
    min (1/2)norm2(Ax-b)**2 + lamb*[alpha*norm1(x) + ((1-alpha)/2)*norm2(x)**2]

    Input Arguments:
    A           -- Matrix of coefficients - typically data
    y           -- Data to match Ax to
    x0          -- Initial guess for weights
    lamb        -- Regularization Parameter
    alpha       -- Convex Combination Parameter between Lasso and Ridge Regression
                   - alpha = 1 is Lasso
                   - alpha = 0 is Ridge Regression
    tol         -- Error tolerance for stopping condition - see cost_or_pos
    step_size   -- Step size to take
                   - If None - will use 1/(largest eigenvalue of A)
    cost_or_pos -- Whether the stopping condition is based on cost or position
                   - "cost" will terminate if cost(x_{k+1}) - cost(x_{k}) < tol
                   - "pos" will terminate if norm2(x_{k+1} - x_k) < tol
    kmax        -- Maximum steps allowed, used for stopping condition

    Returns:
    If the optimal is found within tolerance
    x        -- Coordinates of the optimal value
    k        -- The number of total iterations required

    Errors:
    Raises a DimensionMismatchError if the dimensions of the matrices are not compatible
    Raises a NonConvergenceError if the optimum cannot be found within tolerance
    Raises a InvalidArgumentError if cost_or_pos takes a value other than "cost" or "pos"

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
    lamb = 0.2
    alpha = 0.5
    tol = 1e-8

    # Run Elastic Net
    x_RR, k_RR = ElasticNet(A, y, x0, lamb, alpha, tol)
    """

    # This is used a couple of times so calculate it once
    ATA = np.matmul(A.T, A)

    # If step_size is not explicitly given - calculate it from A
    if step_size is None:
        step_size = 1.0/max(LA.eigvalsh(ATA))

    # Make sure the cost_or_pos is recognized
    if cost_or_pos not in ('cost', 'pos'):
        raise InvalidArgumentError("cost_or_pos must either be cost or pos")

    # Check if the dimensions of A and b are compatible
    compat, error = _DimensionsCompatible_Lasso(A, y, x0)
    if not compat:
        raise DimensionMismatchError(error)

    # Prox is the one-norm prox with a extra weighting by alpha
    proxg = (lambda v, theta: _Prox_1Norm(v, theta*alpha))

    gradf = (lambda x: np.matmul(ATA, x) - np.matmul(A.T, y) + (1-alpha)*lamb*x)
    if cost_or_pos == "cost":
        cost = (lambda x: 0.5*LA.norm(np.matmul(A, x) - y)**2 \
                          + lamb*(alpha*LA.norm(x, 1) \
                          + ((1-alpha)/2)*LA.norm(x)**2))
    else:
        cost = (lambda x: x)

    return ProximalMethod(x0, gradf, proxg, lamb, tol, step_size, cost, kmax)

def ProximalMethod(x0, gradf, proxg, lamb, tol, step_size, cost, kmax=100000):
    """
    Implement the general Proximal method to solve
    min f(x) + lamb*g(x)

    Input Arguments:
    gradf       -- Callable which calculates the gradient of f(x)
                   - takes 1 argument - the position x
    proxg       -- Callable which calculates Prox operator for g
                   - takes 2 arguments, the position x and weight theta
    lamb        -- Regularization Parameter
    tol         -- Error tolerance for stopping condition
    step_size   -- Step size to take
                   - typically 1/(largest eigenvalue of A) if available
    cost        -- For Stopping condition
                   - Takes 1 argument - the position
                   Function returns when norm(cost_k - cost_{k+1}) < tol
    kmax        -- Maximum steps allowed, used for stopping condition

    Returns:
    If the optimal is found within tolerance
    x        -- Coordinates of the optimal value
    k        -- The number of total iterations required

    Errors:
    Raises a NonConvergenceError if the optimum cannot be found within tolerance
    Raises a InvalidArgumentError if any of the expected callable arguments are not callable
    """

    # Check if gradf is callable
    if not callable(gradf):
        raise InvalidArgumentError("gradf is not a callable")
    # Check if proxg is callable
    if not callable(proxg):
        raise InvalidArgumentError("proxg is not a callable")
    # Check if cost is callable
    if not callable(cost):
        raise InvalidArgumentError("cost is not a callable")

    # Put these far enough apart to make sure the tolerance is exceeded
    xnew = x0 + 2*tol
    xold = x0

    k = 1

    cost_curr = cost(xnew)
    cost_old = cost_curr + 2*tol

    while LA.norm(cost_old - cost_curr) > tol and k < kmax:
        xold = xnew
        yk = xold - step_size*gradf(xold)
        xnew = proxg(yk, lamb*step_size)

        cost_old = cost_curr
        cost_curr = cost(xnew)

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

def _test_Lasso(n):
    from Newton import GradDescent_BB
    import time

    for i in range(n):
        # Select random matrix size
        m = np.random.randint(8, 50)
        n = m+1
        while n >= m:
            n = np.random.randint(5, 50)

        # Select random A, y, x0
        A = np.random.normal(size=(m, n))
        y = np.random.normal(size=(m, 1))
        x0 = np.random.normal(size=(n, 1))

        # Problem Params
        tol = 1e-8
        lamb = 0.2

        # Run LASSO and time it
        start = time.time()
        x_Lasso, k_Lasso = Lasso(A, y, x0, lamb, tol, cost_or_pos="cost", kmax=100000)
        end = time.time()
        Lasso_time = end - start

        # Now we need code to check our results, we'll use GradDescent_BB
        Atilde = np.hstack((A, -A))
        z = np.vstack((np.maximum(0, x0), -np.minimum(0, x0)))
        Q = np.matmul(Atilde.T, Atilde)
        c = -np.matmul(Atilde.T, y) + lamb*np.ones(z.shape)
        tol = 1e-3
        CD_tao = 1e-4

        # Cost function
        q = (lambda z: 0.5*np.matmul(z.T, np.matmul(Q, z)) + np.matmul(c.T, z))

        # Run BB and time it
        start = time.time()
        x_BB, k_BB = GradDescent_BB(q, "CD", z, tol, 100000, CD_tao=CD_tao)
        x_BB = x_BB[:n] - x_BB[n:]
        end = time.time()
        BB_time = end - start

        Lasso_cost = 0.5*LA.norm(np.matmul(A, x_Lasso) - y)**2 + lamb*LA.norm(x_Lasso, 1)
        BB_cost = 0.5*LA.norm(np.matmul(A, x_BB) - y)**2 + lamb*LA.norm(x_BB, 1)

        print("============")
        print(f"Test                 : {i}")
        print(f"m x n                : {m} x {n}")
        print(f"Cost Lasso           : {Lasso_cost}")
        print(f"Cost BB              : {BB_cost}")
        print(f"Iter Lasso           : {k_Lasso}")
        print(f"Iter BB              : {k_BB}")
        print(f"Time Lasso (sec)     : {Lasso_time}")
        print(f"Time BB    (sec)     : {BB_time}")

def _test_RidgeRegression(n):
    from Newton import GradDescent_BB
    import time

    for i in range(n):
        # Select random matrix size
        # m = np.random.randint(6, 15)
        m = np.random.randint(6, 50)
        n = m+1
        while n >= m:
            n = np.random.randint(5, 50)

        # Select random A, y, x0
        A = np.random.normal(size=(m, n))
        y = np.random.normal(size=(m, 1))
        x0 = np.random.normal(size=(n, 1))

        # Problem Params
        tol = 1e-8
        lamb = 0.2

        # Run RidgeRegression and time it
        start = time.time()
        x_RR, k_RR = RidgeRegression(A, y, x0, lamb, tol, cost_or_pos="pos", kmax=100000)
        end = time.time()
        RR_time = end - start

        # Set up cost for BB
        Q = np.matmul(A.T, A) + lamb*np.eye(A.shape[1])
        c = -np.matmul(A.T, y)
        tol = 1e-3
        CD_tao = 1e-4
        q = (lambda x: 0.5*np.matmul(x.T, np.matmul(Q, x)) + np.matmul(c.T, x))

        # Run BB and time it
        start = time.time()
        x_BB, k_BB = GradDescent_BB(q, "CD", x0, tol, 100000, CD_tao=CD_tao)
        end = time.time()
        BB_time = end - start

        RR_cost = 0.5*LA.norm(np.matmul(A, x_RR) - y)**2 + lamb*LA.norm(x_RR, 1)
        BB_cost = 0.5*LA.norm(np.matmul(A, x_BB) - y)**2 + lamb*LA.norm(x_BB, 1)

        print("============")
        print(f"Test                 : {i}")
        print(f"m x n                : {m} x {n}")
        print(f"Cost RR              : {RR_cost}")
        print(f"Cost BB              : {BB_cost}")
        print(f"Iter RR              : {k_RR}")
        print(f"Iter BB              : {k_BB}")
        print(f"Time RR (sec)        : {RR_time}")
        print(f"Time BB (sec)        : {BB_time}")

def _test_ElasticNet(n):
    from Newton import GradDescent_BB
    import time

    for i in range(n):
        # Select random matrix size
        m = np.random.randint(8, 50)
        n = m+1
        while n >= m:
            n = np.random.randint(5, 50)

        # Select random A, y, x0
        A = np.random.normal(size=(m, n))
        y = np.random.normal(size=(m, 1))
        x0 = np.random.normal(size=(n, 1))

        # Problem Params
        lamb = 0.2
        alpha = 0.5
        tol = 1e-8

        # Run ElasticNet and time it
        start = time.time()
        x_EN, k_EN = ElasticNet(A, y, x0, lamb, alpha, tol, cost_or_pos="cost")
        end = time.time()
        EN_time = end - start

        # Now we need code to check our results, we'll use GradDescent_BB
        Atilde = np.hstack((A, -A))
        z = np.vstack((np.maximum(0, x0), -np.minimum(0, x0)))
        Q = np.matmul(Atilde.T, Atilde) + (1-alpha)*lamb*np.eye(Atilde.shape[1])
        c = -np.matmul(Atilde.T, y) + alpha*lamb*np.ones(z.shape)
        tol = 1e-3
        CD_tao = 1e-4

        # Cost function
        q = (lambda z: 0.5*np.matmul(z.T, np.matmul(Q, z)) + np.matmul(c.T, z))

        # Run BB and time it
        start = time.time()
        x_BB, k_BB = GradDescent_BB(q, "CD", z, tol, 100000, CD_tao=CD_tao)
        x_BB = x_BB[:n] - x_BB[n:]
        end = time.time()
        BB_time = end - start

        EN_cost = 0.5*LA.norm(np.matmul(A, x_EN) - y)**2 \
                + lamb*(alpha*LA.norm(x_EN, 1) \
                + ((1-alpha)/2)*LA.norm(x_EN)**2)
        BB_cost = 0.5*LA.norm(np.matmul(A, x_BB) - y)**2 \
                + lamb*(alpha*LA.norm(x_BB, 1) \
                + ((1-alpha)/2)*LA.norm(x_BB)**2)

        print("============")
        print(f"Test                 : {i}")
        print(f"m x n                : {m} x {n}")
        print(f"Cost EN              : {EN_cost}")
        print(f"Cost BB              : {BB_cost}")
        print(f"Iter EN              : {k_EN}")
        print(f"Iter BB              : {k_BB}")
        print(f"Time EN    (sec)     : {EN_time}")
        print(f"Time BB    (sec)     : {BB_time}")

if __name__ == "__main__":
    # _test_Lasso(5)
    # _test_RidgeRegression(5)
    # _test_ElasticNet(5)
