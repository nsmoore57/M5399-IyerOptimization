#!/usr/bin/python
"""Module for Proximal Methods"""
import warnings
import numpy as np
import numpy.linalg as LA

# Define Exceptions
class ProxWarning(UserWarning):
    """Base class for other warnings"""
    pass

class ProxError(Exception):
    """Base class for other exceptions"""
    pass

class NonConvergenceWarning(ProxWarning):
    """For Warnings where convergence is not complete"""
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
    Raises a InvalidArgumentError if cost_or_pos takes a value other than "cost" or "pos"

    Warnings:
    Issues a NonConvergenceWarning if the optimum cannot be found within tolerance

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
        step_size = 0.1/L

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

    return ProximalMethod(x0, gradf, Prox_1Norm, lamb, tol, step_size, cost, kmax)

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
    Raises a InvalidArgumentError if cost_or_pos takes a value other than "cost" or "pos"

    Warnings:
    Issues a NonConvergenceWarning if the optimum cannot be found within tolerance

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
        step_size = 0.1/L

    # Make sure the cost_or_pos is recognized
    if cost_or_pos not in ('cost', 'pos'):
        raise InvalidArgumentError("cost_or_pos must either be cost or pos")

    # Check if the dimensions of A and b are compatible
    compat, error = _DimensionsCompatible_Lasso(A, y, x0)
    if not compat:
        raise DimensionMismatchError(error)

    gradf = (lambda x: np.matmul(ATA, x) - np.matmul(A.T, y))
    if cost_or_pos == "cost":
        cost = (lambda x: 0.5*LA.norm(np.matmul(A, x) - y)**2 + (lamb/2.0)*LA.norm(x)**2)
    else:
        cost = (lambda x: x)

    return ProximalMethod(x0, gradf, Prox_2Norm, lamb, tol, step_size, cost, kmax)

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
    Raises a InvalidArgumentError if cost_or_pos takes a value other than "cost" or "pos"

    Warnings:
    Issues a NonConvergenceWarning if the optimum cannot be found within tolerance

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
        step_size = 0.1/max(LA.eigvalsh(ATA))

    # Make sure the cost_or_pos is recognized
    if cost_or_pos not in ('cost', 'pos'):
        raise InvalidArgumentError("cost_or_pos must either be cost or pos")

    # Check if the dimensions of A and b are compatible
    compat, error = _DimensionsCompatible_Lasso(A, y, x0)
    if not compat:
        raise DimensionMismatchError(error)

    # Prox is the one-norm prox with a extra weighting by alpha
    proxg = (lambda v, theta: Prox_1Norm(v, theta*alpha))

    gradf = (lambda x: np.matmul(ATA, x) - np.matmul(A.T, y) + (1-alpha)*lamb*x)
    if cost_or_pos == "cost":
        cost = (lambda x: 0.5*LA.norm(np.matmul(A, x) - y)**2 \
                          + lamb*(alpha*LA.norm(x, 1) \
                          + ((1-alpha)/2)*LA.norm(x)**2))
    else:
        cost = (lambda x: x)

    return ProximalMethod(x0, gradf, proxg, lamb, tol, step_size, cost, kmax)

def ProximalMethod(x0, gradf, proxg, lamb, tol, step_size, cost, kmax=100000, accel=None, accel_args=None):
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
    accel       -- None, "nesterov", or "fista"
    accel_args  -- If using nesterov acceleration - accel[0] = m, and accel[1] = L
                   - ignored if accel != "nesterov"

    Returns:
    If the optimal is found within tolerance
    x        -- Coordinates of the optimal value
    k        -- The number of total iterations required

    Errors:
    Raises a InvalidArgumentError if any of the expected callable arguments are not callable

    Warnings:
    Issues a NonConvergenceWarning if the optimum cannot be found within tolerance
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

    # Check the acceleration for valid params
    if accel not in (None, "nesterov", "fista"):
        raise InvalidArgumentError("Unknown acceleration method")

    if accel == "nesterov":
        accel_coeff = (1 - np.sqrt(accel_args[0]/accel_args[1]))/(1 + np.sqrt(accel_args[0]/accel_args[1]))

    # Put these far enough apart to make sure the tolerance is exceeded
    xnew = x0 + 2*tol
    xold = x0

    k = 1

    cost_curr = cost(xnew)
    cost_old = cost_curr + 2*tol

    while LA.norm(cost_old - cost_curr) > tol and k < kmax:
        if accel is None or k == 1:
            xold = xnew
        elif accel == "nesterov":
            xold = xnew + accel_coeff*(xnew - xold)
        elif accel == "fista":
            xold = xnew + (k - 2)/(k + 1)*(xnew - xold)
        else:
            raise InvalidArgumentError("Only nesterov and fista accelerations are supported")

        yk = xold - step_size*gradf(xold)
        xnew = proxg(yk, lamb*step_size)

        cost_old = cost_curr
        cost_curr = cost(xnew)

        k += 1

    if k >= kmax:
        warnings.warn("kmax exceeded, consider raising it", NonConvergenceWarning)

    return xnew, k

def Prox_1Norm(v, theta):
    """Calculate the Prox operator for the 1 norm - used for Lasso"""
    ret = np.zeros(v.shape)
    for i in range(ret.shape[0]):
        if abs(v[i, 0]) >= theta:
            ret[i, 0] = v[i, 0] - theta*np.sign(v[i, 0])
    return ret

def Prox_2Norm(v, theta):
    """Calculate the Prox operator for the 2 norm"""
    return v/(theta + 1)

def Proj_1NormBall(v, r, tol=1e-5, kmax=1000):
    """
    Calculates the projection of v on the ball of radius r in the 1-norm

    Uses the bisection method to determine the t value within tolerance tol
    using a maximum of kmax steps
    """

    # Find t - Use Bisection Method
    n = v.shape[0]
    tmin = np.min(v) - r/n
    tmax = np.max(v) - r/n
    t = (tmax + tmin)/2.0

    diff = np.sum(np.maximum(np.abs(v) - t, 0)) - r
    k = 0

    while abs(diff) > tol and k < kmax:
        if diff == 0:
            # We're done
            break
        elif diff > 0:
            tmin = t
        else:
            tmax = t
        t = (tmax + tmin)/2.0
        diff = np.sum(np.maximum(np.abs(v) - t, 0)) - r
        k += 1

    if k >= kmax:
        warnings.warn("kmax exceeded, consider raising it", NonConvergenceWarning)

    # Use t to find the projection
    ret = np.zeros(v.shape)
    for i in range(ret.shape[0]):
        if v[i, 0] >= t:
            ret[i, 0] = v[i, 0] - t
        elif v[i, 0] <= -t:
            ret[i, 0] = v[i, 0] + t

    return ret

def Proj_2NormBall(v, r):
    """Calculates the projection of v on the ball of radius r in the 2-norm"""
    return v if LA.norm(v) <= r else r*v/LA.norm(v)

def Proj_InfNormBall(v, r):
    """Calculates the projection of v on the ball of radius r in the infinity-norm"""
    return np.maximum(np.minimum(v, r), -r)

def Proj_EqualityAffine(C, d, v):
    """Calculates the projection of b onto the affine subspace defined by Cx = d"""
    # Calculate thin QR decomp of C.T
    Q, R = LA.qr(C.T)

    # Find theta s.t. R^T*R theta = d
    # R^T*R is upper triangular and square so we use
    theta = LA.solve(np.matmul(R.T, R), d)

    # Find x0 as C^T*theta
    x0 = np.matmul(C.T, theta)

    # return the proj: x0 + v - Q*Q^T*v
    return x0 + v - np.matmul(np.matmul(Q, Q.T), v)

def Proj_InequalityAffine(A, b, v):
    """Calculates the projection of b onto the affine subspace defined by Ax >= b"""
    if all(np.matmul(A, v) - b >= 0):
        return v
    return Proj_EqualityAffine(A, b, v)

def Proj_Intersection(v, projs, tol=1e-7, kmax=1000):
    """
    Uses the alternating projections method to find the projection of v
    onto the intersection of all S_i

    Input Arguments:
    v           -- vector to project
    projs       -- tuple of callables, projs[i] accepts one argument v, and projects it into S_i
    tol         -- Error tolerance for stopping condition
                   -- Proj complete when distance between projections is < tol
    kmax        -- Maximum steps allowed, used for stopping condition

    Returns:
    If the optimal is found within tolerance
    x        -- Coordinates of the projection

    Example:
    # Pick a pick to project
    x0 = np.random.normal(size=(4, 1))

    # Set up an affine space to project into
    C = np.array([[2, 1, 1, 4], [1, 1, 2, 1]])
    d = np.array([[7, 6]]).T

    # Projector onto 2-norm ball of radius 2
    proj1 = (lambda v: Proj_2NormBall(v, 2))

    # Project onto affine Cx=d
    proj2 = (lambda v: Proj_EqualityAffine(C, d, v))

    # Now the Prox operator in the alternating method between the two
    proj = Proj_Intersection(v, (proj1, proj2)))
    """
    for i, c in enumerate(projs):
        if not callable(c):
            raise InvalidArgumentError(f"gradf[{i}] is not a callable")

    x = v.copy()
    xold = None
    k = 0
    diff = 2*tol
    # Loop until the projections converge
    while diff > tol and k < kmax:
        # Copy x to use for projections
        xold = x.copy()

        # Run all the projections
        for _, p in enumerate(projs):
            x = p(x)

        diff = LA.norm(x - xold)
        k += 1

    if k >= kmax:
        warnings.warn("kmax exceeded, consider raising it", NonConvergenceWarning)

    return x

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

        print(f"Test                 : {i}")
        print(f"m x n                : {m} x {n}")
        print(f"Cost Lasso           : {Lasso_cost}")
        print(f"Cost BB              : {BB_cost}")
        print(f"Iter Lasso           : {k_Lasso}")
        print(f"Iter BB              : {k_BB}")
        print(f"Time Lasso (sec)     : {Lasso_time}")
        print(f"Time BB    (sec)     : {BB_time}")
        print("===================================")

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

        print(f"Test                 : {i}")
        print(f"m x n                : {m} x {n}")
        print(f"Cost RR              : {RR_cost}")
        print(f"Cost BB              : {BB_cost}")
        print(f"Iter RR              : {k_RR}")
        print(f"Iter BB              : {k_BB}")
        print(f"Time RR (sec)        : {RR_time}")
        print(f"Time BB (sec)        : {BB_time}")
        print("===================================")

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

        print(f"Test                 : {i}")
        print(f"m x n                : {m} x {n}")
        print(f"Cost EN              : {EN_cost}")
        print(f"Cost BB              : {BB_cost}")
        print(f"Iter EN              : {k_EN}")
        print(f"Iter BB              : {k_BB}")
        print(f"Time EN    (sec)     : {EN_time}")
        print(f"Time BB    (sec)     : {BB_time}")
        print("===========================================")

def _test_Proj_1NormBall():
    r = 1
    v = np.array([[2, 3]]).T

    # Should be [0, 1]'
    proj = Proj_1NormBall(v, r)
    print("Projection should be [0, 1].T")
    print("proj: ")
    print(proj)
    print("===========================================")

def _test_Prob1():
    """
    min x_1^2 + x_2^2 + x_3^2 + x_4^2 - 2x_1 - 3x_4
    subj to:
      2x_1 + x_2 +  x_3 + 4x_4 = 7
       x_1 + x_2 + 2x_3 +  x_4 = 6
    """

    Q = np.eye(4)
    c = np.array([[-2, 0, 0, -3]]).T
    C = np.array([[2, 1, 1, 4], [1, 1, 2, 1]])
    d = np.array([[7, 6]]).T

    x0 = np.random.normal(size=(4, 1))
    gradf = (lambda x: 2*x + c)
    proxg = (lambda v, theta: Proj_EqualityAffine(C, d, v))
    lamb = 0.2
    tol = 1e-9
    step_size = 1e-4
    cost = (lambda x: np.matmul(x.T, np.matmul(Q, x)) + np.matmul(c.T, x))

    x_Iyer = np.array([[1.12, 0.65, 1.83, 0.57]]).T

    x, k = ProximalMethod(x0, gradf, proxg, lamb, tol, step_size, cost)
    print("x:")
    print(x)
    print(f"k = {k}")
    print(f"Cost of found solution: {cost(x)}")
    print(f"Cost of Iyer's solution: {cost(x_Iyer)}")
    print("Cx - d: ")
    print(np.matmul(C, x)-d)
    print("===========================================")

def _test_Prob2():
    """
    min x_1^2 + x_2^2 + x_3^2 + x_4^2 - 2x_1 - 3x_4
    subj to:
      2x_1 + x_2 +  x_3 + 4x_4 = 7
       x_1 + x_2 + 2x_3 +  x_4 = 6
                    norm_2(x) <= 3
    """
    Q = np.eye(4)
    c = np.array([[-2, 0, 0, -3]]).T
    C = np.array([[2, 1, 1, 4], [1, 1, 2, 1]])
    d = np.array([[7, 6]]).T

    x0 = np.random.normal(size=(4, 1))
    gradf = (lambda x: 2*x + c)

    # Project onto 2-norm ball of radius 3
    proj1 = (lambda v: Proj_2NormBall(v, 3))

    # Project onto affine Cx = d
    proj2 = (lambda v: Proj_EqualityAffine(C, d, v))

    # Now the Prox operator is the alternating method between the two
    proxg = (lambda v, theta: Proj_Intersection(v, (proj1, proj2), tol=1e-12))
    # proxg = (lambda v, theta: Proj_Intersection(v, (proj2, proj1), tol=1e-8))

    lamb = 0.005
    tol = 1e-10
    step_size = 1e-4
    # Use the cost function as the stopping criteria
    cost = (lambda x: np.matmul(x.T, np.matmul(Q, x)) + np.matmul(c.T, x))

    x_Iyer = np.array([[1.1234, 0.6506, 1.8288, 0.5684]]).T

    x, k = ProximalMethod(x0, gradf, proxg, lamb, tol, step_size, cost, kmax=1e6)
    print("x:")
    print(x)
    print(f"k = {k}")
    print(f"Cost of found solution: {cost(x)}")
    print(f"Cost of Iyer's solution: {cost(x_Iyer)}")
    print(f"Cx - d of found solution:")
    print(np.matmul(C, x)-d)
    print(f"Cx - d of Iyer solution:")
    print(np.matmul(C, x_Iyer)-d)
    print(f"2-norm of found solution: {LA.norm(x)}")
    print(f"2-norm of Iyer's solution: {LA.norm(x_Iyer)}")
    print("===========================================")

def _test_Prob3():
    """
    min x_1^2 + x_2^2 + x_3^2 + x_4^2 - 2x_1 - 3x_4
    subj to:
      2x_1 + x_2 +  x_3 + 4x_4 <= 7
       x_1 + x_2 + 2x_3 +  x_4 <= 6
                    norm_2(x) <= sqrt(2)
                      x_1, x_2 >= 0
    """
    Q = np.eye(4)
    c = np.array([[-2, 0, 0, -3]]).T
    A = -1*np.array([[2, 1, 1, 4], [1, 1, 2, 1]])
    b = -1*np.array([[7, 6]]).T

    x0 = np.random.normal(size=(4, 1))
    gradf = (lambda x: 2*x + c)

    # Project onto 2-norm ball of radius sqrt(2)
    proj1 = (lambda v: Proj_2NormBall(v, np.sqrt(2)))

    # Project onto First Octant
    proj2 = (lambda v: np.maximum(v, 0))

    # Project onto affine Ax >= b
    proj3 = (lambda v: Proj_InequalityAffine(A, b, v))

    # Now the Prox operator is the alternating method between the two
    proxg = (lambda v, theta: Proj_Intersection(v, (proj1, proj2, proj3), tol=1e-6))

    lamb = 0.005
    tol = 1e-6
    step_size = 1e-2
    # Use the cost function as the stopping criteria
    cost = (lambda x: np.matmul(x.T, np.matmul(Q, x)) + np.matmul(c.T, x))

    x_Iyer = np.array([[0.7827, 0.0, 0.0, 1.1741]]).T

    x, k = ProximalMethod(x0, gradf, proxg, lamb, tol, step_size, cost, kmax=1e6)
    print("x:")
    print(x)
    print(f"k = {k}")
    print(f"Cost of found solution: {cost(x)}")
    print(f"Cost of Iyer's solution: {cost(x_Iyer)}")
    print(f"Ax - b of found solution:")
    print(np.matmul(A, x)-b)
    print(f"Ax - b of Iyer solution:")
    print(np.matmul(A, x_Iyer)-b)
    print(f"2-norm of found solution: {LA.norm(x)}")
    print(f"2-norm of Iyer's solution: {LA.norm(x_Iyer)}")
    print("===========================================")

def _test_Prob4():
    """
    min x_1^2 + (x_1 + x_2)^2 - 10(x_1 + x_2)
    subject to: 3x_1 + x_2 <= 6
                norm2(x) <= sqrt(5)
    """
    Q = np.array([[2, 1], [1, 1]])
    c = np.array([[-10, -10]]).T
    A = -1*np.array([[3, 1]])
    b = -1*np.array([[6]]).T

    x0 = np.random.normal(size=(2, 1))
    gradf = (lambda x: 2*np.matmul(Q.T, x) + c)

    # Project onto 2-norm ball of radius sqrt(5)
    proj1 = (lambda v: Proj_2NormBall(v, np.sqrt(5)))

    # Project onto affine Ax >= b
    proj2 = (lambda v: Proj_InequalityAffine(A, b, v))

    # Now the Prox operator is the alternating method between the two
    proxg = (lambda v, theta: Proj_Intersection(v, (proj1, proj2), tol=1e-6))

    lamb = 0.005
    tol = 1e-8
    step_size = 1e-5
    # Use the cost function as the stopping criteria
    cost = (lambda x: np.matmul(x.T, np.matmul(Q, x)) + np.matmul(c.T, x))

    x_true = np.array([[1, 2]]).T

    x, k = ProximalMethod(x0, gradf, proxg, lamb, tol, step_size, cost, kmax=1e6)
    print("x:")
    print(x)
    print(f"k = {k}")
    print(f"Cost of found solution: {cost(x)}")
    print(f"Cost of true solution: {cost(x_true)}")
    print(f"Ax - b of found solution:")
    print(np.matmul(A, x)-b)
    print(f"Ax - b of true solution:")
    print(np.matmul(A, x_true)-b)
    print(f"2-norm of found solution: {LA.norm(x)}")
    print(f"2-norm of true solution: {LA.norm(x_true)}")
    print("===========================================")

def _test_Prob5():
    """
    min x_1 + x_2^2 + x_2*x_3 + 2x_3^2
    subject to: (1/2)*norm2(x) = 1
    """
    Q = np.zeros((3,3),dtype=float)
    Q[1,1] = 1.0
    Q[1,2] = 0.5
    Q[2,1] = 0.5
    Q[2,2] = 2
    c = np.array([[1, 0, 0]]).T

    x0 = np.random.normal(size=(3, 1))
    gradf = (lambda x: 2*np.matmul(Q.T, x) + c)

    # Project onto 2-norm ball of radius sqrt(5)
    proxg = (lambda v, theta: Proj_2NormBall(v, np.sqrt(2)))

    lamb = 0.005
    tol = 1e-9
    step_size = 1e-5
    # Use the cost function as the stopping criteria
    cost = (lambda x: np.matmul(x.T, np.matmul(Q, x)) + np.matmul(c.T, x))

    x_true = np.array([[-np.sqrt(2), 0, 0]]).T

    x, k = ProximalMethod(x0, gradf, proxg, lamb, tol, step_size, cost, kmax=1e6, accel="fista")
    print("x:")
    print(x)
    print(f"k = {k}")
    print(f"Cost of found solution: {cost(x)}")
    print(f"Cost of true solution: {cost(x_true)}")
    print(f"2-norm of found solution: {LA.norm(x)}")
    print(f"2-norm of true solution: {LA.norm(x_true)}")
    print("===========================================")

def _test_Nesterov_Accel_Prob1():
    """
    Using Nesterov Acceleration
    min x_1^2 + (x_1 + x_2)^2 - 10(x_1 + x_2)
    subject to: 3x_1 + x_2 <= 6
    """
    Q = np.array([[2, 1], [1, 1]])
    c = np.array([[-10, -10]]).T
    A = -1*np.array([[3, 1]])
    b = -1*np.array([[6]]).T

    x0 = np.random.normal(size=(2, 1))
    gradf = (lambda x: 2*np.matmul(Q.T, x) + c)

    # Prox is projection onto affine Ax >= b
    proxg = (lambda v, theta: Proj_InequalityAffine(A, b, v))

    lamb = 0.005
    tol = 1e-8
    step_size = 1e-5
    # Use the cost function as the stopping criteria
    cost = (lambda x: np.matmul(x.T, np.matmul(Q, x)) + np.matmul(c.T, x))

    # Set up the acceleration
    eigs = LA.eigvalsh(Q)
    accel_args = (min(eigs), max(eigs))

    x_true = np.array([[0, 5]]).T

    x, k = ProximalMethod(x0, gradf, proxg, lamb, tol, step_size, cost, kmax=1e6, accel="nesterov", accel_args=accel_args)
    print("x:")
    print(x)
    print(f"k = {k}")
    print(f"Cost of found solution: {cost(x)}")
    print(f"Cost of true solution: {cost(x_true)}")
    print(f"Ax - b of found solution:")
    print(np.matmul(A, x)-b)
    print(f"Ax - b of true solution:")
    print(np.matmul(A, x_true)-b)
    print("===========================================")

def _test_Nesterov_Accel_Prob2():
    """
    Using Nesterov Acceleration
    min x_1^2 + (x_1 + x_2)^2 - 10(x_1 + x_2)
    subject to: 3x_1 + x_2 <= 6
                norm2(x) <= sqrt(5)
    """
    Q = np.array([[2, 1], [1, 1]])
    c = np.array([[-10, -10]]).T
    A = -1*np.array([[3, 1]])
    b = -1*np.array([[6]]).T

    x0 = np.random.normal(size=(2, 1))
    gradf = (lambda x: 2*np.matmul(Q.T, x) + c)

    # Project onto 2-norm ball of radius 2
    proj1 = (lambda v: Proj_2NormBall(v, np.sqrt(5)))

    # Project onto affine Ax >= b
    proj2 = (lambda v: Proj_InequalityAffine(A, b, v))

    # Now the Prox operator is the alternating method between the two
    proxg = (lambda v, theta: Proj_Intersection(v, (proj1, proj2), tol=1e-6))

    lamb = 0.005
    tol = 1e-8
    step_size = 1e-5
    # Use the cost function as the stopping criteria
    cost = (lambda x: np.matmul(x.T, np.matmul(Q, x)) + np.matmul(c.T, x))

    # Set up the acceleration
    eigs = LA.eigvalsh(Q)
    accel_args = (min(eigs), max(eigs))

    x_true = np.array([[1, 2]]).T

    x, k = ProximalMethod(x0, gradf, proxg, lamb, tol, step_size, cost, kmax=1e6, accel="nesterov", accel_args=accel_args)
    print("x:")
    print(x)
    print(f"k = {k}")
    print(f"Cost of found solution: {cost(x)}")
    print(f"Cost of true solution: {cost(x_true)}")
    print(f"Ax - b of found solution:")
    print(np.matmul(A, x)-b)
    print(f"Ax - b of true solution:")
    print(np.matmul(A, x_true)-b)
    print(f"2-norm of found solution: {LA.norm(x)}")
    print(f"2-norm of true solution: {LA.norm(x_true)}")
    print("===========================================")

def _test_FISTA_Accel_Prob1():
    """
    Using Nesterov Acceleration
    min x_1^2 + (x_1 + x_2)^2 - 10(x_1 + x_2)
    subject to: 3x_1 + x_2 <= 6
    """
    Q = np.array([[2, 1], [1, 1]])
    c = np.array([[-10, -10]]).T
    A = -1*np.array([[3, 1]])
    b = -1*np.array([[6]]).T

    x0 = np.random.normal(size=(2, 1))
    gradf = (lambda x: 2*np.matmul(Q.T, x) + c)

    # Prox is projection onto affine Ax >= b
    proxg = (lambda v, theta: Proj_InequalityAffine(A, b, v))

    lamb = 0.005
    tol = 1e-8
    step_size = 1e-5
    # Use the cost function as the stopping criteria
    cost = (lambda x: np.matmul(x.T, np.matmul(Q, x)) + np.matmul(c.T, x))

    x_true = np.array([[0, 5]]).T

    x, k = ProximalMethod(x0, gradf, proxg, lamb, tol, step_size, cost, kmax=1e6, accel="fista")
    print("x:")
    print(x)
    print(f"k = {k}")
    print(f"Cost of found solution: {cost(x)}")
    print(f"Cost of true solution: {cost(x_true)}")
    print(f"Ax - b of found solution:")
    print(np.matmul(A, x)-b)
    print(f"Ax - b of true solution:")
    print(np.matmul(A, x_true)-b)
    print("===========================================")

def _test_FISTA_Accel_Prob2():
    """
    Using FISTA Acceleration
    min x_1^2 + (x_1 + x_2)^2 - 10(x_1 + x_2)
    subject to: 3x_1 + x_2 <= 6
                norm2(x) <= sqrt(5)
    """
    Q = np.array([[2, 1], [1, 1]])
    c = np.array([[-10, -10]]).T
    A = -1*np.array([[3, 1]])
    b = -1*np.array([[6]]).T

    x0 = np.random.normal(size=(2, 1))
    gradf = (lambda x: 2*np.matmul(Q.T, x) + c)

    # Project onto 2-norm ball of radius 2
    proj1 = (lambda v: Proj_2NormBall(v, np.sqrt(5)))

    # Project onto affine Ax >= b
    proj2 = (lambda v: Proj_InequalityAffine(A, b, v))

    # Now the Prox operator is the alternating method between the two
    proxg = (lambda v, theta: Proj_Intersection(v, (proj1, proj2), tol=1e-6))

    lamb = 0.005
    tol = 1e-8
    step_size = 1e-5
    # Use the cost function as the stopping criteria
    cost = (lambda x: np.matmul(x.T, np.matmul(Q, x)) + np.matmul(c.T, x))

    x_true = np.array([[1, 2]]).T

    x, k = ProximalMethod(x0, gradf, proxg, lamb, tol, step_size, cost, kmax=1e6, accel="fista")
    print("x:")
    print(x)
    print(f"k = {k}")
    print(f"Cost of found solution: {cost(x)}")
    print(f"Cost of true solution: {cost(x_true)}")
    print(f"Ax - b of found solution:")
    print(np.matmul(A, x)-b)
    print(f"Ax - b of true solution:")
    print(np.matmul(A, x_true)-b)
    print(f"2-norm of found solution: {LA.norm(x)}")
    print(f"2-norm of true solution: {LA.norm(x_true)}")
    print("===========================================")

def _Prox_Accel_Comparison():
    """
    Using FISTA Acceleration
    min x_1^2 + (x_1 + x_2)^2 - 10(x_1 + x_2)
    subject to: 3x_1 + x_2 <= 6
                norm2(x) <= sqrt(5)
    """
    import time

    Q = np.array([[2, 1], [1, 1]])
    c = np.array([[-10, -10]]).T
    A = -1*np.array([[3, 1]])
    b = -1*np.array([[6]]).T

    x0 = np.random.normal(size=(2, 1))
    gradf = (lambda x: 2*np.matmul(Q.T, x) + c)

    # Project onto 2-norm ball of radius 2
    proj1 = (lambda v: Proj_2NormBall(v, np.sqrt(5)))

    # Project onto affine Ax >= b
    proj2 = (lambda v: Proj_InequalityAffine(A, b, v))

    # Now the Prox operator is the alternating method between the two
    proxg = (lambda v, theta: Proj_Intersection(v, (proj1, proj2), tol=1e-6))

    lamb = 0.005
    tol = 1e-8
    step_size = 1e-5
    # Use the cost function as the stopping criteria
    cost = (lambda x: np.matmul(x.T, np.matmul(Q, x)) + np.matmul(c.T, x))

    # Set up the acceleration
    eigs = LA.eigvalsh(Q)
    accel_args = (min(eigs), max(eigs))

    start = time.time()
    _, k_unaccel = ProximalMethod(x0, gradf, proxg, lamb, tol, step_size, cost, kmax=1e6, accel=None)
    end = time.time()
    time_unaccel = end - start

    start = time.time()
    _, k_nesterov = ProximalMethod(x0, gradf, proxg, lamb, tol, step_size, cost, kmax=1e6, accel="nesterov", accel_args=accel_args)
    end = time.time()
    time_nesterov = end - start

    start = time.time()
    _, k_fista = ProximalMethod(x0, gradf, proxg, lamb, tol, step_size, cost, kmax=1e6, accel="fista")
    end = time.time()
    time_fista = end - start

    print("These methods have already been tested for accuracy.")
    print("We just compare iteration counts and timings")
    print(f"k (no accel)    = {k_unaccel}")
    print(f"k (nesterov)    = {k_nesterov}")
    print(f"k (FISTA)       = {k_fista}")
    print(f"time (no accel) = {time_unaccel} ")
    print(f"time (nesterov) = {time_nesterov}")
    print(f"time (FISTA)    = {time_fista}")
    print("===========================================")

if __name__ == "__main__":
    # _test_Lasso(5)
    # _test_RidgeRegression(5)

    # print("Elastic Net:")
    # print("===================================")
    # _test_ElasticNet(5)

    # print("Projection onto 1-Norm Ball:")
    # print("===================================")
    # _test_Proj_1NormBall()

    # print("Problem 1: Projection onto Cx=d Affine Set:")
    # print("===================================")
    # _test_Prob1()

    # print("Problem 2: Projection onto Cx=d Affine Set and 2-Norm Ball:")
    # print("===================================")
    # _test_Prob2()

    # print("Problem 3: Projection onto Ax >= b and 2-Norm Ball and Positive Xs:")
    # print("===================================")
    # _test_Prob3()

    # print("Problem 4: Projection onto Ax >= b and 2-Norm Ball:")
    # print("===================================")
    # _test_Prob4()

    print("Problem 5: Projection onto Norm Circle:")
    print("===================================")
    _test_Prob5()

    # print("Nesterov Acceleration Problem 1: Projection onto Ax >= b:")
    # print("===================================")
    # _test_Nesterov_Accel_Prob1()

    # print("Nesterov Acceleration Problem 2: Projection onto Ax >= b and 2-Norm Ball:")
    # print("===================================")
    # _test_Nesterov_Accel_Prob2()

    # print("FISTA Acceleration Problem 1: Projection onto Ax >= b:")
    # print("===================================")
    # _test_FISTA_Accel_Prob1()

    # print("FISTA Acceleration Problem 2: Projection onto Ax >= b and 2-Norm Ball:")
    # print("===================================")
    # _test_FISTA_Accel_Prob2()

    # print("Comparision of Acceleration Methods:")
    # print("===================================")
    # _Prox_Accel_Comparison()





    # print("Nothing here")
