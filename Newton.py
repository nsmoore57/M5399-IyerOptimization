#!/usr/bin/python
"""Module for Newton's Method code - Newton's Method for Rooting and Gradient Descent'"""
import numpy.linalg as LA
import numpy as np

def Newton(f, x0, tol, kmax, c=0.5, tao=1e-6, reg_const=1e-7):
    """
    Run Newton's Rootfinding method to find the the location of a zero of f

    Uses a central difference approximation to the Jacobian.
    Step size determined by the method in Stoer and Bulirsch.
    Also performs a regularization of the Jacobian if Central Difference Approx is singular

    Input Arguments:
    f         -- The function of which to find the zero
    x0        -- Initial guess for zero
                 - the closer the actual zero the better
    tol       -- Error tolerance for stopping condition
    kmax      -- Maximum steps allowed, used for stopping condition
    c         -- Constant between 0 and 1, used to determine step sizes
    tao       -- Perturbation size for Central Difference Approx to Jacobian
    reg_const -- Perturbation to add to Central Difference Jacobian to regularize
                 in the case that the Jacobian is singular

    Returns coordinates, x, such that norm_2(f(x)) < tol if found, None otherwise

    Example:
    x0 = np.array([10], dtype="float")
    tao = 1e-5
    c = 0.5
    tol = 1e-4
    kmax = 1e3
    print(Newton((lambda x: np.arctan(x-np.pi/4)), x0, tao, c, tol, kmax))
    """

    k = 1

    xk = x0.copy() # We don't want to change x0 from the outside scope

    # Cache for f(xk) in case the calculation is expensive
    #  - Should save 2-3 function evaluations per outside loop iteration
    f_xk = f(xk)

    s = LA.norm(f_xk)**2

    while np.sqrt(s) > tol and k < kmax:
        # Build the Jacobian matrix approximation using Central Differences
        H = _CentralDifferencesJacobian(f, xk, tao)

        try:
            # Uses LAPACK _gesv to solve H*dk = -f(xk)
            dk = LA.solve(H, -f_xk)
        except LA.LinAlgError:
            # Most likely here because the above H is singular
            # Therefore we regularize by adding a small (1e-7) multiple of I:
            theta = 1e-7
            dk = LA.solve(H + theta*np.eye(xk.shape[0]), -f_xk)

        z = c*LA.norm(np.matmul(f_xk.transpose(),H))*LA.norm(dk)

        # Solve the step size using the method by Stoer and Bulirsch
        j = 0
        L = LA.norm(f(xk + dk))**2
        R = s-z
        Lmin = L
        index = 0
        while L > R:
            j += 1
            L = LA.norm(f(xk + 2**(-j)*dk))**2
            R = s - 2**(-j)*z
            if L < Lmin:
                Lmin = L
                index = j

        # Update xk
        xk = xk + 2**(-index)*dk

        k += 1

        # Cache the new f(xk)
        f_xk = f(xk)

        s = LA.norm(f_xk)**2

    # If kmax gets exceeded, we can't trust the answer so return None
    if k >= kmax:
        # For debugging purposes
        # print("k exceeded kmax, can't trust answer")
        # return xk
        return None

    # Otherwise, we stopped the above loop because we're within tolerance so the answer is good
    return xk

def GradDescent_BB(f, x0, tol, kmax):
    """
    Run Gradient Descent to find the location of a zero of f
    Uses a central difference approximation to the Jacobian.
    Step size determined by the method in Stoer and Bulirsch.

    Input Arguments:
    f           -- The function of which to find the zero
    x0          -- Initial guess for zero
                   - the closer the actual zero the better
    tol         -- Error tolerance for stopping condition
    kmax        -- Maximum steps allowed, used for stopping condition

    Returns coordinates, x, such that norm_2(f(x)) < tol if found, None otherwise

    Example:
    x0 = np.array([10], dtype="float")
    tol = 1e-4
    kmax = 1000
    print(GradDescent_BB((lambda x: np.arctan(x-np.pi/4)), x0, tol, kmax))
    """

    k = 1
    xold = np.zeros(x0.shape)
    xnew = x0
    dold = np.zeros(x0.shape)
    while LA.norm(f(xnew)) > tol and k < kmax:
        # Step direction
        dnew = -f(xnew)

        # Step size
        gamma = np.matmul((xnew - xold).transpose(), dnew - dold)/LA.norm(dnew-dold)**2

        # Reset all old values to avoid evaluating f more times than necessary
        xold = xnew
        dold = dnew
        xnew = xold - gamma*dnew
        k += 1

    # If kmax gets exceeded, we can't trust the answer so return None
    if k >= kmax:
        # For debugging purposes
        # print("k exceeded kmax, can't trust answer")
        # return xk
        return None

    # Otherwise, we stopped the above loop because we're within tolerance so the answer is good
    return xnew


def GradDescent_ILS(q, gradq, x0, tol, kmax, a_low=1e-9, a_high=0.9, N=20, CD_tao = 1e-6):
    """
    Run Gradient Descent to find the location of a zero of f
    Uses a central difference approximation to the Jacobian.
    Step size determined by an inexact line search

    Input Arguments:
    q           -- The function of which to find the mininum
    gradq       -- The gradient of q - use "CD" for Central Difference Approx
    x0          -- Initial guess for zero
                   - the closer the actual zero the better
    tol         -- Error tolerance for stopping condition
    kmax        -- Maximum steps allowed, used for stopping condition
    a_low       -- Proportion of slope for low end of search
    a_high      -- Proportion of slope for high end of search
    N           -- Number of points in logarthmic grid between a_low and a_high
                   - Lower is faster but less accurate
    CD_tao      -- Perturbation of x for approximation of gradient using CD
                   -- Ignored if gradq != "CD"

    Returns coordinates, x, of the minimum (where grad q = 0) if found within tolerance, otherwise None

    Example:
    def Rosenbrock2(x):
        return (2-x[0])**2 + 100*(x[1]-x[0]**2)**2
    def GRosenbrock2(x):
        dfdx1 = -2*(2-x[0]) - 400*(x[1]-x[0]**2)*x[0]
        dfdx2 = 200*(x[1]-x[0]**2)
        return np.vstack((dfdx1, dfdx2))

    x0 = np.array([[3], [3]], dtype="float")
    tol = 1e-4
    kmax = 50000
    a_low = 1e-9
    a_high = 0.7
    N = 20
    CD_tao = 1e-5

    print(GradDescent_ILS(Rosenbrock2, GRosenbrock2, x0, tol, kmax, a_low=a_low, a_high=a_high, N=N, CD_tao=CD_tao))
    """

    # Use CentralDifferences to approximate the gradient
    if type(gradq) == str and gradq == "CD":
        # Lambda function so that we can pass function and perturbation through
        gradq = (lambda x:_CentralDifferencesGradient(q, x, CD_tao))
    # If not a function and not "CD" then error
    elif not callable(gradq):
        print("Undefined gradq - should be a function or CD")
        return None


    # Iteration Counter
    k = 1

    # Make a copy in case we try to change it
    xk = x0.copy()

    # Cache gradq(xk) and q(xk) to speed up computation
    gq_cache = gradq(xk)
    q_cache = q(xk)

    # Alpha grid for step size search
    alpha = np.logspace(np.log10(a_low), np.log10(a_high), N, endpoint=True)

    while LA.norm(gq_cache) > tol*(1 + np.abs(q_cache)) and k < kmax:
        # Track minimum phi value in the grid
        phimin = None
        imin = None

        # Move along the direction of the gradient, testing q at each grid point
        for i in range(N):
            phi = q(xk - alpha[i]*gq_cache/LA.norm(gq_cache)) - q_cache

            # Want the minimum phi value
            if phimin == None or phi < phimin:
                phimin = phi
                imin = i

        # Not good, probably a step size issue
        if phimin > 0:
            print("No more steps to take downward, don't trust answer. Probably a step size issue.  Consider increasing N")
            return xk

        # Update xk
        xk -= alpha[imin]*(gq_cache/LA.norm(gq_cache))

        # print(xk)

        # Cache new values
        gq_cache = gradq(xk)
        q_cache = q(xk)

        # Increase iteration count
        k += 1

    # If kmax gets exceeded, we can't trust the answer so return None
    if k >= kmax:
        # For debugging purposes
        print("k exceeded kmax, can't trust answer")
        return xk
        # return None

    # Otherwise, we stopped the above loop because we're within tolerance so the answer is good
    return xk

def GradDescent_Armijo(q, gradq, x0, tol, kmax, a_low=1e-9, a_high=0.9, N=20, c_low=0.1, c_high=0.9, CD_tao = 1e-6):
    """
    Run Gradient Descent to find the location of a zero of f
    Uses a central difference approximation to the Jacobian.
    Step size determined by the Armijo condition

    Input Arguments:
    q           -- The function of which to find the mininum
    gradq       -- The gradient of q - use "CD" for Central Difference Approx
    x0          -- Initial guess for zero
                   - the closer the actual zero the better
    tol         -- Error tolerance for stopping condition
    kmax        -- Maximum steps allowed, used for stopping condition
    a_low       -- Proportion of gradient direction for low end of search
    a_high      -- Proportion of gradient direction for high end of search
    N           -- Number of points in logarthmic grid between a_low and a_high
                   - Lower is faster but less accurate
    c_low       -- Proportion of slope for low end of allowed step sizes
    c_high      -- Proportion of slope for high end of allowed step sizes
    CD_tao      -- Perturbation of x for approximation of gradient using CD
                   -- Ignored if gradq != "CD"

    Returns coordinates, x, of the minimum (where grad q = 0) if found within tolerance, otherwise None

    Example:
    def Rosenbrock2(x):
        return (2-x[0])**2 + 100*(x[1]-x[0]**2)**2
    def GRosenbrock2(x):
        dfdx1 = -2*(2-x[0]) - 400*(x[1]-x[0]**2)*x[0]
        dfdx2 = 200*(x[1]-x[0]**2)
        return np.vstack((dfdx1, dfdx2))

    x0 = np.array([[3], [3]], dtype="float")
    tol = 1e-4
    kmax = 50000
    a_low = 1e-9
    a_high = 0.7
    N = 20
    c_low = 0.1
    c_high = 0.8
    CD_tao = 1e-5

    print(GradDescent_Armijo(Rosenbrock2, GRosenbrock2, x0, tol, kmax, a_low=a_low, a_high=a_high, N=N, c_low=c_low, c_high=c_high, CD_tao=CD_tao))
    """

    # Use CentralDifferences to approximate the gradient
    if type(gradq) == str and gradq == "CD":
        # Lambda function so that we can pass function and perturbation through
        gradq = (lambda x:_CentralDifferencesGradient(q, x, CD_tao))
    # If not a function and not "CD" then error
    elif not callable(gradq):
        print("Undefined gradq - should be a function or CD")
        return None


    # Iteration Counter
    k = 1

    # Make a copy in case we try to change it
    xk = x0.copy()

    # Cache gradq(xk), q(xk), and norm(gradq(xk)) to speed up computation
    gq_cache = gradq(xk)
    q_cache = q(xk)
    n_gq_cache = LA.norm(gq_cache)

    # Alpha grid for step size search
    alpha = np.logspace(np.log10(a_low), np.log10(a_high), N, endpoint=True)

    while n_gq_cache > tol*(1 + np.abs(q_cache)) and k < kmax:
        # Track minimum phi value in the grid
        phimin = None
        imin = None

        # Move along the direction of the gradient, testing q at each grid point
        for i in range(N):
            phi = q(xk - alpha[i]*gq_cache) - q_cache
            h = -c_low*alpha[i]*n_gq_cache*n_gq_cache
            l = -c_high*alpha[i]*n_gq_cache*n_gq_cache

            # Want the minimum phi value s.t. phi <= h and phi >= l
            if (phi <= h and phi >= l) and (phimin == None or phi < phimin):
                phimin = phi
                imin = i

        # Not good, probably a step size issue
        if phimin == None:
            print("No more steps to take downward, don't trust answer. Probably a step size issue.  Consider increasing N")
            return xk

        # Update xk
        xk -= alpha[imin]*(gq_cache/LA.norm(gq_cache))

        # print(xk)

        # Cache new values
        gq_cache = gradq(xk)
        n_gq_cache = LA.norm(gq_cache)
        q_cache = q(xk)

        # Increase iteration count
        k += 1

    # If kmax gets exceeded, we can't trust the answer so return None
    if k >= kmax:
        # For debugging purposes
        print("k exceeded kmax, can't trust answer")
        return xk
        # return None

    # Otherwise, we stopped the above loop because we're within tolerance so the answer is good
    return xk

# Not exported with Module
def _CentralDifferencesJacobian(f, x, tao):
    """
    Calculates an approximation to the Jacobian based on Central Differences:

    Input Arguments:
    f    -- function to approximate the Jacobian from
    x    -- The point to approximate the Jacobian around
    tao  -- 2*tao is length of the secant line between central difference points

    H(:,i) = 1/(2tao) (f(x + tao*ei) - f(x - tao*ei))
    """

    # Preallocate a matrix
    H = np.empty((x.shape[0], x.shape[0]), dtype="float")

    for i in range(x.shape[0]):
        xhigh = x.copy()
        xlow = x.copy()
        xhigh[i] += tao
        xlow[i] -= tao

        H[:, i] = (1/(2*tao)*(f(xhigh) - f(xlow))).flatten()
    return H

# Not exported with Module
def _CentralDifferencesGradient(f, x, tao):
    """
    Calculates an approximation to the Gradient based on Central Differences:

    Input Arguments:
    f    -- function to approximate the Gradient from - R^n -> R
    x    -- The point to approximate the Gradient at
    tao  -- 2*tao is length of the secant line between central difference points

    H(i) = 1/(2tao) (f(x + tao*ei) - f(x - tao*ei))
    """

    # Preallocate a matrix
    H = np.empty((x.shape[0], 1), dtype="float")

    for i in range(x.shape[0]):
        xhigh = x.copy()
        xlow = x.copy()
        xhigh[i] += tao
        xlow[i] -= tao

        H[i] = (1/(2*tao)*(f(xhigh) - f(xlow)))
    return H

# Testing from module load
if __name__ == "__main__":
    def Rosenbrock(a,x):
        return (a-x[0])**2 + 100*(x[1]-x[0]**2)**2
    def Grad_Rosenbrock(a, x):
        """
        Gradient of the Rosenbrock function
        Used for obtaining the mininum of the Rosenbrock
        """
        dfdx1 = -2*(a-x[0]) - 400*(x[1]-x[0]**2)*x[0]
        dfdx2 = 200*(x[1]-x[0]**2)
        return np.vstack((dfdx1, dfdx2))

    GRosenbrock1 = (lambda x: Grad_Rosenbrock(1, x))
    GRosenbrock2 = (lambda x: Grad_Rosenbrock(2, x))
    Rosenbrock2 = (lambda x: Rosenbrock(2, x))

    x0 = np.array([[3], [3]], dtype="float")
    tol = 1e-4
    kmax = 50000
    a_low = 1e-9
    a_high = 0.7
    N = 20
    CD_tao = 1e-5

    # print(GradDescent_Armijo(Rosenbrock2, GRosenbrock2, x0, tol, kmax, a_low=a_low, a_high=a_high, N=N, CD_tao=CD_tao))
    # print(GradDescent_ILS(Rosenbrock2, "CD", x0, tol, kmax, a_low=a_low, a_high=a_high, N=N, CD_tao=CD_tao))

    x0 = np.array([[3], [3]], dtype="float")
    tol = 1e-4
    kmax = 80000
    a_low = 1e-10
    a_high = 0.9
    N = 20
    c_low = 0.1
    c_high = 0.8
    CD_tao = 1e-5

    print(GradDescent_Armijo(Rosenbrock2, GRosenbrock2, x0, tol, kmax, a_low=a_low, a_high=a_high, N=N, c_low=c_low, c_high=c_high, CD_tao=CD_tao))
    # print(GradDescent_Armijo(Rosenbrock2, "CD", x0, tol, kmax, a_low=a_low, a_high=a_high, N=N, c_low=c_low, c_high=c_high, CD_tao=CD_tao))
