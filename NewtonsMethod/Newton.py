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
    Also performs a regularization of the Jacobian if Central Difference Approx is singular

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
    print(GradDescent((lambda x: np.arctan(x-np.pi/4)), x0, tol, kmax))
    """

    k = 1
    xold = np.zeros(x0.shape)
    xnew = x0
    dold = np.zeros(x0.shape)
    while LA.norm(f(xnew)) > tol and k < kmax:
        # Step direction
        dnew = -f(xnew)

        # Step size
        gamma = np.matmul((xnew - xold).transpose(), -dnew + dold)/LA.norm(-dnew+dold)**2

        # Reset all old values to avoid evaluating f more times than necessary
        xold = xnew
        dold = dnew
        xnew = xold + gamma*dnew
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
    Step size determined by the method in Stoer and Bulirsch.
    Also performs a regularization of the Jacobian if Central Difference Approx is singular

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

    Returns coordinates, x, such that norm_2(f(x)) < tol if found, None otherwise

    Example:
    x0 = np.array([10], dtype="float")
    tol = 1e-4
    kmax = 1000
    print(GradDescent((lambda x: np.arctan(x-np.pi/4)), x0, tol, kmax))
    """

    if type(gradq) == str and gradq == "CD":
        gradq = (lambda x:_CentralDifferencesGradient(q, x, CD_tao))
    elif type(gradq) != function:
        print("Undefined gradq - should be a function or CD")

    k = 1
    alpha = np.logspace(a_low, a_high, N, endpoint=True)
    xk = x0
    while LA.norm(gradq(xk)) > tol*(1 + np.abs(q(xk))) and k < kmax:
        

        # Step size
        gamma = np.matmul((xnew - xold).transpose(), -dnew + dold)/LA.norm(-dnew+dold)**2

        # Reset all old values to avoid evaluating f more times than necessary
        xold = xnew
        dold = dnew
        xnew = xold + gamma*dnew
        k += 1

    # If kmax gets exceeded, we can't trust the answer so return None
    if k >= kmax:
        # For debugging purposes
        # print("k exceeded kmax, can't trust answer")
        # return xk
        return None

    # Otherwise, we stopped the above loop because we're within tolerance so the answer is good
    return xnew

# Not exported with Module
def _CentralDifferencesJacobian(f, x, tao):
    """
    Calculates an approximation to the Jacobian based on Central Differences:

    Input Arguments:
    f    -- function to approximate the Jacobian from
    x    -- The point to approximate the Jacobian around
    tao  -- 2*taco is length of the secant line between central difference points

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
    tao  -- 2*taco is length of the secant line between central difference points

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
    tao = 1e-5
    print(GRosenbrock2(x0))
    print(_CentralDifferencesGradient(Rosenbrock2, x0, tao))
