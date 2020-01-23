#!/usr/bin/python
"""Module for Newton's Method code - Newton's Method for Rooting and Gradient Descent'"""
import numpy.linalg as LA
import numpy as np

def Newton(f, x0, tao, c, tol, kmax):
    """
    Run Newton's Rootfinding method to find the the location of a zero of f

    Uses a central difference approximation to the Jacobian.
    Step size determined by the method in Stoer and Bulirsch.
    Also performs a regularization of the Jacobian if Central Difference Approx is singular

    Input Arguments:
    f    -- The function of which to find the zero
    x0   -- Initial guess for zero
            - the closer the actual zero the better
    tao  -- Perturbation size for Central Difference Approx to Jacobian
            - usually around 10**(-6)
    c    -- Constant between 0 and 1, used to determine step sizes
            - usually 0.5
    tol  -- Error tolerance for stopping condition
    kmax -- Maximum steps allowed, used for stopping condition
    
    Returns coordinates, x, such that norm_2(f(x)) < tol if found, None otherwise

    Example:
    x0 = np.array([10], dtype="float")
    tao = 10**(-5)
    c = 0.5
    tol = 10**(-4)
    kmax = 1000
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
        H = _CentralDifferences(f, xk, tao)
        
        try:
            # Uses LAPACK _gesv to solve H*dk = -f(xk)
            dk = LA.solve(H, -f_xk)
        except LA.LinAlgError:
            # Most likely here because the above H is singular
            # Therefore we regularize by adding a small (10^-7) multiple of I:
            theta = 10**(-7)
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

def GradDescent(f, x0, tol, kmax):
    """
    Run Gradient Descent to find the location of a zero of f
    Uses a central difference approximation to the Jacobian.
    Step size determined by the method in Stoer and Bulirsch.
    Also performs a regularization of the Jacobian if Central Difference Approx is singular

    Input Arguments:
    f    -- The function of which to find the zero
    x0   -- Initial guess for zero
            - the closer the actual zero the better
    tol  -- Error tolerance for stopping condition
    kmax -- Maximum steps allowed, used for stopping condition
    
    Returns coordinates, x, such that norm_2(f(x)) < tol if found, None otherwise

    Example:
    x0 = np.array([10], dtype="float")
    tol = 10**(-4)
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
        gamma = abs(np.matmul((xnew - xold).transpose(), -dnew + dold))/LA.norm(-dnew+dold)**2

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
def _CentralDifferences(f, x, tao):
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

# Testing from module load
# if __name__ == "__main__":
#     def Grad_Rosenbrock(a, x):
#         """
#         Gradient of the Rosenbrock function
#         Used for obtaining the mininum of the Rosenbrock
#         """
#         dfdx1 = -2*(a-x[0]) - 400*(x[1]-x[0]**2)*x[0]
#         dfdx2 = 200*(x[1]-x[0]**2)
#         return np.vstack((dfdx1, dfdx2))

#     Rosenbrock1 = (lambda x: Grad_Rosenbrock(1, x))
#     Rosenbrock2 = (lambda x: Grad_Rosenbrock(2, x))

#     x0 = np.array([[3], [3]], dtype="float")
#     tao = 10**(-5)
#     c = 0.5
#     tol = 10**(-1)
#     kmax = 100000
#     print(LA.norm(Rosenbrock1(Newton(Rosenbrock1, x0, tao, c, tol, kmax))))
