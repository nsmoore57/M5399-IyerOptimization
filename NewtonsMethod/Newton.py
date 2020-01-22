#!/usr/bin/python
"""Module for Newton's Method code - Newton's Method for Rooting and Gradient Descent'"""
import numpy.linalg as LA
import numpy as np

def Newton(f, x0, tao, c, tol, kmax):
    """
    Run Newton's Rootfinding method to find the zero of f
    """
    k = 1
    s = LA.norm(f(x0))**2
    xk = x0.copy() # We don't want to change x0 from the outside scope

    while np.sqrt(s) > tol and k < kmax:
        # Build the Jacobian matrix approximation using Central Differences
        H = _CentralDifferences(f, xk, tao)

        try:
            # Uses LAPACK _gesv to solve H*dk = -f(xk)
            dk = LA.solve(H, -f(xk))
        except LA.LinAlgError:
            # Most likely here because the above H is singular
            # Therefore we regularize by adding a small (10^-7) multiple of I:
            theta = 10**(-7)
            dk = LA.solve(H + theta*np.eye(xk.shape[0]), -f(xk))

        z = c*LA.norm(f(xk).transpose()*H)*LA.norm(dk)

        # Solve the step size using the method by Stoer and Bulirsch
        j = 0
        L = LA.norm(f(xk + dk))**2
        R = s-z
        Lmin = L
        index = 0
        while L > R:
            j += 1
            L = LA.norm(f(xk + 2**(-j)))**2
            R = s - 2**(-j)*z
            if L < Lmin:
                Lmin = L
                index = j

        # print("dk = " + str(dk))
        # print("index = " + str(index))
        # Update xk
        xk = xk + 2**(-index)*dk
        # print("xk = " + str(xk))
        k += 1
        s = LA.norm(f(xk))**2

    # If kmax gets exceeded, we can't trust the answer so return None
    if k >= kmax:
        print("k exceeded kmax, can't trust answer")
        return None

    # Otherwise, we stopped the above loop because we're within tolerance so the answer is good
    return xk

def GradDescent(f, x0, tol, kmax):
    """Run Gradient Descent to find the zero of f"""
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
        print("k exceeded kmax, can't trust answer")
        return None

    # Otherwise, we stopped the above loop because we're within tolerance so the answer is good
    return xnew

def _CentralDifferences(f, x, tao):
    """
    Calculates an approximation to the Jacobian based on Central Differences:
    -------------------------------------------------------------------------
    f: function to approximate the Jacobian of
    x: the point to approximate the Jacobian around
    tao: 2*tao is the length of secant line between central difference points

    H(:,i) = 1/(2tao) (f(x + tao*ei) - f(x - tao*ei))
    """
    H = np.empty((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        xhigh = x.copy()
        xlow = x.copy()
        xhigh[i] += tao
        xlow[i] -= tao

        H[:, i] = (1/(2*tao)*(f(xhigh) - f(xlow))).flatten()
    return H

if __name__ == "__main__":
    def Grad_Rosenbrock(a, x):
        """
        Gradient of the Rosenbrock function
        Used for obtaining the mininum of the Rosenbrock
        """
        dfdx1 = -2*(a-x[0]) - 400*(x[1]-x[0]**2)*x[0]
        dfdx2 = 200*(x[1]-x[0]**2)
        return np.vstack((dfdx1, dfdx2))

    Rosenbrock1 = (lambda x: Grad_Rosenbrock(1, x))

    def arctan2d(x):
        """
        2D arctan function for testing GradDescent
        """
        res1 = np.arctan(x[0] - np.pi/4)
        res2 = np.arctan(x[1] - np.pi/4)
        return np.vstack((res1, res2))

    x0 = np.array([[1.01], [1.01]], dtype="float")
    tao = 10**(-5)
    c = 0.5
    tol = 10**(-2)
    kmax = 100000
    print(Newton(Rosenbrock1, x0, tao, c, tol, kmax))
