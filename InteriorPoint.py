#!/usr/bin/python
"""Module for Interior Point Methods
    TODO: Custom Exceptions to raise when errors occur
    See https://www.programiz.com/python-programming/user-defined-exception
"""
import numpy as np
import numpy.linalg as LA

def InteriorPointBarrier(c, A, b, tol, kmax, rho, mu0, mumin):
    """
    Run Interior Point Barrier Method to solve the optimzation problem:
    min c^T*x subject to Ax = b and x >= 0,

    That is, it solves
    min c^T*x - mu*sum(ln(x_i)) subject to Ax = b
    using a decreasing mu value to keep the candidate solution contained in x >= 0

    Automatically finds a point in the feasible region to start

    Input Arguments:
    c         -- The vector describing the linear function to minimize
    A         -- Matrix of coefficients for linear constraints
    b         -- The Right-hand side vector for the linear constraints
    tol       -- Error tolerance for stopping condition
    kmax      -- Maximum steps allowed, used for stopping condition
    rho       -- Used for decreasing the strength of the barrier
                 - at each iteration mu becomes rho*mu
    mumin     -- Minimum strength of the barrier

    Returns:
    If the optimal is found within tolerance
    x        -- Coordinates of the optimal value
    k        -- The number of total iterations required
                - Includes the iterations needed to find the first feasible point
    If the optimal is not found within tolerance, return None, "Error Message"

    Example:
    A = np.array([[1, 2, -1, 1],[2, -2, 3, 3],[1, -1, 2, -1]],dtype="float")
    b = np.array([[0, 9, 6]]).transpose()
    c = np.array([[-3, 1, 3, -1]]).transpose()
    tol = 1e-5
    kmax = 10000
    rho = .9
    mu0 = 1e4
    mumin = 1e-8

    x, k = InteriorPointBarrier(c, A, b, tol, kmax, rho, mu0, mumin)

    TODO: Add checks for error messages and pass them out of the function
    """
    m,n = A.shape
    totalk = 0
    # Phase I - find a solution in the feasible region
    Q_p1 = np.eye(n)
    c_p1 = np.ones((n,1))
    x = np.ones((n,1))
    lamb = np.zeros((m,1))
    s = np.ones((n,1))

    x,lamb,s,k = _IPBarrier_Worker(Q_p1, c_p1, A, b, x, lamb, s, tol, kmax, rho, mu0, mumin)
    totalk += k

    # Phase II
    Q_p2 = np.zeros((n,n))
    c_p2 = c.copy()

    x,lamb,s,k = _IPBarrier_Worker(Q_p2, c_p2, A, b, x, lamb, s, tol, kmax, rho, mu0, mumin)
    totalk += k

    return x,totalk


def _F(A, Q, b, x, s, lamb, c, mu):
    m,n = A.shape
    F_row1 = np.matmul(A,x) - b
    F_row2 = np.matmul(-Q,x) + np.matmul(A.transpose(),lamb) + s - c
    F_row3 = np.array([[x[i,0]*s[i,0]] for i in range(n)]) - mu*np.ones((n,1))
    return np.vstack((F_row1, F_row2, F_row3))

def _IPBarrier_Worker(Q, c, A, b, x, lamb, s, tol, kmax, rho, mu0, mumin):
    """
    Runs the interior point method for constrained optimization where
    the constraints are Ax = b

    TODO: Better docstring
    TODO: Cache values calculated more than once
    """
    if A.shape[0] != b.shape[0]:
        print("sizes of A and b don't match")
        return None

    def Jacobian():
        J_row1 = np.hstack((A, np.zeros((m,m)), np.zeros((m,n))))
        J_row2 = np.hstack((-1*Q, A.transpose(), np.eye(n)))
        J_row3 = np.hstack((np.diagflat(s),np.zeros((n,m)),np.diagflat(x)))
        return np.vstack((J_row1, J_row2, J_row3))

    m,n = A.shape
    mu = mu0

    k = 1
    r = -_F(A, Q, b, x, s, lamb, c, mu)
    normr = LA.norm(r)**2

    while LA.norm(r) > tol and k < kmax and mu > mumin:
        # Calculate the Jacobian
        J = Jacobian()

        # Solve Jd = r
        d = LA.solve(J,r)

        # Split the d apart into the different component pieces
        dx = d[:n,0].reshape((-1,1))
        dlamb = d[n:n+m,0].reshape((-1,1))
        ds = d[n+m:,0].reshape((-1,1))
        z = 0.9*LA.norm(np.matmul(r.transpose(),J))*LA.norm(d)

        # Select the largest alpha_0 with 0 <= alpha_0 <=1 so that x + alpha_0*dx > 0 and s + alpha*ds > 0
        # This will take some work to implement - probably a pythonic way to do this
        if all(v > 0 for v in dx) and all(v > 0 for v in ds):
            alpha_bar = 1
        else:
            alpha_bar = None
            # search for min(-xk[i]/dxk[i]) among i s.t. dx[i] < 0
            for i in range(dx.shape[0]):
                if dx[i] < 0:
                    t = -x[i]/dx[i]
                    if alpha_bar == None or t < alpha_bar:
                        alpha_bar = t
            # search for min(-sk[i]/dsk[i]) among i s.t. ds[i] < 0
            for i in range(ds.shape[0]):
                if s[i] < 0:
                    t = -s[i]/ds[i]
                    if alpha_bar == None or t < alpha_bar:
                        alpha_bar = t

        alpha_0 = 0.99995*min(alpha_bar,1)


        # Set L and R
        L = LA.norm(_F(A, Q, b, x + alpha_0*dx, s + alpha_0*ds, lamb + alpha_0*dlamb, c, mu))**2
        R = normr - alpha_0*z

        # Find min L
        Lmin = L
        index = 0

        j = 0
        while L > R and j < 1000:
            j += 1
            # Calculate L, R
            L = LA.norm(_F(A, Q, b, x + 2**(-j)*alpha_0*dx, s + 2**(-j)*alpha_0*ds, lamb + 2**(-j)*alpha_0*dlamb, c, mu))**2
            R = normr - 2**(-j)*alpha_0*z
            if L < Lmin:
                Lmin = L
                index = j
        # Check if we
        if j >= 1000:
            return None, None, None,"Good step size couldn't be found"
        # Update the Solution
        x += 2**(-index)*alpha_0*dx
        lamb += 2**(-index)*alpha_0*dlamb
        s += 2**(-index)*alpha_0*ds

        k += 1
        mu = rho*mu
        r = -_F(A, Q, b, x, s, lamb, c, mu)
        normr = LA.norm(r)**2

    if k > kmax:
        return None,None,None,"kmax exceeded, consider raising it"
    if mu < mumin:
        return None,None,None,"mu became smaller than mumin before reaching convergence. Consider lowering mumin"

    return x,lamb,s,k

if __name__ == "__main__":
    A = np.array([[1, 2, -1, 1],[2, -2, 3, 3],[1, -1, 2, -1]],dtype="float")
    b = np.array([[0, 9, 6]]).transpose()
    c = np.array([[-3, 1, 3, -1]]).transpose()
    tol = 1e-5
    kmax = 10000
    rho = .9
    mu0 = 1e4
    mumin = 1e-8

    x,k = InteriorPointBarrier(c, A, b, tol, kmax, rho, mu0, mumin)
    print(LA.norm(x-np.array([[1, 1, 3, 0]]).transpose()))
