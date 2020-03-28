#!/usr/bin/python
"""Module for Interior Point Methods"""
import numpy as np
import numpy.linalg as LA
import LinearAlgebra as myLA

# Define Exceptions
class IPError(Exception):
    """Base class for other exceptions"""
    pass

class NonConvergenceError(IPError):
    """For Errors where convergence is not complete"""
    pass

class DimensionMismatchError(IPError):
    """For errors involving incorrect dimensions"""
    pass


def Barrier_EqualityOnly(Q, c, A, b, tol, kmax=1000, rho=.9, mu0=1e2, mumin=1e-9):
    """
    Run Interior Point Barrier Method to solve the quadratic programming problem:
    min (1/2)x^T*Q*x + c^T*x subject to Ax = b and x >= 0,

    That is, it solves
    min (1/2)x^T*Q*x + c^T*x - mu*sum(ln(x_i)) subject to Ax = b
    using a decreasing mu value to keep the candidate solution contained in x >= 0

    Automatically finds a point in the feasible region to start

    Input Arguments:
    Q         -- The matrix of coefficients in the quadratic portion of the problem
    c         -- The vector describing the linear function to minimize
    A         -- Matrix of coefficients for linear constraints
    b         -- The Right-hand side vector for the linear constraints
    tol       -- Error tolerance for stopping condition
                 - If a scalar, then used as tolerance for both phases
                 - If a tuple, tol[0] is tolerance for Phase I and
                   tol[1] is tolerance for Phase II
    kmax      -- Maximum steps allowed, used for stopping condition
    rho       -- Used for decreasing the strength of the barrier
                 - at each iteration mu becomes rho*mu
    mu0       -- Starting value of mu for strength of barrier
    mumin     -- Minimum strength of the barrier

    Returns:
    If the optimal is found within tolerance
    x        -- Coordinates of the optimal value
    k        -- The number of total iterations required
                - Includes the iterations needed to find the first feasible point

    Errors:
    Raises a DimensionMismatchError if the dimensions of the matrices are not compatible
    Raises a NonConvergenceError if the optimum cannot be found within tolerance

    Example:
    # Set up and solve a linear programming problem (Q = 0)
    A = np.array([[1, 2, -1, 1],[2, -2, 3, 3],[1, -1, 2, -1]],dtype="float")
    b = np.array([[0, 9, 6]]).T
    c = np.array([[-3, 1, 3, -1]]).T
    Q = np.zeros((4,4))
    tol = 1e-5

    x, k = InteriorPointBarrier_EqualityOnly(Q, c, A, b, tol)
    """
    # Check if the dimensions of A and b are compatible
    compat, error = _DimensionsCompatible_EqualityOnly(Q, c, A, b)
    if not compat:
        raise DimensionMismatchError(error)

    # Check if tolerance is tuple or scalar
    if not isinstance(tol, tuple):
        tol = (tol, tol)

    # For convenience in defining the needed matrices
    m,n = A.shape

    # To track the total number of iterations for both phases
    totalk = 0

    # Phase I - find a solution in the feasible region
    x = np.ones((n,1))
    lamb = np.zeros((m,1))
    s = np.ones((n,1))


    if np.count_nonzero(Q) != 0:
        print("Running Phase 1")
        Q_p1 = np.eye(n)
        c_p1 = -1*np.ones((n,1))
        x,lamb,s,k = _Barrier_Worker_EqualityOnly(Q_p1, c_p1, A, b, x, lamb, s, tol[0], kmax, rho, mu0, mumin)
        totalk += k

    # Phase II
    Q_p2 = Q.copy()
    c_p2 = c.copy()

    x,lamb,s,k = _Barrier_Worker_EqualityOnly(Q_p2, c_p2, A, b, x, lamb, s, tol[1], kmax, rho, mu0, mumin)
    totalk += k

    return x,totalk

def _DimensionsCompatible_EqualityOnly(Q, c, A, b):
    """Check to make sure the dimensions of a quadratic programming problem are compatible"""
    if A.shape[0] != b.shape[0]: return False, f"Rows A ({A.shape[0]}) != Rows b ({b.shape[0]})"
    if Q.shape[0] != Q.shape[1]: return False, f"Q must be square - is currently {Q.shape}"
    if Q.shape[0] != A.shape[1]: return False, f"Cols A ({A.shape[1]}) != Rows Q ({Q.shape[0]})"
    if Q.shape[0] != c.shape[0]: return False, f"Rows c ({c.shape[0]}) != Rows Q ({Q.shape[0]})"
    if c.shape[1] != 1: return False, "c must be a column vector"
    if b.shape[1] != 1: return False, "b msut be a column vector"
    return True, "No error"

def _Barrier_Worker_EqualityOnly(Q, c, A, b, x, lamb, s, tol, kmax, rho, mu0, mumin):
    """
    Worker method for the InteriorPointBarrier_EqualityOnly function

    Runs Newton's method with a step size search to locate the optimal point which minimizes
    (1/2)x^T Q x + c^T x - mu * sum(ln(xi)) subject to Ax=b

    Same input params as the Driver function above
    """

    def Jacobian():
        """Jacobian of the function F(x,lamb,s) below"""
        return np.block([[A,              np.zeros((m,m+n))],
                         [-1*Q,           A.T,             np.eye(n)],
                         [np.diagflat(s), np.zeros((n,m)), np.diagflat(x)]])

    def F(Fx, Fs, Flamb):
        """Used in the Barrier method to calculate the overall system value"""
        m,n = A.shape
        F_row1 = np.matmul(A,Fx) - b
        F_row2 = np.matmul(-Q,Fx) + np.matmul(A.T,Flamb) + Fs - c
        F_row3 = np.array([[Fx[i,0]*Fs[i,0]] for i in range(n)]) - mu*np.ones((n,1))
        return np.vstack((F_row1, F_row2, F_row3))

    # For convenience
    m,n = A.shape
    mu = mu0

    k = 1

    # Cache frequently needed values
    r = -F(x,s,lamb)
    normr = LA.norm(r)**2

    while np.sqrt(normr) > tol and k < kmax and mu > mumin:
        # Calculate the Jacobian
        J = Jacobian()

        # Solve Jd = r using least squares method
        d = LA.lstsq(J,r,rcond=None)[0]

        # Split the d apart into the different component pieces
        dx = d[:n]
        dlamb = d[n:n+m]
        ds = d[n+m:]

        z = 0.9*LA.norm(np.matmul(r.T,J))*LA.norm(d)

        # Select the largest alpha_0 with 0 <= alpha_0 <=1 so that x + alpha_0*dx > 0 and s + alpha*ds > 0
        if all(v > 0 for v in dx) and all(v > 0 for v in ds):
            # Step is away from the boundary so we can take a full step
            alpha_bar = 1
        else:
            # Step is moving toward the boundary, we need to make sure we don't cross over it

            alpha_bar = None
            # search for min(-xk[i]/dxk[i]) among i s.t. dx[i] < 0
            for i in range(dx.shape[0]):
                if dx[i] < 0:
                    t = -x[i]/dx[i]
                    if alpha_bar == None or t < alpha_bar:
                        alpha_bar = t
            # search for min(-sk[i]/dsk[i]) among i s.t. ds[i] < 0
            for i in range(ds.shape[0]):
                if ds[i] < 0:
                    t = -s[i]/ds[i]
                    if alpha_bar == None or t < alpha_bar:
                        alpha_bar = t

        # alpha_bar is the distance to the boundary (or 1 if we are moving away from the boundary, 
        # want a little bit on the inside of the boundary
        alpha_0 = 0.99995*min(alpha_bar,1)

        # Now we need to find the "best" Newton step size (minimizes F along the direction of d)
        # Set L and R
        L = LA.norm(F(x + alpha_0*dx, s + alpha_0*ds, lamb + alpha_0*dlamb))**2
        R = normr - alpha_0*z

        # Find min L
        Lmin = L
        index = 0

        j = 0
        while L > R and j < 1000:
            j += 1

            # Calculate potential stepsize
            stepsize = 2**(-j)*alpha_0

            # Calculate L, R
            L = LA.norm(F(x + stepsize*dx, s + stepsize*ds, lamb + stepsize*dlamb))**2
            R = normr - stepsize*z
            if L < Lmin:
                Lmin = L
                index = j
        # Check if the above loop was infinite
        if j >= 1000:
            raise NonConvergenceError("Cannot determine the correct Newton step size")

        # Update the Solution
        stepsize = 2**(-index)*alpha_0
        x += stepsize*dx
        lamb += stepsize*dlamb
        s += stepsize*ds

        # Update counter and decrease mu
        k += 1
        mu = rho*mu

        # Update cache
        r = -F(x, s, lamb)
        normr = LA.norm(r)**2

    if k > kmax:
        raise NonConvergenceError("kmax exceeded, consider raising it")
    if mu < mumin:
        raise NonConvergenceError("mu became smaller than mumin before reaching convergence. Consider lowering mumin")

    return x,lamb,s,k

def Barrier_EqualityInequality(Q, c, A, b, C, d, tol, kmax=1000, rho=.9, mu0=1e2, mumin=1e-9):
    """
    Run Interior Point Barrier Method to solve the quadratic programming problem:
    min (1/2)x^T*Q*x + c^T*x subject to Ax >= b , x >= 0, and Cx = d

    That is, it solves
    min (1/2)x^T*Q*x + c^T*x - mu*sum(ln(x_i)) - mu(sum(ln(ri*x-b_i))) subject to Cx = d
    where r_i is the ith row of A using a decreasing mu value to keep the candidate solution
    contained in x >= 0

    Automatically finds a point in the feasible region to start

    Input Arguments:
    Q         -- The matrix of coefficients in the quadratic portion of the problem
    c         -- The vector describing the linear function to minimize
    A         -- Matrix of coefficients for linear constraints
    b         -- The Right-hand side vector for the linear constraints
    C         -- Matrix of coefficients for the equality constraints
    d         -- RHS vector of linear equality constraints
    tol       -- Error tolerance for stopping condition
                 - If a scalar, then used as tolerance for both phases
                 - If a tuple, tol[0] is tolerance for Phase I and
                   tol[1] is tolerance for Phase II
    kmax      -- Maximum steps allowed, used for stopping condition
    rho       -- Used for decreasing the strength of the barrier
                 - at each iteration mu becomes rho*mu
    mu0       -- Starting value of mu for the step size
    mumin     -- Minimum strength of the barrier

    Returns:
    If the optimal is found within tolerance
    x        -- Coordinates of the optimal value
    k        -- The number of total iterations required
                - Includes the iterations needed to find the first feasible point

    Errors:
    Raises a DimensionMismatchError if the dimensions of the matrices are not compatible
    Raises a NonConvergenceError if the optimum cannot be found within tolerance

    Example:
    # Solve the problem:
    # min 2x_1 + 3x_2 + 6x_3
    # subj. to x_1 -  x_2 +  x_3 = 2
    #         2x_1 + 2x_2 - 3x_3 = 0
    #          x_1 + 3x_2 + 2x_3 <= 3
    #               -2x_2 +  x_3 <= 1
    #              x_1, x_2, x_3 >= 0
    A = np.array([[-1, -3, -2], [0, 2, 1]])
    b = np.array([[-3, -1]]).T
    c = np.array([[2, 3, 6]]).T
    Q = np.zeros((3,3))
    C = np.array([[1, -1, 1],[2, 2, -3]])
    d = np.array([[2, 0]]).T
    tol = 1e-8
    kmax = 10000
    rho = .9
    mu0 = 1e4
    mumin = 1e-9

    x,k = InteriorPointBarrier_EqualityInequality(Q,c,A,b,C,d,tol,kmax,rho,mu0,mumin)
    print("Found Optimal : \n" + str(x))
    print("Num Iterations: " + str(k))
    true_answer = np.array([[1.2,0,0.8]]).T
    print("Norm of Error is: " + str(LA.norm(x - true_answer)))
    """

    # Check if the dimensions of A and b are compatible
    compat, error = _DimensionsCompatible_EqualityInequality(Q, c, A, b, C, d)
    if not compat:
        raise DimensionMismatchError(error)

    # Check if tolerance is tuple or scalar
    if not isinstance(tol, tuple):
        tol = (tol, tol)

    # For convenience in defining the needed matrices
    m,n = A.shape
    p,_ = C.shape

    # To track the total number of iterations for both phases
    totalk = 0

    x = np.ones((n,1))
    lamb = np.zeros((p,1))
    s = np.ones((n,1))
    t = np.ones((m,1))
    theta = np.ones((m,1))

    # Phase I - find a solution in the feasible region - if required
    if np.count_nonzero(Q) != 0:
        print("Running Phase 1")
        Q_p1 = np.eye(n)
        c_p1 = -1*np.ones((n,1))
        x,lamb,s,t,theta,k = _Barrier_Worker_EqualityInequality(Q_p1, c_p1, A, b, C, d, x, lamb, s, t, theta, tol[0], kmax, rho, mu0, mumin)
        totalk += k

    # Phase II
    Q_p2 = Q.copy()
    c_p2 = c.copy()

    x,lamb,s,t,theta,k = _Barrier_Worker_EqualityInequality(Q_p2, c_p2, A, b, C, d, x, lamb, s, t, theta, tol[1], kmax, rho, mu0, mumin)
    totalk += k

    return x,k

def _DimensionsCompatible_EqualityInequality(Q, c, A, b, C, d):
    """Check to make sure the dimensions of a quadratic programming problem are compatible"""
    if A.shape[0] != b.shape[0]: return False, f"Rows A ({A.shape[0]}) != Rows b ({b.shape[0]})"
    if Q.shape[0] != Q.shape[1]: return False, f"Q must be square - is currently {Q.shape}"
    if Q.shape[0] != A.shape[1]: return False, f"Cols A ({A.shape[1]}) != Rows Q ({Q.shape[0]})"
    if Q.shape[0] != c.shape[0]: return False, f"Rows c ({c.shape[0]}) != Rows Q ({Q.shape[0]})"
    if C.shape[0] != d.shape[0]: return False, f"Rows C ({C.shape[0]}) != Rows d ({d.shape[0]})"
    if b.shape[1] != 1: return False, "b must be a column vector"
    if d.shape[1] != 1: return False, "d must be a column vector"
    if c.shape[1] != 1: return False, "c must be a column vector"
    return True, "No error"

def _Barrier_Worker_EqualityInequality(Q, c, A, b, C, d, x, lamb, s, t, theta, tol, kmax, rho, mu0, mumin):
    """
    Worker method for the InteriorPointBarrier_EqualityInequality function

    Runs Newton's method with a step size search to locate the optimal point which minimizes
    (1/2)x^T Q x + c^T x - mu * sum(ln(xi)) - mu * sum(ln(r_i*x - b_i)) subject to Cx=D

    Same input params as the Driver function above
    """

    def Jacobian():
        """Jacobian of the function F(x,lamb,s) below"""
        return np.block([[C,  np.zeros((p,p+n+2*m))],
                         [-1*Q, C.T, np.eye(n), A.T, np.zeros((n,m))],
                         [A, np.zeros((m,p+n+m)),-1*np.eye(m)],
                         [np.diagflat(s), np.zeros((n,p)), np.diagflat(x), np.zeros((n,2*m))],
                         [np.zeros((m,2*n+p)), np.diagflat(theta), np.diagflat(t)]])

    def F(Fx, Flamb, Fs, Ft, Ftheta):
        """Used in the Barrier method to calculate the overall system value"""
        m,n = A.shape
        p,_ = C.shape
        F_row1 = np.matmul(C,Fx) - d
        F_row2 = np.matmul(-Q,Fx) + np.matmul(C.T,Flamb) + Fs + np.matmul(A.T,Ft) - c
        F_row3 = np.matmul(A,Fx) - b - theta
        F_row4 = np.array([[Fx[i,0]*Fs[i,0]] for i in range(n)]) - mu*np.ones((n,1))
        F_row5 = np.array([[Ft[i,0]*Ftheta[i,0]] for i in range(m)]) - mu*np.ones((m,1))
        return np.vstack((F_row1, F_row2, F_row3, F_row4, F_row5))

    # For convenience
    m,n = A.shape
    p,_ = C.shape
    mu = mu0

    k = 1

    # Cache frequently needed values
    r = -F(x,lamb,s,t,theta)
    normr = LA.norm(r)**2

    while np.sqrt(normr) > tol and k < kmax and mu > mumin:
        # Calculate the Jacobian
        J = Jacobian()

        # Solve Jd = r using least squares method
        dir = LA.lstsq(J,r,rcond=None)[0]

        # Split the d apart into the different component pieces
        dx = dir[:n]
        dlamb = dir[n:n+p]
        ds = dir[n+p:2*n+p]
        dt = dir[2*n+p:2*n+p+m]
        dtheta = dir[2*n+p+m:]

        z = 0.9*LA.norm(np.matmul(r.T,J))*LA.norm(dir)

        # Select the largest alpha_0 with 0 <= alpha_0 <=1 so that x + alpha_0*dx > 0 and s + alpha*ds > 0
        if all(v >= 0 for v in dx) and all(v >= 0 for v in ds) and all(v >= 0 for v in dt) and all(v >= 0 for v in dtheta):
            # Step is away from the boundary so we can take a full step
            alpha_bar = 1
        else:
            # Step is moving toward the boundary, we need to make sure we don't cross over it
            alpha_bar = None
            # search for min(-xk[i]/dxk[i]) among i s.t. dx[i] < 0
            for i in range(dx.shape[0]):
                if dx[i] < 0:
                    val = -x[i]/dx[i]
                    if alpha_bar == None or val < alpha_bar:
                        alpha_bar = val
            # search for min(-sk[i]/dsk[i]) among i s.t. ds[i] < 0
            for i in range(ds.shape[0]):
                if ds[i] < 0:
                    val = -s[i]/ds[i]
                    if alpha_bar == None or val < alpha_bar:
                        alpha_bar = val
            # # search for min(-tk[i]/dtk[i]) among i s.t. dt[i] < 0
            for i in range(dt.shape[0]):
                if dt[i] < 0:
                    val = -t[i]/dt[i]
                    if alpha_bar == None or val < alpha_bar:
                        alpha_bar = val
            # # search for min(-thetak[i]/dthetak[i]) among i s.t. dtheta[i] < 0
            for i in range(dtheta.shape[0]):
                if dtheta[i] < 0:
                    val = -theta[i]/dtheta[i]
                    if alpha_bar == None or val < alpha_bar:
                        alpha_bar = val

        # alpha_bar is the distance to the boundary (or 1 if we are moving away from the boundary, 
        # want a little bit on the inside of the boundary
        alpha_0 = 0.99995*min(alpha_bar,1)
        # print(alpha_0)

        # Now we need to find the "best" Newton step size (minimizes F along the direction of d)
        # Set L and R
        L = LA.norm(F(x + alpha_0*dx, lamb + alpha_0*dlamb, s + alpha_0*ds, t + alpha_0*dt, theta + alpha_0*dtheta))**2
        R = normr - alpha_0*z

        # Find min L
        Lmin = L
        index = 0

        j = 0
        while L > R and j < 1000:
            j += 1

            # Calculate potential stepsize
            stepsize = 2**(-j)*alpha_0

            # Calculate L, R
            L = LA.norm(F(x + stepsize*dx, lamb + stepsize*dlamb, s + stepsize*ds, t + stepsize*dt, theta + stepsize*dtheta))**2
            R = normr - stepsize*z
            if L < Lmin:
                Lmin = L
                index = j
        # Check if the above loop was infinite
        if j >= 1000:
            raise NonConvergenceError("Cannot determine the correct Newton step size")

        # Update the Solution
        stepsize = 2**(-index)*alpha_0
        x += stepsize*dx
        lamb += stepsize*dlamb
        s += stepsize*ds
        t += stepsize*dt
        theta += stepsize*dtheta

        # Update counter and decrease mu
        k += 1
        mu = rho*mu

        # Update cache
        r = -F(x, lamb, s, t, theta)
        normr = LA.norm(r)**2

    if k > kmax:
        raise NonConvergenceError("kmax exceeded, consider raising it")
    if mu < mumin:
        raise NonConvergenceError(f"mu became smaller than mumin before reaching convergence. Consider lowering mumin.\n x = {x}")

    return x,lamb,s,t,theta,k

def Predictor_Corrector(Q, c, A, b, tol, kmax=1000, rho=.95, mu0=1e1, mumin=1e-9):
    """
    Run Interior Point Meherotra Predictor Corrector Method to solve the quadratic programming problem:
    min (1/2)x^T*Q*x + c^T*x subject to Ax = b and x >= 0,

    That is, it solves
    min (1/2)x^T*Q*x + c^T*x - mu*sum(ln(x_i)) subject to Ax = b
    using a two phase approach:
    1) Find feasible initial condition with Newton Inexact Line Search
    2) Predictor Corrector Method for determining actual optimal

    Input Arguments:
    Q         -- The matrix of coefficients in the quadratic portion of the problem
    c         -- The vector describing the linear function to minimize
    A         -- Matrix of coefficients for linear constraints
    b         -- The Right-hand side vector for the linear constraints
    tol       -- Error tolerance for stopping conditions
                 - If a scalar, then used as tolerance for both phases
                 - If a tuple, tol[0] is tolerance for Phase I and
                   tol[1] is tolerance for Phase II
    kmax      -- Maximum steps allowed, used for stopping condition
                 - If a scalar, then used as kmax for both phases
                 - If a tuple, kmax[0] is kmax for Phase I and
                   kmax[1] is kmax for Phase II
    rho       -- Used for decreasing the strength of the barrier
                 - at each iteration mu becomes rho*mu
                 - Only used in Phase 1
    mu0       -- Starting value of mu for the step size
                 - Only used in Phase 1
    mumin     -- Minimum strength of the barrier

    Returns:
    If the optimal is found within tolerance
    x        -- Coordinates of the optimal value
    k        -- Three element array:
                [Total iters, # iters Phase I, # iters Phase II]

    Errors:
    Raises a DimensionMismatchError if the dimensions of the matrices are not compatible
    Raises a NonConvergenceError if the optimum cannot be found within tolerance

    Example:
    """
    # Check if the dimensions of A and b are compatible
    compat, error = _DimensionsCompatible_EqualityOnly(Q, c, A, b)
    if not compat:
        raise DimensionMismatchError(error)

    # Check if tolerance is tuple or scalar
    if not isinstance(tol, tuple):
        tol = (tol, tol)
    if not isinstance(kmax, tuple):
        kmax = (kmax, kmax)

    # For convenience in defining the needed matrices
    m,n = A.shape

    # To track the total number of iterations for both phases
    iters = [0,0,0]

    x = np.ones((n,1))
    lamb = np.zeros((m,1))
    s = np.ones((n,1))

    # Phase I - find a solution in the feasible region
    # - Can use the Newton's Inexact Line Search for this
    Q_p1 = np.eye(n)
    c_p1 = -1*np.ones((n,1))
    x,lamb,s,iters[1] = _Barrier_Worker_EqualityOnly(Q, c, A, b, x, lamb, s, tol[0], kmax[0], rho, mu0, mumin)

    # The above Phase one doesn't take Q or c into account, in particular with regards to lamb and s
    # Here so improve the point from Phase I for s and lamb
    Qxc = np.matmul(Q,x) + c
    s = np.maximum(Qxc,0) + 1e-4*np.ones(c.shape)
    lamb = myLA.lowRank_MinNormLS(A.T,-np.minimum(Qxc,0) - 1e-4*np.ones(c.shape))

    # Phase II
    Q_p2 = Q.copy()
    c_p2 = c.copy()

    x,lamb,s,iters[2] = _PD_Worker(Q_p2, c_p2, A, b, x, lamb, s, tol[1], kmax[1], mumin)

    iters[0] = iters[1] + iters[2]
    return x,iters

def _PD_Worker(Q, c, A, b, x, lamb, s, tol, kmax, mumin):
    """
    Worker method for the Predictor_Corrector function

    Runs Predictor Corrector method for stepping to find the minimum of
    (1/2)x^T Q x + c^T x - mu * sum(ln(xi)) subject to Ax=b

    Same input params as the Driver function above
    """

    def Jacobian():
        """Jacobian of the function F(x,lamb,s) below"""
        return np.block([[A,              np.zeros((m,m+n))],
                         [-1*Q,           A.T,             np.eye(n)],
                         [np.diagflat(s), np.zeros((n,m)), np.diagflat(x)]])

    def F(Fx, Fs, Flamb):
        """Used in the Barrier method to calculate the overall system value"""
        m,n = A.shape
        F_row1 = np.matmul(A,Fx) - b
        F_row2 = np.matmul(-Q,Fx) + np.matmul(A.T,Flamb) + Fs - c
        # F_row3 = np.array([[Fx[i,0]*Fs[i,0]] for i in range(n)]) - sigma*mu*np.ones((n,1))
        F_row3 = np.array([[Fx[i,0]*Fs[i,0]] for i in range(n)])
        return np.vstack((F_row1, F_row2, F_row3))

    # For convenience
    m,n = A.shape

    k = 1

    # Cache frequently needed values
    r = -F(x,s,lamb)
    mu = np.matmul(s.T, x)/n
    rx = r[:n]
    rlamb = r[n:n+m]

    while _PredCorr_CalcTolerance(rx, rlamb, mu, b, c, x, Q) > tol and k < kmax and mu >= mumin:
        # Calculate the Jacobian
        J = Jacobian()

        # Solve Jd = r using least squares method
        d_pred = LA.lstsq(J,r,rcond=None)[0]

        # Split the d_pred apart into the different component pieces
        dx_pred = d_pred[:n]
        dlamb_pred = d_pred[n:n+m]
        ds_pred = d_pred[n+m:]

        # Find the alpha_preds
        alpha_pred_primal = _PredCorr_AlphaBinarySearch(x, dx_pred)
        alpha_pred_dual = _PredCorr_AlphaBinarySearch(s, ds_pred)

        # Now find mupred and sigma
        mu_pred = np.matmul((x + alpha_pred_primal*dx_pred).T, s + alpha_pred_dual*ds_pred)/n
        sigma = (mu_pred/mu)**3

        # Solve for dcorr
        # Need RHS first
        rhs_row1 = np.zeros((m+n,1))
        rhs_row2 = sigma*mu*np.ones((n,1)) - np.array([[dx_pred[i,0]*ds_pred[i,0]] for i in range(n)])
        rhs = np.vstack((rhs_row1,rhs_row2))
        # Now solve for d_corr
        d_corr = LA.lstsq(J,rhs,rcond=None)[0]

        # Create the full d vector
        d = d_pred + d_corr

        # Split the d vector apart into the different component pieces
        dx = d[:n]
        dlamb = d[n:n+m]
        ds = d[n+m:]

        # Find the alphas
        # Find the alpha_preds
        alpha_primal = _PredCorr_AlphaBinarySearch(x, dx)
        alpha_dual = _PredCorr_AlphaBinarySearch(s, ds)

        # Update the Solution
        x += alpha_primal*dx
        lamb += alpha_dual*dlamb
        s += alpha_dual*ds

        # Update counter and mu
        k += 1
        mu = np.matmul(x.T, s)/n

        # Update cache
        r = -F(x, s, lamb)
        rx = r[:n]
        rlamb = r[n:n+m]

    if k > kmax and _PredCorr_CalcTolerance(rx, rlamb, mu, b, c, x, Q) > tol:
        raise NonConvergenceError("kmax exceeded, consider raising it")
    if mu < mumin and _PredCorr_CalcTolerance(rx, rlamb, mu, b, c, x, Q) > tol:
        raise NonConvergenceError("mu became smaller than mumin before reaching convergence. Consider lowering mumin")

    return x,lamb,s,k

def _PredCorr_CalcTolerance(rx, rlamb, mu, b, c, x, Q):
    check1 = LA.norm(rx)/(1 + LA.norm(b))
    check2 = LA.norm(rlamb)/(1 + LA.norm(c))
    check3 = mu/(1 + 0.5*np.matmul(x.T, np.matmul(Q,x)) + np.matmul(c.T, x))
    return max(check1, check2, check3[0,0])

def _PredCorr_AlphaBinarySearch(x, dx,tol=1e-2):
    """Perform a line search to find maximum alpha s.t. x + alpha*dx >= 0"""
    # If we can take a full step without crossing the boundary, then we just return alpha = 1
    if all(x[i,0] + dx[i,0] >= 0 for i in range(x.shape[0])):
        return 1.0

    # If we can't take a full step, then we use a binary search
    L = 1e-5
    R = 1.0
    while R-L > tol:
        M = 0.5*(R + L)
        if all(x[i,0] + M*dx[i,0] >= 0 for i in range(x.shape[0])):
            L = M
        else:
            R = M

    # L is confirmed to be s.t. x + L*dx >= 0
    return L

if __name__ == "__main__":
    # To solve the following problem:
    # min -10x_1 - 9x_2
    # subj. to 7x_1 + 10x_2 <= 6300
    #          3x_1 +  5x_2 <= 3600
    #          3x_1 +  2x_2 <= 2124
    #          2x_1 +  5x_2 <= 2700

    # A = np.array([[-7, -10, -1, 0, 0, 0],[-3, -5, 0, -1, 0, 0],[-3, -2, 0, 0, -1, 0],[-2, -5, 0, 0, 0, -1]])
    # b = np.array([[-6300, -3600, -2124, -2700]]).T
    # c = np.array([[-10, -9, 0, 0, 0, 0]]).T
    # Q = np.zeros((6,6))
    # tol = 1e-5
    # kmax = 10000
    # rho = .9
    # mu0 = 1e4
    # mumin = 1e-8

    # x,k = InteriorPointBarrier_EqualityOnly(Q, c, A, b, tol, kmax, rho, mu0, mumin)
    # print("Found optimal : \n" + str(x[0:2]))
    # print("Num Iterations: " + str(k))

    # Solve the problem:
    # min 2x_1 + 3x_2 + 6x_3
    # subj. to x_1 -  x_2 +  x_3  = 2
    #         2x_1 + 2x_2 - 3x_3  = 0
    #          x_1 + 3x_2 + 2x_3 <= 3
    #               -2x_2 +  x_3 <= 1
    #              x_1, x_2, x_3 >= 0
    # A = np.array([[-1, -3, -2], [0, 2, -1]])
    # b = np.array([[-3, -1]]).T
    # c = np.array([[2, 3, 6]]).T
    # Q = np.zeros((3,3))
    # C = np.array([[1, -1, 1],[2, 2, -3]])
    # d = np.array([[2, 0]]).T
    # tol = 1e-8
    # kmax = 10000
    # rho = .9
    # mu0 = 1e4
    # mumin = 1e-9

    # x,k = InteriorPointBarrier_EqualityInequality(Q,c,A,b,C,d,tol,kmax,rho,mu0,mumin)
    # print("Found Optimal : \n" + str(x))
    # print("Num Iterations: " + str(k))
    # true_answer = np.array([[1.208,0.042,0.833]]).T
    # print("Norm of Error is: " + str(LA.norm(x - true_answer)))
    # print("Cost of found solution:" + str(np.matmul(c.T,x)))
    # print("Cost of 'true' solution: " + str(np.matmul(c.T,true_answer)))

    # Solve the following problem using the predictor-corrector method:
    # min -3x_1 + x_2 + 3x_3 - x_4
    # subj. to  x_1 + 2x_2 -  x_3 +  x_4 = 0
    #          2x_1 - 2x_2 + 3x_3 + 3x_4 = 9
    #           x_1 -  x_2 + 2x_3 -  x_4 = 6
    #           x_1, x_2, x_3, x_4 >= 0
    # A = np.array([[1,  2, -1,  1],
    #               [2, -2,  3,  3],
    #               [1, -1,  2, -1]])
    # b = np.array([[0, 9, 6]]).T
    # Q = np.zeros((4,4))
    # c = np.array([[-3, 1, 3, -1]]).T
    # tol = (1e0,1e-9)

    # x,k = Predictor_Corrector(Q, c, A, b, tol, mumin=1e-12)
    # print("Found Optimal : \n" + str(x))
    # print("Num Iterations: " + str(k))
    # true_answer = np.array([[1, 1, 3, 0]]).T
    # print("Norm of Error is: " + str(LA.norm(x - true_answer)))
    # print("Cost of found solution:" + str(np.matmul(c.T,x)))
    # print("Cost of 'true' solution: " + str(np.matmul(c.T,true_answer)))

    # Solve the following problem using the predictor-corrector method:
    # min x_1 + 6x_2 - 7x_3 + x_4 + 5x_5
    # subj. to 5x_1 - 4x_2 + 13x_3 - 2x_4 + x_5 = 20
    #           x_1 -  x_2 +  5x_3 -  x_4 + x_5 = 8
    #           x_1, x_2, x_3, x_4, x_5 >= 0
    A = np.array([[5, -4, 13, -2, 1],
                  [1, -1,  5, -1, 1]])
    b = np.array([[20, 8]]).T
    Q = np.zeros((5,5))
    c = np.array([[1, 6, -7, 1, 5]]).T
    tol = (1e-3,1e-14)
    kmax= (100,1000)

    x,k = Predictor_Corrector(Q, c, A, b, tol, mumin=1e-14, kmax=kmax)
    print("Found Optimal : \n" + str(x))
    print("Num Iterations: " + str(k))
    true_answer = np.array([[0, 0.5714, 1.7143, 0, 0]]).T
    print("Norm of Error is: " + str(LA.norm(x - true_answer)))
    print("Cost of found solution: " + str(np.matmul(c.T,x)[0,0]))
    print("Cost of 'true' solution: " + str(np.matmul(c.T,true_answer)[0,0]))
