#!/usr/bin/python
"""Module for Interior Point Methods"""
import numpy as np
import numpy.linalg as LA

def InteriorPointBarrier(A, b, tol, kmax, rho, mu0, mumin):
    """
    Runs the interior point method for constrained optimization where
    the constraints are Ax = b
    
    TODO: Better docstring
    TODO: Split apart this code into a "Worker" who runs the method given all the
            necessary matrices and a "Driver" who can find the x0, then solve the
            cTx problem, then the general problem
           
    """
    if A.shape[0] != b.shape[0]:
        print("sizes of A and b don't match")
        return None
    
    def F():
        F_row1 = np.matmul(A,x) - b
        F_row2 = np.matmul(-Q,x) = np.matmul(A.transpose(),lamb) + s - c
        F_row3 = np.array([[x[i]*s[i] for i=1:n]].transpose()) - mu*ones(
        
    
    m,n = A.shape
    Q = np.eye(n)
    c = np.ones((n,1))
    x = np.ones((n,1))
    mu = mu0.copy()

    lamb = np.zeros((m,1))
    s = np.ones((n,1))
    
    k = 1
    r = -F(x, lamb,s)
    t = LA.norm(r)**2
    
    while LA.norm(r) > tol and k < kmax, and mu > mumin:
        # Calculate the Jacobian
        J_row1 = np.hstack((A, np.zeros((m,m)), np.zeros((m,n))))
        J_row2 = np.hstack((-Q, A.transpose, np.eye(n)))
        J_row3 = np.hstack((np.diagflat(s),np.zeros((n,m)),np.diagflat(x)))
        J = vstack((J_row1, J_row2, J_row3))
        
        # Solve Jd = r
        LA.solve(J,r)
        
        # Split the d apart into the different component pieces
        dx = d(:n)
        dlamb = d(n:n+m+1)
        ds = d(n+m+1:)
        z = 0.9*LA.norm(np.matmul(r.transpose(),J))*LA.norm(d)
        
        # Select the largest alpha_0 with 0 <= alpha_0 <=1 so that x + alpha_0*dx > 0 and s + alpha*ds > 0
        # This will take some work to implement - probably a pythonic way to do this
        if all(v > 0 for v in dx) and all(v > 0 for v in ds):
            alpha_bar = 1
        else
            alpha_bar = None
            # search for min(-xk[i]/dxk[i]) among i s.t. dx[i] < 0
            for i in 1:dx.shape[0]:
                if dx[i] < 0:
                    t = -x[i]/dx[i]
                    if alpha_bar == None or t < alpha_bar:
                    alpha_bar = t
            # search for min(-sk[i]/dsk[i]) among i s.t. ds[i] < 0
            for i in 1:ds.shape[0]:
                if s[i] < 0:
                    t = -s[i]/ds[i]
                    if alpha_bar == None or t < alpha_bar:
                    alpha_bar = t
        
        alpha_0 = 0.99995*min(alpha_bar,1)
        
        # Set L and R
        
        R = t - alpha_0*z 
        
        # Find min L
        Lmin = L
        index = 0
        
        j = 0
        while L > R:
            j = j + 1
            # Calculate L, R
            if L < Lmin:
                Lmin = L
                index = j
        
        # Update the Solution
            

    return None

if __name__ == "__main__":
    print("Hello World")