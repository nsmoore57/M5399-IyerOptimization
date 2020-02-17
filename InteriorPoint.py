#!/usr/bin/python
"""Module for Interior Point Methods"""
import numpy as np
import numpy.linalg as LA

def InteriorPointBarrier(A, b, tol, kmax, mumin):
    if A.shape[0] != b.shape[0]:
        print("sizes of A and b don't match")
        return None
    
    m,n = A.shape    
    Q = np.eye(n)
    c = np.ones((n,1))
    x = np.ones((n,1))
    lamb = 0
    s = np.ones((n,1))
    
    k = 1
    r = -F(x, lamb,s)
    t = LA.norm(r)**2
    
    while LA.norm(r) > tol and k < kmax, and mu > mumin:
        # Calculate the Jacobian
        
        # Solve Jd = r
        
        # Split the d apart into the different component pieces
        dx = d(:n)
        dlamb = d(n:n+m+1)
        ds = d(n+m+1:)
        z = 0.9*LA.norm(np.matmul(r.transpose(),J))*LA.norm(d)
        
        # Select the largest alpha_0 with 0 <= alpha_0 <=1 so that x + alpha_0*dx > 0 and s + alpha*ds > 0
        # This will take some work to implement - probably a pythonic way to do this
        # Placeholder for now
        alpha_bar = 1.
        
        alpha_0 = 0.99995*min(alpha_bar,1)
        
        # Set L and R
        
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