#!/usr/bin/python
"""Module for Newton's Method code - Newton's Method for Rooting and Gradient Descent'"""
import numpy.linalg as LA
import numpy as np

def Newton(f, x0, tao, c, tol, kmax):
    """Run Newton's Rootfinding method to find the zero of f"""
    k = 1
    s = LA.norm(f(x0))**2
    xk = x0
    while s > tol and k < kmax:
        H = _CentralDifferences(f,xk,tao)

        # Uses LAPACK _gesv
        dk = LA.solve(H,-f(xk))

        z = c*LA.norm(f(xk).transpose()*H)*LA.norm(dk)

        j = 0
        L = LA.norm(f(xk + dk))**2
        R = s-z
        Lmin = L
        index = 0
        while L > R:
            j = j + 1
            L = LA.norm(f(xk + 2**(-j)))**2
            R = s = 2**(-j)*z
            if L < Lmin:
                Lmin = L
                index = j

        xk = xk + 2**(-index)*dk
        k = k + 1
        s = LA.norm(f(xk))**2

    return xk

def GradDescent(f, x0, tol, kmax):
    """Run Gradient Descent to find the zero of f"""
    k = 1
    xold = np.zeros(x0.shape)
    xnew = x0
    dold = np.zeros(x0.shape)
    while LA.norm(f(xnew)) > tol and k < kmax:
        dnew = -f(xnew)
        gamma = abs(np.matmul((xnew - xold).transpose(), -dnew + dold))/LA.norm(-dnew+dold)**2
        xold = xnew
        dold = dnew
        xnew = xold + gamma*dnew
        k = k+1
    if k >= kmax:
        print("kmax exceeded - don't trust the answer")
    return xnew

def _CentralDifferences(f,x,tao):
    H = np.eye(x.shape[0])
    for i in range(x.shape[0]):
        H[:,i] = 1/(2*tao)*(f(x + tao*H[:,i]) - f(x - tao*H[:,i]))
    return H

