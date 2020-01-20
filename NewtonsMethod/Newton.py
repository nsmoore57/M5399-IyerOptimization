#!/usr/bin/python
"""Module for Newton's Method code - Newton's Method for Rooting and Gradient Descent'"""
import numpy.linalg as LA
import numpy as np

def Newton(f, x0, tao, c, tol, kmax):
    """Run Newton's Rootfinding method to find the zero of f"""
    return None

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
