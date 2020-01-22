#!/usr/bin/env python
"""This script tests the code in Newton.py"""

import numpy as np
import numpy.linalg as LA
import Newton

def Rosenbrock(a, x):
    """
    Function for testing optimization, vector-valued to repeat same value
    in both dimensions
    """
    res = (a - x[0])**2 + 100*(x[1]-x[0]**2)**2
    return np.array([res, res])
def Grad_Rosenbrock(a, x):
    """
    Gradient of the Rosenbrock function
    Used for obtaining the mininum of the Rosenbrock
    """
    dfdx1 = -2*(a-x[0]) - 400*(x[1]-x[0]**2)*x[0]
    dfdx2 = 200*(x[1]-x[0]**2)
    return np.vstack((dfdx1, dfdx2))

def arctan2d(x):
    """
    2D arctan function for testing GradDescent
    """
    res1 = np.arctan(x[0] - np.pi/4)
    res2 = np.arctan(x[1] - np.pi/4)
    return np.vstack((res1, res2))

Rosenbrock1 = (lambda x: Grad_Rosenbrock(1, x))
Rosenbrock2 = (lambda x: Grad_Rosenbrock(2, x))
Rosenbrock3 = (lambda x: Grad_Rosenbrock(3, x))
arctan1d = (lambda x: np.arctan(x - np.pi/4))

def test_NewtonsMethod():
    """Test Newton.Newton function"""
    x0 = np.array([2], dtype="float")
    tao = 10**(-5)
    c = 0.5
    tol = 10**(-6)
    kmax = 1000
    assert arctan1d(Newton.Newton(arctan1d, x0, tao, c, tol, kmax)) < tol

    x0 = np.array([[3], [3]], dtype="float")
    tao = 10**(-5)
    c = 0.5
    tol = 10**(-1)
    kmax = 100000
    assert LA.norm(Rosenbrock1(Newton.Newton(Rosenbrock1, x0, tao, c, tol, kmax))) < tol
    assert LA.norm(Rosenbrock2(Newton.Newton(Rosenbrock2, x0, tao, c, tol, kmax))) < tol
    assert LA.norm(Rosenbrock3(Newton.Newton(Rosenbrock3, x0, tao, c, tol, kmax))) < tol

def test_GradDescent():
    """Test Newton.GradDescent function"""
    tol = 10**(-6)
    kmax = 1000
    assert LA.norm(arctan1d(Newton.GradDescent(arctan1d, np.array([10]), tol, kmax))) < tol
    assert LA.norm(arctan2d(Newton.GradDescent(arctan2d, np.array([[10], [10]]), tol, kmax))) < tol
