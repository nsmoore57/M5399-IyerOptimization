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
def Rosenbrock_Jac(a, x):
    """
    Function for testing optimization, Jacobian of Rosenbrock function
    """
    df1dx1 = -2*(a-x[0]) - 400*(x[1]-x[0]**2)*x[0]
    df1dx2 = 200*(x[1]-x[0])
    return np.array([[df1dx1, df1dx2], [df1dx1, df1dx2]])

def arctan2d(x):
    """
    2D arctan function for testing GradDescent
    """
    res1 = np.arctan(x[0] - np.pi/4)
    res2 = np.arctan(x[1] - np.pi/4)
    return np.vstack((res1, res2))

Rosenbrock1 = (lambda x: Rosenbrock(1, x))
Rosenbrock2 = (lambda x: Rosenbrock(2, x))
Rosenbrock3 = (lambda x: Rosenbrock(3, x))
arctan1d = (lambda x: np.arctan(x - np.pi/4))

def test_NewtonsMethod():
    """Test Newton.Newton function"""
    x0 = np.array([[1], [1]])
    tao = 10**(-5)
    c = 10*(-6)
    tol = 10**(-6)
    kmax = 1000
    assert Newton.Newton(Rosenbrock1, x0, tao, c, tol, kmax) is None
    assert Newton.Newton(Rosenbrock2, x0, tao, c, tol, kmax) is None
    assert Newton.Newton(Rosenbrock3, x0, tao, c, tol, kmax) is None

def test_GradDescent():
    """Test Newton.GradDescent function"""
    tol = 10**(-6)
    kmax = 1000
    assert LA.norm(arctan1d(Newton.GradDescent(arctan1d, np.array([10]), tol, kmax))) < tol
    assert LA.norm(arctan2d(Newton.GradDescent(arctan2d, np.array([[10], [10]]), tol, kmax))) < tol
