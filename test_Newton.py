#!/usr/bin/env python
"""This script tests the exported functions in Newton.py"""

import numpy as np
import numpy.linalg as LA
import Newton

def Rosenbrock(a,x):
    """The Rosenbrock function"""
    return (a-x[0])**2 + 100*(x[1]-x[0]**2)**2

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

Rosenbrock1 = (lambda x: Rosenbrock(1, x))
Rosenbrock2 = (lambda x: Rosenbrock(2, x))
Rosenbrock3 = (lambda x: Rosenbrock(3, x))
GRosenbrock1 = (lambda x: Grad_Rosenbrock(1, x))
GRosenbrock2 = (lambda x: Grad_Rosenbrock(2, x))
GRosenbrock3 = (lambda x: Grad_Rosenbrock(3, x))
arctan1d = (lambda x: np.arctan(x - np.pi/4))

def test_NewtonsMethod():
    """Test Newton.Newton function"""
    x0 = np.array([2], dtype="float")
    tao = 1e-5
    c = 0.7
    tol = 1e-6
    kmax = 6
    assert arctan1d(Newton.Newton(arctan1d, x0, tol, kmax)) < tol  # Check with default arguments
    assert arctan1d(Newton.Newton(arctan1d, x0, tol, kmax, c, tao)) < tol

    x0 = np.array([[3], [3]], dtype="float")
    tao = 1e-5
    c = 0.7
    tol = 1e-6
    kmax = 1e5
    assert LA.norm(GRosenbrock1(Newton.Newton(GRosenbrock1, x0, tol, kmax))) < tol # Check with default arguments
    assert LA.norm(GRosenbrock1(Newton.Newton(GRosenbrock1, x0, tol, kmax, c, tao))) < tol
    assert LA.norm(GRosenbrock2(Newton.Newton(GRosenbrock2, x0, tol, kmax, c, tao))) < tol
    assert LA.norm(GRosenbrock3(Newton.Newton(GRosenbrock3, x0, tol, kmax, c, tao))) < tol

def test_GradDescent():
    """Test Newton.GradDescent function"""

    # For BB Method

    tol = 1e-6
    kmax = 1e3
    assert LA.norm(arctan1d(Newton.GradDescent_BB(arctan1d, np.array([10]), tol, kmax))) < tol
    assert LA.norm(arctan2d(Newton.GradDescent_BB(arctan2d, np.array([[10], [10]]), tol, kmax))) < tol
    assert LA.norm(GRosenbrock1(Newton.GradDescent_BB(GRosenbrock1, np.array([[3], [3]]), tol, kmax))) < tol
    assert LA.norm(GRosenbrock2(Newton.GradDescent_BB(GRosenbrock2, np.array([[3], [3]]), tol, kmax))) < tol
    assert LA.norm(GRosenbrock3(Newton.GradDescent_BB(GRosenbrock3, np.array([[3], [3]]), tol, kmax))) < tol

    # For ILS
    tol = 1e-4
    kmax = 50000
    a_low = 1e-9
    a_high = 0.7
    N = 200
    CD_tao = 1e-5

    assert LA.norm(Rosenbrock1(Newton.GradDescent_ILS(Rosenbrock1, "CD", np.array([[3], [3]]), tol, kmax, a_low=a_low, a_high=a_high, N=N, CD_tao=CD_tao))) < tol*(1e2)
    assert LA.norm(Rosenbrock2(Newton.GradDescent_ILS(Rosenbrock2, "CD", np.array([[3], [3]]), tol, kmax, a_low=a_low, a_high=a_high, N=N, CD_tao=CD_tao))) < tol*(1e2)
    assert LA.norm(Rosenbrock3(Newton.GradDescent_ILS(Rosenbrock3, "CD", np.array([[3], [3]]), tol, kmax, a_low=a_low, a_high=a_high, N=N, CD_tao=CD_tao))) < tol*(1e2)

    # For Armijo
    tol = 1e-4
    kmax = 50000
    a_low = 1e-9
    a_high = 0.7
    N = 200
    c_low = 0.1
    c_high = 0.8
    CD_tao = 1e-5

    assert LA.norm(Rosenbrock1(Newton.GradDescent_Armijo(Rosenbrock1, "CD", np.array([[3], [3]]), tol, kmax, a_low=a_low, a_high=a_high, N=N, c_low=c_low, c_high=c_high, CD_tao=CD_tao))) < tol*(1e2)
    assert LA.norm(Rosenbrock2(Newton.GradDescent_Armijo(Rosenbrock2, "CD", np.array([[3], [3]]), tol, kmax, a_low=a_low, a_high=a_high, N=N, c_low=c_low, c_high=c_high, CD_tao=CD_tao))) < tol*(1e2)
    assert LA.norm(Rosenbrock3(Newton.GradDescent_Armijo(Rosenbrock3, "CD", np.array([[3], [3]]), tol, kmax, a_low=a_low, a_high=a_high, N=N, c_low=c_low, c_high=c_high, CD_tao=CD_tao))) < tol*(1e2)
