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
    assert arctan1d(Newton.Newton(arctan1d, x0, tol, kmax)[0]) < tol  # Check with default arguments
    assert arctan1d(Newton.Newton(arctan1d, x0, tol, kmax, c, tao)[0]) < tol

    x0 = np.array([[3], [3]], dtype="float")
    tao = 1e-5
    c = 0.7
    tol = 1e-6
    kmax = 1e5
    assert LA.norm(GRosenbrock1(Newton.Newton(GRosenbrock1, x0, tol, kmax)[0])) < tol # Check with default arguments
    assert LA.norm(GRosenbrock1(Newton.Newton(GRosenbrock1, x0, tol, kmax, c, tao)[0])) < tol
    assert LA.norm(GRosenbrock2(Newton.Newton(GRosenbrock2, x0, tol, kmax, c, tao)[0])) < tol
    assert LA.norm(GRosenbrock3(Newton.Newton(GRosenbrock3, x0, tol, kmax, c, tao)[0])) < tol

def test_GradDescent_BB():
    """Test Newton.GradDescent_BB function"""

    tol = 1e-6
    kmax = 1e3
    CD_tao = 1e-5

    x0 = np.array([[3.],[3.]])

    x,k = Newton.GradDescent_BB(Rosenbrock1, "CD", x0, tol, kmax,CD_tao=CD_tao)
    assert LA.norm(GRosenbrock1(x)) < tol*(1.1 + np.abs(Rosenbrock1(x)))
    x,k = Newton.GradDescent_BB(Rosenbrock2, "CD", x0, tol, kmax,CD_tao=CD_tao)
    assert LA.norm(GRosenbrock2(x)) < tol*(1.1 + np.abs(Rosenbrock2(x)))
    x,k = Newton.GradDescent_BB(Rosenbrock3, "CD", x0, tol, kmax,CD_tao=CD_tao)
    assert LA.norm(GRosenbrock3(x)) < tol*(1.1 + np.abs(Rosenbrock3(x)))

def test_GradDescent_ILS():
    """Test Newton.GradDescent_ILS function"""

    tol = 1e-5
    kmax = 300000
    a_low = 1e-9
    a_high = 0.99
    N = 25
    CD_tao = 1e-5
    x0 = np.array([[3.],[3.]])

    x,k = Newton.GradDescent_ILS(Rosenbrock1, "CD", x0, tol, kmax, a_low=a_low, a_high=a_high, N=N, CD_tao=CD_tao)
    assert LA.norm(GRosenbrock1(x)) < tol*(1.1 + np.abs(Rosenbrock1(x)))
    x,k = Newton.GradDescent_ILS(Rosenbrock2, "CD", x0, tol, kmax, a_low=a_low, a_high=a_high, N=N, CD_tao=CD_tao)
    assert LA.norm(GRosenbrock2(x)) < tol*(1.1 + np.abs(Rosenbrock2(x)))
    x,k = Newton.GradDescent_ILS(Rosenbrock3, "CD", x0, tol, kmax, a_low=a_low, a_high=a_high, N=N, CD_tao=CD_tao)
    assert LA.norm(GRosenbrock3(x)) < tol*(1.1 + np.abs(Rosenbrock3(x)))

def test_GradDescent_Armijo():
    """Test Newton.GradDescent_Armijo function"""
    tol = 1e-5
    kmax = 350000
    a_low = 1e-9
    a_high = 0.99
    N = 25
    c_low = 0.1
    c_high = 0.8
    CD_tao = 1e-5

    x0 = np.array([[3.],[3.]])

    x,k = Newton.GradDescent_Armijo(Rosenbrock1, "CD", x0, tol, kmax, a_low=a_low, a_high=a_high, N=N, c_low=c_low, c_high=c_high, CD_tao=CD_tao)
    assert LA.norm(GRosenbrock1(x)) < tol*(1.1 + np.abs(Rosenbrock1(x)))

    x,k = Newton.GradDescent_Armijo(Rosenbrock2, "CD", x0, tol, kmax, a_low=a_low, a_high=a_high, N=N, c_low=c_low, c_high=c_high, CD_tao=CD_tao)
    assert LA.norm(GRosenbrock2(x)) < tol*(1.1 + np.abs(Rosenbrock2(x)))

    x,k = Newton.GradDescent_Armijo(Rosenbrock3, "CD", x0, tol, kmax, a_low=a_low, a_high=a_high, N=N, c_low=c_low, c_high=c_high, CD_tao=CD_tao)
    assert LA.norm(GRosenbrock3(x)) < tol*(1.1 + np.abs(Rosenbrock3(x)))

def test_BGFS():
    """Test the Newton.BGFS function"""
    tol = 1e-5
    kmax = 1000
    a_low = 1e-9
    a_high = 0.99
    N = 25
    CD_tao = 1e-5

    x0 = np.array([[3], [3]], dtype="float")

    x,k = Newton.BFGS(Rosenbrock1, "CD", x0, tol, kmax, a_low=a_low, a_high=a_high, N=N, CD_tao=CD_tao)
    assert LA.norm(GRosenbrock1(x)) < tol*(1.1 + np.abs(Rosenbrock1(x)))

    x,k = Newton.BFGS(Rosenbrock2, "CD", x0, tol, kmax, a_low=a_low, a_high=a_high, N=N, CD_tao=CD_tao)
    assert LA.norm(GRosenbrock2(x)) < tol*(1.1 + np.abs(Rosenbrock2(x)))

    x,k = Newton.BFGS(Rosenbrock3, "CD", x0, tol, kmax, a_low=a_low, a_high=a_high, N=N, CD_tao=CD_tao)
    assert LA.norm(GRosenbrock3(x)) < tol*(1.1 + np.abs(Rosenbrock3(x)))
