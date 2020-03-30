#!/usr/bin/env python
"""This script tests the exported functions in InterierPoint.py"""

import numpy as np
import numpy.linalg as LA
import InteriorPoint as IP

def test_Barrier_EqualityOnly():
    """Tests the IP.InteriorPointBarrier_EqualityOnly Method"""

    A = np.array([[1, 2, -1, 1], [2, -2, 3, 3], [1, -1, 2, -1]], dtype="float")
    b = np.array([[0, 9, 6]]).T
    c = np.array([[-3, 1, 3, -1]]).T
    Q = np.zeros((4, 4))
    tol = 1e-5
    kmax = 10000
    rho = .9
    mu0 = 1e4
    mumin = 1e-8

    x, _ = IP.Barrier_EqualityOnly(Q, c, A, b, tol, kmax, rho, mu0, mumin)
    true_Opt = np.array([[1, 1, 3, 0]]).T
    assert LA.norm(x - true_Opt) < 10*tol # Should be close to true optimal
    assert LA.norm(np.matmul(A, x)-b) < tol # and should satisfy Ax = b

    A = np.array([[5, -4, 13, -2, 1], [1, -1, 5, -1, 1]], dtype="float")
    b = np.array([[20, 8]]).T
    c = np.array([[1, 6, -7, 1, 5]]).T
    Q = np.zeros((5, 5))

    x, k = IP.Barrier_EqualityOnly(Q, c, A, b, tol, kmax, rho, mu0, mumin)
    true_Opt = np.array([[0, 0.5714, 1.7143, 0, 0]]).T
    assert LA.norm(x - true_Opt) < 10*tol # Should be close to true optimal
    assert LA.norm(np.matmul(A, x) - b) < tol # and should satisfy Ax = b

def test_Barrier_EqualityInequality():
    """Tests the IP.InteriorPointBarrier_EqualityInequality Method"""

    A = np.array([[-1, -3, -2], [0, 2, 1]])
    b = np.array([[-3, -1]]).T
    c = np.array([[2, 3, 6]]).T
    Q = np.zeros((3, 3))
    C = np.array([[1, -1, 1], [2, 2, -3]])
    d = np.array([[2, 0]]).T
    tol = 1e-8
    kmax = 10000
    rho = .9
    mu0 = 1e4
    mumin = 1e-9

    x, _ = IP.Barrier_EqualityInequality(Q, c, A, b, C, d, tol, kmax, rho, mu0, mumin)
    true_Opt = np.array([[1.2, 0, 0.8]]).T
    assert LA.norm(x - true_Opt) < 10*tol # should be close to true optimal
    assert LA.norm(np.matmul(C, x)-d) < tol # and should satisfy Cx = d
    assert all(np.matmul(A, x) > b) # and should satisfy Ax >= b

    A = np.zeros((3, 8), dtype="float")
    A[0, 0] = -2.0
    A[0, 1] = A[1, 0] = A[1, 1] = A[2, 0] = -1.
    b = np.array([[-1500, -1200, -500]]).T

    C = np.array([[3, 2, -1, 1,  0, 0, 0,  0],
                  [1, 1,  0, 0, -1, 1, 0,  0],
                  [1, 0,  0, 0,  0, 0, -1, 1]])
    d = np.array([[2600, 1150, 400]]).T

    c = np.array([[0, 0, 0, 2.5, 0, 0.3, 0.2, 0.2]]).T
    Q = np.zeros((8, 8))

    rho = .9
    mu0 = 100
    mumin = 1e-12

    x, _ = IP.Barrier_EqualityInequality(Q, c, A, b, C, d, tol, kmax, rho, mu0, mumin)
    true_Opt = np.array([[350, 800]]).T
    assert LA.norm(x[0:2] - true_Opt) < 10*tol
    assert LA.norm(np.matmul(C, x)-d) < tol
    assert all(np.matmul(A, x) > b)

def test_Predictor_Corrector():
    """Tests the IP.Predictor_Corrector Method"""

    A = np.array([[1,  2, -1,  1],
                  [2, -2,  3,  3],
                  [1, -1,  2, -1]])
    b = np.array([[0, 9, 6]]).T
    Q = np.zeros((4, 4))
    c = np.array([[-3, 1, 3, -1]]).T
    tol = (1e-9, 1e-9)

    x, _ = IP.Predictor_Corrector(Q, c, A, b, tol, mumin=1e-12)
    true_Opt = np.array([[1, 1, 3, 0]]).T
    assert LA.norm(x - true_Opt) < 10*tol[1] # Should be close to true optimal
    assert LA.norm(np.matmul(A, x)-b) < tol[1] # and should satisfy Ax = b

    A = np.array([[5, -4, 13, -2, 1],
                  [1, -1,  5, -1, 1]])
    b = np.array([[20, 8]]).T
    Q = np.zeros((5, 5))
    c = np.array([[1, 6, -7, 1, 5]]).T
    tol = (1e-12, 1e-12)
    kmax = (100, 1000)

    x, _ = IP.Predictor_Corrector(Q, c, A, b, tol, mumin=1e-14, kmax=kmax)
    true_Opt = np.array([[0, 0.5714, 1.7143, 0, 0]]).T
    # Not sure of the exact solution - solution this code finds has a lower cost than Iyer's "true" solution
    assert LA.norm(x - true_Opt) < 1e-4
    assert LA.norm(np.matmul(A, x)-b) < tol[1]
