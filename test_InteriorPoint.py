#!/usr/bin/env python
"""This script tests the exported functions in InterierPoint.py"""

import numpy as np
import numpy.linalg as LA
import InteriorPoint as IP

def test_InteriorPointBarrier_EqualityOnly():
    """Tests the IP.InteriorPointBarrier_EqualityOnly Method"""

    A = np.array([[1, 2, -1, 1],[2, -2, 3, 3],[1, -1, 2, -1]],dtype="float")
    b = np.array([[0, 9, 6]]).T
    c = np.array([[-3, 1, 3, -1]]).T
    Q = np.zeros((4,4))
    tol = 1e-5
    kmax = 10000
    rho = .9
    mu0 = 1e4
    mumin = 1e-8

    x,k = IP.InteriorPointBarrier_EqualityOnly(Q, c, A, b, tol, kmax, rho, mu0, mumin)
    true_Opt = np.array([[1, 1, 3, 0]]).T
    assert LA.norm(x - true_Opt) < 10*tol # Should be close to true optimal
    assert LA.norm(np.matmul(A,x)-b) < tol # and should satisfy Ax = b

    A = np.array([[5, -4, 13, -2, 1],[1, -1, 5, -1, 1]],dtype="float")
    b = np.array([[20, 8]]).T
    c = np.array([[1, 6, -7, 1, 5]]).T
    Q = np.zeros((5,5))

    x,k = IP.InteriorPointBarrier_EqualityOnly(Q, c, A, b, tol, kmax, rho, mu0, mumin)
    true_Opt = np.array([[0, 0.5714, 1.7143, 0, 0]]).T
    assert LA.norm(x - true_Opt) < 10*tol # Should be close to true optimal
    assert LA.norm(np.matmul(A,x) - b) < tol # and should satisfy Ax = b
    
def test_InteriorPointBarrier_EqualityInequality():
    """Tests the IP.InteriorPointBarrier_EqualityInequality Method"""
    
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
    
    x,k = IP.InteriorPointBarrier_EqualityInequality(Q,c,A,b,C,d,tol,kmax,rho,mu0,mumin)
    true_Opt = np.array([[1.2,0,0.8]]).T
    assert LA.norm(x - true_Opt) < 10*tol # should be close to true optimal
    assert LA.norm(np.matmul(C,x)-d) < tol # and should satisfy Cx = d
    assert all(np.matmul(A,x) > b) # and should satisfy Ax >= b