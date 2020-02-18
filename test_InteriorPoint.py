#!/usr/bin/env python
"""This script tests the exported functions in InterierPoint.py"""

import numpy as np
import numpy.linalg as LA
import InteriorPoint as IP

def test_InteriorPointBarrier():
    A = np.array([[1, 2, -1, 1],[2, -2, 3, 3],[1, -1, 2, -1]],dtype="float")
    b = np.array([[0, 9, 6]]).transpose()
    tol = 1e-5
    kmax = 10000
    rho = .9
    mu0 = 1e4
    mumin = 1e-8
    
    x,k = IP.InteriorPointBarrier(A, b, tol, kmax, rho, mu0, mumin)
    assert LA.norm(np.matmul(A,x)-b) < 1e-8