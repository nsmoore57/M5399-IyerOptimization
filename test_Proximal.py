#!/usr/bin/env python
"""This script tests the exported functions in InterierPoint.py"""

import numpy as np
import numpy.linalg as LA
from Proximal import Lasso
from Newton import GradDescent_ILS


def test_Lasso():
    """Tests the Prox.Lasso Method - Lasso is quick, but ILS is slow"""

    # Seeds for testing - need to have convergence for the Newton methods
    seeds = range(2020,2025)

    for i in seeds:
        np.random.seed(i)

        # Select random matrix size
        m = np.random.randint(8, 15)
        n = m+1
        while n >= m:
            n = np.random.randint(5, 15)

        # Select random A, y, x0
        A = np.random.normal(size=(m, n))
        y = np.random.normal(size=(m, 1))
        x0 = np.random.normal(size=(n, 1))

        # Problem Params
        tol = 1e-8
        L = max(LA.eigvalsh(np.matmul(A.T, A)))
        step_size = 1.0/L
        lamb = 0.2

        # Run LASSO
        x_Lasso, k_Lasso = Lasso(A, y, x0, tol, lamb, step_size, cost_or_pos="pos", kmax=100000)
        print("x_Lasso = ", x_Lasso)

        # Now we need code to check our results, we'll use GradDescent_BB
        Atilde = np.hstack((A, -A))
        z = np.vstack((np.maximum(0, x0), -np.minimum(0, x0)))
        Q = np.matmul(Atilde.T, Atilde)
        c = -np.matmul(Atilde.T, y) + lamb*np.sum(z)
        tol = 1e-4
        CD_tao = 1e-8

        q = (lambda z: 0.5*np.matmul(z.T, np.matmul(Q, z)) + np.matmul(c.T, z))
        x_ILS, _ = GradDescent_ILS(q, "CD", z, tol, 1000000, CD_tao=CD_tao)
        x_ILS = x_ILS[:n] - x_ILS[n:]

        Lasso_cost = 0.5*LA.norm(np.matmul(A,x_Lasso) - y)**2 + lamb*LA.norm(x_Lasso,1)
        ILS_cost = 0.5*LA.norm(np.matmul(A,x_ILS) - y)**2 + lamb*LA.norm(x_ILS,1)

        # Relative Error to ILS optimum
        assert Lasso_cost < ILS_cost or abs(Lasso_cost - ILS_cost)/ILS_cost < 0.03
