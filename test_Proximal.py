#!/usr/bin/env python
"""This script tests the exported functions in InterierPoint.py"""

import numpy as np
import numpy.linalg as LA
from Proximal import Lasso, RidgeRegression
from Newton import GradDescent_BB


def test_Lasso():
    """Tests the Prox.Lasso Method - Lasso is quick, but ILS is slow"""

    for i in range(10):
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
        x_Lasso, k_Lasso = Lasso(A, y, x0, lamb, tol, step_size, cost_or_pos="cost", kmax=100000)
        print("x_Lasso = ", x_Lasso)

        # Now we need code to check our results, we'll use GradDescent_BB
        Atilde = np.hstack((A, -A))
        z = np.vstack((np.maximum(0, x0), -np.minimum(0, x0)))
        Q = np.matmul(Atilde.T, Atilde)
        c = -np.matmul(Atilde.T, y) + lamb*np.sum(z)
        tol = 1e-3
        CD_tao = 1e-4

        q = (lambda z: 0.5*np.matmul(z.T, np.matmul(Q, z)) + np.matmul(c.T, z))
        x_BB, _ = GradDescent_BB(q, "CD", z, tol, 1000000, CD_tao=CD_tao)
        x_BB = x_BB[:n] - x_BB[n:]

        Lasso_cost = 0.5*LA.norm(np.matmul(A,x_Lasso) - y)**2 + lamb*LA.norm(x_Lasso,1)
        BB_cost = 0.5*LA.norm(np.matmul(A,x_BB) - y)**2 + lamb*LA.norm(x_BB,1)

        # Relative Error to ILS optimum
        assert Lasso_cost < BB_cost or abs(Lasso_cost - BB_cost)/BB_cost < 0.03

def test_RidgeRegression():
    """Tests the Prox.RidgeRegression Method"""

    for i in range(10):
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

        # Run RidgeRegression
        x_RR, _ = RidgeRegression(A, y, x0, lamb, tol, step_size, cost_or_pos="cost", kmax=100000)
        print("x_RR = ", x_RR)

        # Now we need code to check our results, we'll use GradDescent_BB
        Q = np.matmul(A.T, A) + lamb*np.eye(A.shape[1])
        c = -np.matmul(A.T, y)
        tol = 1e-3
        CD_tao = 1e-4
        q = (lambda x: 0.5*np.matmul(x.T, np.matmul(Q, x)) + np.matmul(c.T, x))

        # Run BB
        x_BB, _ = GradDescent_BB(q, "CD", x0, tol, 100000, CD_tao=CD_tao)

        RR_cost = 0.5*LA.norm(np.matmul(A,x_RR) - y)**2 + lamb*LA.norm(x_RR,1)
        BB_cost = 0.5*LA.norm(np.matmul(A,x_BB) - y)**2 + lamb*LA.norm(x_BB,1)

        # Relative Error to ILS optimum
        assert RR_cost < BB_cost or abs(RR_cost - BB_cost)/BB_cost < 0.03
