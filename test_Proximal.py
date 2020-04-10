#!/usr/bin/env python
"""This script tests the exported functions in Proximal.py"""

import numpy as np
import numpy.linalg as LA
from Proximal import Lasso, RidgeRegression, ElasticNet
from Newton import GradDescent_BB

def test_Lasso():
    """Tests the Prox.Lasso Method"""

    for i in range(10):
        # Select random matrix size
        m = np.random.randint(8, 50)
        n = m+1
        while n >= m:
            n = np.random.randint(5, 50)

        # Select random A, y, x0
        A = np.random.normal(size=(m, n))
        y = np.random.normal(size=(m, 1))
        x0 = np.random.normal(size=(n, 1))

        # Problem Params
        lamb = 0.2
        tol = 1e-8
        cost_or_pos = "cost" if i < 5 else "pos"

        # Run LASSO
        x_Lasso, _ = Lasso(A, y, x0, lamb, tol, cost_or_pos=cost_or_pos)
        print("x_Lasso = ", x_Lasso)

        # Now we need code to check our results, we'll use GradDescent_BB
        Atilde = np.hstack((A, -A))
        z = np.vstack((np.maximum(0, x0), -np.minimum(0, x0)))
        Q = np.matmul(Atilde.T, Atilde)
        c = -np.matmul(Atilde.T, y) + lamb*np.ones(z.shape)
        tol = 1e-3
        CD_tao = 1e-4

        q = (lambda z: 0.5*np.matmul(z.T, np.matmul(Q, z)) + np.matmul(c.T, z))
        x_BB, _ = GradDescent_BB(q, "CD", z, tol, 1000000, CD_tao=CD_tao)
        x_BB = x_BB[:n] - x_BB[n:]

        Lasso_cost = 0.5*LA.norm(np.matmul(A, x_Lasso) - y)**2 + lamb*LA.norm(x_Lasso, 1)
        BB_cost = 0.5*LA.norm(np.matmul(A, x_BB) - y)**2 + lamb*LA.norm(x_BB, 1)

        # Relative Error to BB optimum
        assert Lasso_cost < BB_cost or abs(Lasso_cost - BB_cost)/BB_cost < 0.03

def test_RidgeRegression():
    """Tests the Prox.RidgeRegression Method"""

    for i in range(10):
        # Select random matrix size
        m = np.random.randint(8, 50)
        n = m+1
        while n >= m:
            n = np.random.randint(5, 50)

        # Select random A, y, x0
        A = np.random.normal(size=(m, n))
        y = np.random.normal(size=(m, 1))
        x0 = np.random.normal(size=(n, 1))

        # Problem Params
        lamb = 0.2
        tol = 1e-8
        cost_or_pos = "cost" if i < 5 else "pos"

        # Run RidgeRegression
        x_RR, _ = RidgeRegression(A, y, x0, lamb, tol, cost_or_pos=cost_or_pos)

        # Now we need code to check our results, we'll use GradDescent_BB
        Q = np.matmul(A.T, A) + lamb*np.eye(A.shape[1])
        c = -np.matmul(A.T, y)
        tol = 1e-3
        CD_tao = 1e-4
        q = (lambda x: 0.5*np.matmul(x.T, np.matmul(Q, x)) + np.matmul(c.T, x))

        # Run BB
        x_BB, _ = GradDescent_BB(q, "CD", x0, tol, 100000, CD_tao=CD_tao)

        RR_cost = 0.5*LA.norm(np.matmul(A, x_RR) - y)**2 + lamb*LA.norm(x_RR, 1)
        BB_cost = 0.5*LA.norm(np.matmul(A, x_BB) - y)**2 + lamb*LA.norm(x_BB, 1)

        # Relative Error to BB optimum
        assert RR_cost < BB_cost or abs(RR_cost - BB_cost)/BB_cost < 0.03

def test_ElasticNet():
    """Tests for the Prox.ElasticNet Method"""
    for i in range(10):
        # Select random matrix size
        m = np.random.randint(8, 50)
        n = m+1
        while n >= m:
            n = np.random.randint(5, 50)

        # Select random A, y, x0
        A = np.random.normal(size=(m, n))
        y = np.random.normal(size=(m, 1))
        x0 = np.random.normal(size=(n, 1))

        # Problem Params
        lamb = 0.2
        alpha = 0.5
        tol = 1e-8
        cost_or_pos = "cost" if i < 5 else "pos"

        # Run ElasticNet
        x_EN, _ = ElasticNet(A, y, x0, lamb, alpha, tol, cost_or_pos=cost_or_pos)

        # Now we need code to check our results, we'll use GradDescent_BB
        Atilde = np.hstack((A, -A))
        z = np.vstack((np.maximum(0, x0), -np.minimum(0, x0)))
        Q = np.matmul(Atilde.T, Atilde) + (1-alpha)*lamb*np.eye(Atilde.shape[1])
        c = -np.matmul(Atilde.T, y) + alpha*lamb*np.ones(z.shape)
        tol = 1e-3
        CD_tao = 1e-4

        # Cost function
        q = (lambda z: 0.5*np.matmul(z.T, np.matmul(Q, z)) + np.matmul(c.T, z))

        # Run BB and time it
        x_BB, _ = GradDescent_BB(q, "CD", z, tol, 100000, CD_tao=CD_tao)
        x_BB = x_BB[:n] - x_BB[n:]

        EN_cost = 0.5*LA.norm(np.matmul(A, x_EN) - y)**2 \
                + lamb*(alpha*LA.norm(x_EN, 1) \
                + ((1-alpha)/2)*LA.norm(x_EN)**2)
        BB_cost = 0.5*LA.norm(np.matmul(A, x_BB) - y)**2 \
                + lamb*(alpha*LA.norm(x_BB, 1) \
                + ((1-alpha)/2)*LA.norm(x_BB)**2)
        # Relative Error to BB optimum
        assert EN_cost < BB_cost or abs(EN_cost - BB_cost)/BB_cost < 0.03
