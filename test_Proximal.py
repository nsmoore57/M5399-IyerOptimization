#!/usr/bin/env python
"""This script tests the exported functions in Proximal.py"""

import numpy as np
import numpy.linalg as LA
import Proximal as Prox
from Newton import GradDescent_BB

def test_Lasso():
    """Tests the Prox.Lasso Function"""

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
        x_Lasso, _ = Prox.Lasso(A, y, x0, lamb, tol, cost_or_pos=cost_or_pos)
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
    """Tests the Prox.RidgeRegression Function"""

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
        x_RR, _ = Prox.RidgeRegression(A, y, x0, lamb, tol, cost_or_pos=cost_or_pos)

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
        x_EN, _ = Prox.ElasticNet(A, y, x0, lamb, alpha, tol, cost_or_pos=cost_or_pos)

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

def test_Prox_Prob1():
    """
    Tests Prox on Projection to Affine Cx=d
    min x_1^2 + x_2^2 + x_3^2 + x_4^2 - 2x_1 - 3x_4
    subj to:
      2x_1 + x_2 +  x_3 + 4x_4 = 7
       x_1 + x_2 + 2x_3 +  x_4 = 6
    """

    Q = np.eye(4)
    c = np.array([[-2, 0, 0, -3]]).T
    C = np.array([[2, 1, 1, 4], [1, 1, 2, 1]])
    d = np.array([[7, 6]]).T

    x0 = np.random.normal(size=(4, 1))
    gradf = (lambda x: 2*x + c)
    proj = Prox.Get_Proj_EqualityAffine_Func(C, d)
    proxg = (lambda v, theta: proj(v))
    lamb = 0.2
    tol = 1e-9
    step_size = 1e-4
    cost = (lambda x: np.matmul(x.T, np.matmul(Q, x)) + np.matmul(c.T, x))

    x_Iyer = np.array([[1.12, 0.65, 1.83, 0.57]]).T

    x, _ = Prox.ProximalMethod(x0, gradf, proxg, lamb, tol, step_size, cost)
    assert all(np.matmul(C, x) - d > -1e-8)
    assert cost(x_Iyer) >= cost(x)

def test_Prox_Prob2():
    """
    Tests Prox on Projection to intersection of Affine Ax >= b, 2norm ball, and octant
    min x_1^2 + x_2^2 + x_3^2 + x_4^2 - 2x_1 - 3x_4
    subj to:
      2x_1 + x_2 +  x_3 + 4x_4 <= 7
       x_1 + x_2 + 2x_3 +  x_4 <= 6
                    norm_2(x) <= sqrt(2)
                      x_1, x_2 >= 0
    """
    Q = np.eye(4)
    c = np.array([[-2, 0, 0, -3]]).T
    A = -1*np.array([[2, 1, 1, 4], [1, 1, 2, 1]])
    b = -1*np.array([[7, 6]]).T

    x0 = np.random.normal(size=(4, 1))
    gradf = (lambda x: 2*x + c)

    # Project onto 2-norm ball of radius sqrt(2)
    proj1 = (lambda v: Prox.Proj_2NormBall(v, np.sqrt(2)))

    # Project onto First Octant
    proj2 = (lambda v: np.maximum(v, 0))

    # Project onto affine Ax >= b
    proj3 = Prox.Get_Proj_InequalityAffine_Func(A, b)

    # Now the Prox operator is the alternating method between the two
    proxg = (lambda v, theta: Prox.Proj_Intersection(v, (proj1, proj2, proj3), tol=1e-6))

    lamb = 0.005
    tol = 1e-6
    step_size = 1e-2

    # Use the cost function as the stopping criteria
    cost = (lambda x: np.matmul(x.T, np.matmul(Q, x)) + np.matmul(c.T, x))

    # Not actual truth - but known to be close
    x_true = np.array([[0.78436313], [0.00264116], [0.00277106], [1.17675819]])

    x, _ = Prox.ProximalMethod(x0, gradf, proxg, lamb, tol, step_size, cost, kmax=1e6)
    assert all(np.matmul(A, x) - b > 0)
    assert LA.norm(x) < np.sqrt(2) + 1e-5
    assert min(x) > -1e-4
    assert abs(cost(x) - cost(x_true)) < 1e-4

def test_Nesterov_Accel_Prox():
    """
    Using Nesterov Acceleration
    min x_1^2 + (x_1 + x_2)^2 - 10(x_1 + x_2)
    subject to: 3x_1 + x_2 <= 6
                norm2(x) <= sqrt(5)
    """
    Q = np.array([[2, 1], [1, 1]])
    c = np.array([[-10, -10]]).T
    A = -1*np.array([[3, 1]])
    b = -1*np.array([[6]]).T

    x0 = np.random.normal(size=(2, 1))
    gradf = (lambda x: 2*np.matmul(Q.T, x) + c)

    # Project onto 2-norm ball of radius 2
    proj1 = (lambda v: Prox.Proj_2NormBall(v, np.sqrt(5)))

    # Project onto affine Ax >= b
    proj2 = Prox.Get_Proj_InequalityAffine_Func(A, b)

    # Now the Prox operator is the alternating method between the two
    proxg = (lambda v, theta: Prox.Proj_Intersection(v, (proj1, proj2), tol=1e-6))

    lamb = 0.005
    tol = 1e-8
    step_size = 1e-5
    # Use the cost function as the stopping criteria
    cost = (lambda x: np.matmul(x.T, np.matmul(Q, x)) + np.matmul(c.T, x))

    # Set up the acceleration
    eigs = LA.eigvalsh(Q)
    accel_args = (min(eigs), max(eigs))

    x_true = np.array([[1, 2]]).T

    x, k = Prox.ProximalMethod(x0, gradf, proxg, lamb, tol, step_size, cost, kmax=1e6,
                               accel="nesterov", accel_args=accel_args)
    assert all(np.matmul(A,x) - b > -1e-8)
    assert LA.norm(x) < np.sqrt(5) + 1e-6
    assert abs(cost(x) - cost(x_true)) < 1e-3


def test_FISTA_Accel():
    """
    Using FISTA Acceleration
    min x_1^2 + (x_1 + x_2)^2 - 10(x_1 + x_2)
    subject to: 3x_1 + x_2 <= 6
                norm2(x) <= sqrt(5)
    """
    Q = np.array([[2, 1], [1, 1]])
    c = np.array([[-10, -10]]).T
    A = -1*np.array([[3, 1]])
    b = -1*np.array([[6]]).T

    x0 = np.random.normal(size=(2, 1))
    gradf = (lambda x: 2*np.matmul(Q.T, x) + c)

    # Project onto 2-norm ball of radius 2
    proj1 = (lambda v: Prox.Proj_2NormBall(v, np.sqrt(5)))

    # Project onto affine Ax >= b
    proj2 = Prox.Get_Proj_InequalityAffine_Func(A, b)

    # Now the Prox operator is the alternating method between the two
    proxg = (lambda v, theta: Prox.Proj_Intersection(v, (proj1, proj2), tol=1e-6))

    lamb = 0.005
    tol = 1e-8
    step_size = 1e-5
    # Use the cost function as the stopping criteria
    cost = (lambda x: np.matmul(x.T, np.matmul(Q, x)) + np.matmul(c.T, x))

    x_true = np.array([[1, 2]]).T

    x, k = Prox.ProximalMethod(x0, gradf, proxg, lamb, tol, step_size, cost, kmax=1e6, accel="fista")
    assert all(np.matmul(A,x) - b > -1e-8)
    assert LA.norm(x) < np.sqrt(5) + 1e-6
    assert abs(cost(x) - cost(x_true)) < 1e-3
