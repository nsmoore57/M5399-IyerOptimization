import numpy.linalg as LA
import numpy as np

def Newton(f, x0, tao, c, tol, kmax):
    return None

def GradDescent(f, x0, tol, kmax):
    k =1
    xold = np.zeros(x0.shape)
    xnew = x0
    dold = np.zeros(x0.shape)
    while LA.norm(f(xnew)) > tol and k < kmax:
        dnew = -f(xnew)
        gamma = abs(np.matmul((xnew - xold).transpose(),-dnew + dold))/LA.norm(-dnew+dold)**2
        xold = xnew
        dold = dnew
        xnew = xold + gamma*dnew
        k = k+1
    if k >= kmax:
        print("kmax exceeded - don't trust the answer")
    return xnew

if __name__ == "__main__":
    import numpy as np
    def Rosenbrock(a, x):
        """
        Function for testing optimization, vector-valued to repeat same value
        in both dimensions
        """
        res = (a - x[0])**2 + 100*(x[1]-x[0]**2)**2
        return np.vstack((res,res))
    def Rosenbrock_Jac(a, x):
        """
        Function for testing optimization, Jacobian of Rosenbrock function
        """
        df1dx1 = -2*(a-x[0]) - 400*(x[1]-x[0]**2)*x[0]
        df1dx2 = 200*(x[1]-x[0])
        return np.array([[df1dx1, df1dx2], [df1dx1, df1dx2]])

    def arctan2d(x):
        res = np.arctan(x[0] - np.pi/4)
        return np.vstack((res,np.array([0])))


    x0 = np.array([10.1])
    tol = 10**(-8)
    kmax = 100
    # print(GradDescent((lambda x: Rosenbrock(2, x)), x0, tol, kmax))
    print(GradDescent((lambda x: np.arctan(x - np.pi/4)), x0, tol, kmax))
