import numpy as np
import LinearAlgebra as testLA

def test_forwardSubstitution_LowerTri():
    """Test LinearAlgebra.forwardSubstitution_LowerTri function"""
    L = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])

    b = np.array([[1, 1, 1, 1]]).transpose()
    sol = np.array([[1, 0, 0, 0]]).transpose()
    assert all(testLA.forwardSubstitution_LowerTri(L, b) == sol)

    b = np.array([[1, 2, 3, 4]]).transpose()
    sol = np.array([[1, 1, 1, 1]]).transpose()
    assert all(testLA.forwardSubstitution_LowerTri(L, b) == sol)


def test_backSubstitution_UpperTri():
    """Test LinearAlgebra.backSubstitution_UpperTri function"""

    U = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]).transpose()


    b = np.array([[1, 1, 1, 1]]).transpose()
    sol = np.array([[0, 0, 0, 1]]).transpose()
    assert all(testLA.backSubstitution_UpperTri(U, b) == sol)

    b = np.array([[4, 3, 2, 1]]).transpose()
    sol = np.array([[1, 1, 1, 1]]).transpose()
    assert all(testLA.backSubstitution_UpperTri(U, b) == sol)
