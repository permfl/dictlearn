# Ryan
from dictlearn.linalg import *


def test_chap2():
    n = 10
    d = 9
    b = random.random((n, 1))

    A = matrix(zeros((n, n)))
    for k in range(n):
        for s in range(d + 1):
            if k + s < n:
                A[k + s, k] = random.random()
    bcopy = b.copy()
    rforwardsolve(A, bcopy, d)
    xver = linalg.solve(A, b)
    assert abs(xver - bcopy).max() < 1E-4, 'rforwardsolve failed'

    bcopy = b.copy()
    cforwardsolve(A, bcopy, d)
    assert abs(xver - bcopy).max() < 1E-4, 'cforwardsolve failed'

    b = random.random((n, 1))
    A = zeros((n, n))
    for k in range(n):
        for s in range(d + 1):
            if k + s < n:
                A[k, k + s] = random.random()
    bcopy = b.copy()
    rbacksolve(A, bcopy, d)
    xver = linalg.solve(A, b)
    assert abs(xver - bcopy).max() < 1E-4, 'rbacksolve failed'

    bcopy = b.copy()
    cbacksolve(A, bcopy, d)
    assert abs(xver - bcopy).max() < 1E-4, 'cbacksolve failed'

    A = matrix(zeros((n, n)))
    for k in range(n):
        for s in range(n):
            if abs(k - s) <= d:
                A[k, s] = random.random()
    x = matrix(random.random((n, 1)))
    L, U = L1U(A, d)
    assert abs(L * U * x - A * x).max() < 1E-4, 'L1U failed'

    Acopy = A.copy()
    L1Uv2(Acopy, d)
    for k in range(n):
        assert abs(Acopy[k, k] - U[k, k]) < 1E-4, 'L1Uv2 failed'
        for l in range(k):
            assert abs(Acopy[k, l] - L[k, l]) < 1E-4, 'L1Uv2 failed'
            assert abs(Acopy[l, k] - U[l, k]) < 1E-4, 'L1Uv2 failed'


def test_chap3():
    n = 10
    d = 9
    A = matrix(zeros((n, n)))
    for k in range(n):
        for s in range(k + 1):
            if abs(k - s) <= d:
                A[k, s] = random.random()
                A[s, k] = A[k, s]
    L, dg = LDL(A, d)
    L = matrix(L)
    dg = diag(dg)
    newmatr = L * matrix(dg) * L.T
    assert abs(A - newmatr).max() < 1E-4, 'LDL failed'

    A = matrix(zeros((n, n)))
    for k in range(n):
        for s in range(k + 1):
            if abs(k - s) <= d:
                A[s, k] = random.random()
    A = matrix(A) * matrix(A).T
    L = bandcholesky(A, d)
    L = matrix(L)
    newmatr = L * L.T
    assert abs(A - newmatr).max() < 1E-4, 'bandcholesky failed'

    Acopy = A.copy()
    L = bandsemicholeskyL(Acopy, d)
    L = matrix(L)
    newmatr = L * L.T
    assert abs(A - newmatr).max() < 1E-4, 'bandsemicholeskyL failed'


def test_chap4():
    A = array([[1, 3, 1], [1, 3, 7], [1, -1, -4], [1, -1, 2]])
    R, C = housetriang(A, eye(4))
    assert abs(matrix(linalg.inv(C)) * R - A).max() < 1E-4, 'housetriang failed'

    n = 10
    A = matrix(zeros((n, n)))
    for k in range(n):
        for s in range(k + 2):
            if s < n:
                A[s, k] = random.random()
    b = random.random((n, 1))
    xexact = linalg.solve(A, b)
    x = rothesstri(A, b)
    assert abs(x - xexact).max() < 1E-4, 'rothesstri failed'