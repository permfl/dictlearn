# -*- coding: utf-8 -*-
# Øyvind Ryan, oyvindry@math.uio.no
# MAT-INF4130

from __future__ import print_function
from numpy import *


def trifactor(a, d, c):
    """
    Compute the entries in the LU-factorization of a tridiagonal matrix A
     using algorithm 1.8. 
    Returns the sequence l on the sub-diagonal of L, and the sequence u on 
    the diagonal of U. 
    
    a: The n-1 entries on the sub-diagonal of A
    d: The n entries on the diagonal of A
    c: The n-1 entries on the super-diagonal of A
    """
    u, l = zeros_like(d), zeros_like(a)
    u[0] = d[0]
    for k in range(len(a)):
        l[k] = a[k]/u[k]
        u[k+1] = d[k+1] - l[k]*c[k]
    return l, u


def trisolve(l, u, c, b):
    """
    Compute the solution to a system on the form A*x=b with A a tridiagonal 
    matrix, based on the LU-factorization of A 
    as given by the function trifactor (using algorithm 1.9). 
    The input vector b will be updated with the solution x (i.e. in-place).
    
    l: The entries on the sub-diagonal of L in the LU-factorization of A
    u: The entries on the diagonal of U in the LU-factorization of A
    c: The entries on the super-diagonal of U in the LU-factorization of A 
    (this equals the entries on the super-diagonal of A)
    b: The right hand side
    """
    n = shape(b)[0]
    for k in range(1, n):
        b[k] -= l[k-1]*b[k - 1]
    b[n-1] /= u[n-1]
    for k in range(n-2,-1,-1):
        b[k] -= c[k]*b[k + 1]
        b[k] /= u[k]


def splineint(a, b, y, mu1, munp1):
    """
    Use algorithm 1.19 to solve the D_2 spline problem.
    Returns the left end points of all sub-inetrvals, and a matrix holding 
    all coefficients in all the piecewise cubic polynomials 
    (one row for each piecewise polynomial)
    
    a: The left end point of the interval
    b: The rigth end point of the interval
    y: The values at the interpolation points
    mu1 and munp1: Initial conditions/moments.
    """
    n = len(y) - 1
    h = (b - a)/float(n)
    c, d = ones(n - 2), 4*ones(n - 1)
    l ,u = trifactor(c, d, c)
    b1 = (6/h**2)*(y[2:(n + 1)] - 2*y[1:n] + y[:(n - 1)])
    b1[0] -= mu1
    b1[n - 2] -= munp1
    trisolve(l, u, c, b1)
    mu2ton = b1
    mu = zeros(n + 1)
    mu[0] = mu1
    mu[1:n] = mu2ton
    mu[n] = munp1
    
    C = array(zeros((n, 4))) # Apply lemma 1.3 to obtain the coefficient matrix
    C[:, 0] = y[:n]
    C[:, 1] = (y[1:(n+1)] - y[:n])/h - h*mu[:n]/3-h*mu[1:(n+1)]/6
    C[:, 2] = mu[:n]/2
    C[:, 3] = (mu[1:(n + 1)] - mu[0:n])/(6*h)
    C = matrix(C)
    return linspace(a, b - h, n), C


def findsubintervals (t ,x):
    """
    Use algorithm 1.21 to return an index set for the points in x, 
    saying which intervals they belong to.
    
    t: the start points of the intervals
    x: The points to find the interval indices for.
    """
    k, m = len(t), len(x)
    if k<2:
        return zeros(m,1)
    else:
        j = concatenate([t, x]).argsort()
        i = nonzero(j >= k)
        arr = arange(0,m)
        arr =  i - arr - 1
        arr = arr[0]
        return arr


def splineval(x,C,X):
    """
    Use algorithm 1.22 to compute and return the values of a cubic spline 
    (x, C) at points X. (x, C) can be obtained with the function splineint. 
     
    x: The start points of the intervals.
    C: A matrix holding all coefficients in all the piecewise cubic 
    polynomials (one row for each piecewise polynomial)
    X: The points we want to evaluate for the spline
    """
    m = len(X)
    i = findsubintervals(x,X) 
    G = zeros(m)
    for j in range(m):
        k = i[j]
        t = X[j] - x[k]
        G[j]=C[k,:]* t**array([[0],[1],[2],[3]])
    return G


# Chapter 2


def rforwardsolve(A, b, d):
    """
    Solve the system A*x = b for a lower triangular d-banded matrix A, 
    using algorithm 2.6 (row-oriented forward substitution)
    The input vector b will be updated with the solution x (i.e. in-place).
    
    A: A lower triangular matrix
    b: The right hand side
    d: The band-width
    """
    n = len(b)
    b[0] /= A[0, 0]
    for k in range(1,n):
        lk = array([0,k-d]).max()
        b[k] = b[k] - dot(A[k, lk:k],b[lk:k])
        b[k] /= A[k, k]
    

def rbacksolve(A, b, d):
    """
    Solve the system A*x = b for an upper triangular d-banded matrix A, 
    using algorithm 2.7 (row-oriented backward substitution)
    The input vector b will be updated with the solution x (i.e. in-place).
    
    A: An upper triangular matrix
    b: The right hand side
    d: The band-width
    """
    n = len(b)
    b[n - 1] /= A[n - 1,n - 1]
    for k in range(n-2,-1,-1):
        uk = array([n, k + d + 1]).min()
        b[k] = b[k] - dot(A[k,(k+1):uk], b[(k+1):uk])
        b[k] /= A[k,k]
    

def cforwardsolve(A, b, d):
    """
    Solve the system A*x = b for a lower triangular d-banded matrix A, 
    using algorithm 2.9 (column-oriented forward substitution)
    The input vector b will be updated with the solution x (i.e. in-place).
    
    A: A lower triangular matrix
    b: The right hand side
    d: The band-width
    """
    A = matrix(A)
    n = len(b)
    for k in range(n-1):
        b[k] /= A[k, k]
        uk = array([n, k + d + 1]).min()
        b[(k+1):uk] -= A[(k+1):uk, k]*b[k]
    b[n - 1] /= A[n - 1,n - 1] 
    

def cbacksolve(A, b, d):
    """
    Solve the system A*x = b for a upper triangular d-banded matrix A, 
    using algorithm 2.10 (column-oriented backward substitution)
    The input vector b will be updated with the solution x (i.e. in-place).
    
    A: An upper triangular matrix
    b: The right hand side
    d: The band-width
    """
    A = matrix(A)
    n = len(b)
    for k in range(n - 1,0,-1):
        b[k] /= A[k, k]
        lk = array([0, k - d]).max()
        b[lk:k] -= (A[lk:k, k]*b[k])
    b[0] /= A[0,0]
    

# Algorithm 2.18
def L1U(A, d):
    """
    Compute the L1U-factorization of the d-banded matrix A.
    Returns a lower triangular matrix L with with ones on the diagonal, 
    and an upper triangular matrix U. 
    Note that, for d=1, the algorithm is equivalent to trisolve.
    
    A: A matrix
    d: The band-width
    """
    n = shape(A)[0]
    L = eye(n)
    U = matrix(zeros((n,n))); U[0,0] = A[0,0]
    for k in range(1,n):
        km = array([0, k - d]).max()
        if km < k:
            L[k, km:k] = A[k, km:k]
            rforwardsolve(U[km:k, km:k].T, L[k, km:k].T, d) # L
        U[km:(k + 1), k] = A[km:(k + 1), k]
        rforwardsolve(L[km:(k + 1), km:(k + 1)], U[km:(k + 1), k], d) # U
    return L, U


def L1Uv2(A, d):
    """
    Compute the L1U-factorization of the d-banded matrix A.
    This is more advanced than L1U in the sense that it avoids allocation of 
    new arrays: The result is stored directly in A (i.e. in-place), i.e.
    overwriting the previous A. The entries in A and U are at the same places 
    in A, and the diagonal with ones in L is not stored in A.
    In order to achieve this, we can't use the function rforwardsolve, as L1U 
    uses it (due to the special form with ones on the diagonal).
    
    A: A matrix
    d: The band-width
    """
    n = shape(A)[0]
    for k in range(1,n):
        km = array([0, k - d]).max() # First index of r we need to update
        for r in range(km, k - 1):
            A[k, r] /= A[r, r]
            uk = array([k, r + d + 1]).min() # last index not included
            A[k, (r + 1):uk] -= A[r, (r + 1):uk]*A[k, r]
        A[k, k - 1] /= A[k - 1,k - 1] 
        for r in range(km, k):
            uk = array([k + 1, r + d + 1]).min() # last index not included
            A[(r + 1):uk, k] -= A[(r + 1):uk, r]*A[r, k]
                   
# Chapter 3


def LDL(A, d):
    """
    Find the LDL factorization of the hermitian matrix A, using algorithm 3.4
    
    A: A hermitian matrix
    d: The band-width
    """
    n = shape(A)[0]
    L = array(eye(n))
    dg = zeros(n)
    dg[0] = A[0, 0]
    for k in range(1, n):
        m = reshape(array(A[:k, k].copy()), k)
        rforwardsolve(L[:k, :k], m, d)
        L[k, :k] = m/dg[:k]
        dg[k] = A[k, k] - dot(L[k, :k], m)
    return L, dg 


def bandcholesky(A, d):
    """
    Find the Cholesky factorization of the positive definite matrix A, 
    using algorithm 3.13
    
    A: A positive definite matrix
    d: The band-width
    """
    L, dg = LDL(A, d)
    return matrix(L)*diag(sqrt(dg))


def bandsemicholeskyL(A, d):
    """
    Find the Cholesky factorization of the positive semidefinite matrix A, 
    using algorithm 3.24 
    The input matrix A will be changed by this method
    
    A: A positive semidefinite matrix
    d: The band-width
    """
    n = shape(A)[0]
    for k in range(n):
        if A[k,k] > 0:
            kp=array([n, k + 1 + d]).min();
            A[k,k] = sqrt(A[k,k])
            A[(k+1):kp, k] =A [(k+1):kp, k]/A[k, k]
            for j in range(k+1, kp):
                A[j:kp, j] = A[j:kp, j] - A[j, k]*A[j:kp, k]
        else:
            A[k:kp, k] = 0
    return tril(A)
    
# Chapter 4


def housegen(x): 
    """
    Return u and a in Theorem 4.16, using algorithm 4.17
    
    x: The vector to reflect on ae_1
    """
    a = linalg.norm(x)
    if a == 0:
        u=x; u[0]=sqrt(2); return u, a
    if x[0] == 0:
        r = 1
    else:
        r =x[0]/abs(x[0])
    u = conj(r)*x/a
    u[0]=u[0]+1
    u=u/sqrt(u[0])
    a=-r*a
    return u, a


def housetriang(A, B):
    """
    Return R, C so that R is upper trapezoidal and resulting from a series 
    of Householder transformations to A, and C so that it results from 
    applying the same series of Householder transformations to B.
    This is algorithm 4.21

    A: The matrix to apply Householder transformations to, in order to 
    transform it to upper trapezoidal form.
    B: The matrix to apply the same Householder transformations to. 
    """
    m, n = shape(A); r = shape(B)[1] ; A=hstack([A,B]); 
    minval = array([n, m - 1]).min()
    for k in range(minval):
        v, A[k, k] = housegen(A[k:m, k])
        v = matrix(reshape(v, (m - k, 1)))
        C = A[k:m, (k+1):(n+r)] ; A[k:m, (k + 1):(n + r)] = C - v*(v.T*C)
    R = triu(A[:, :n]); C = A[:, n:(n + r)]
    return R, C


def rothesstri(A, b):
    """
    Solve the system Ax=b where A is in upper Hessenberg form using 
    Givens rotations, using algorithm 4.36
    
    A: A matrix in upper Hessenberg form.
    b: The right hand side.
    """
    n = shape(A)[0]
    A = hstack([A, b])
    for k in range(n-1):
        r = linalg.norm([ A[k , k] , A[k + 1, k] ])
        if r>0:
            c=A[k, k]/r; s=A[k + 1, k]/r
            A[[k, k + 1],(k + 1):(n + 1)]=[[c, s],[-s, c]]*A[[k, k + 1],(k + 1):(n + 1)]
        A[k, k] = r; A[k+1,k] = 0
    z = A[:, n].copy()
    rbacksolve(A[:, :n], z, n)
    return z
      
# testcode


def _test_chap1(): 
    import matplotlib.pyplot as plt
    print('Testing all 5 algorithms from chapter 1 with a spline plot')
    xplot = linspace(0, 1, 101)
    xknot = linspace(0, 1, 11)
    yknot = xknot**4
    # xknot is reduced with one in size
    xknot, C = splineint(0, 1, yknot, 0, 12) 
    G = splineval(xknot, C, xplot)
    plt.plot(xplot,G)
    plt.show()


def _test_chap2():
    print('Testing all algorithms from chapter 2')
    
    n = 10
    d = 9
    b = random.random((n, 1))
    
    print('Testing rforwardsolve')
    A = matrix(zeros((n, n)))
    for k in range(n):
        for s in range(d+1):
            if k+s < n:
                A[k + s, k] = random.random()
    bcopy = b.copy()
    rforwardsolve(A, bcopy, d)
    xver = linalg.solve(A,b)
    assert abs(xver-bcopy).max() < 1E-4, 'rforwardsolve failed'

    print('Testing cforwardsolve')
    bcopy = b.copy()
    cforwardsolve(A, bcopy, d)
    assert abs(xver-bcopy).max() < 1E-4, 'cforwardsolve failed'
    
    b = random.random((n,1))
    print ('Testing rbacksolve')
    A = zeros((n, n))
    for k in range(n):
        for s in range(d + 1):
            if k+s < n:
                A[k, k + s] = random.random()
    bcopy = b.copy()
    rbacksolve(A, bcopy, d)
    xver = linalg.solve(A,b)
    assert abs(xver-bcopy).max() < 1E-4, 'rbacksolve failed'

    print ('Testing cbacksolve')
    bcopy = b.copy()
    cbacksolve(A, bcopy, d)
    assert abs(xver-bcopy).max() < 1E-4, 'cbacksolve failed'

    print ('Testing L1U')
    A = matrix(zeros((n, n)))
    for k in range(n):
        for s in range(n):
            if abs(k-s) <= d:
                A[k, s] = random.random()
    x = matrix(random.random((n, 1)))
    L, U = L1U(A, d)
    assert abs(L*U*x -A*x).max() < 1E-4, 'L1U failed'

    print ('Testing L1Uv2')
    Acopy = A.copy()
    L1Uv2(Acopy, d)
    for k in range(n):
        assert abs(Acopy[k,k] - U[k, k]) < 1E-4, 'L1Uv2 failed'
        for l in range(k):
            assert abs(Acopy[k,l] - L[k, l]) < 1E-4, 'L1Uv2 failed'
            assert abs(Acopy[l,k] - U[l, k]) < 1E-4, 'L1Uv2 failed'
 
def _test_chap3():
    print ('Testing LDL')
    n = 10
    d = 9
    A = matrix(zeros((n, n)))
    for k in range(n):
        for s in range(k + 1):
            if abs(k-s) <= d:
                A[k, s] = random.random()
                A[s, k] = A[k, s]
    L, dg = LDL(A, d)
    L = matrix(L)
    dg = diag(dg)
    newmatr = L*matrix(dg)*L.T
    assert abs(A - newmatr).max() < 1E-4, 'LDL failed'

    print ('Testing bandcholesky')
    A = matrix(zeros((n, n)))
    for k in range(n):
        for s in range(k + 1):
            if abs(k-s) <= d:
                A[s, k] = random.random()
    A = matrix(A)*matrix(A).T
    L = bandcholesky(A, d)
    L = matrix(L)
    newmatr = L*L.T
    assert abs(A - newmatr).max() < 1E-4, 'bandcholesky failed'

    print ('Testing bandsemicholeskyL')
    Acopy = A.copy()
    L = bandsemicholeskyL(Acopy, d)
    L = matrix(L)
    newmatr = L*L.T
    assert abs(A - newmatr).max() < 1E-4, 'bandsemicholeskyL failed'


def _test_chap4():
    print('Testing housegen and housetriang')
    A = array([[1,3,1],[1,3,7],[1,-1,-4],[1,-1,2]])
    R, C = housetriang(A,eye(4))
    assert abs(matrix(linalg.inv(C))*R - A).max() < 1E-4, 'housetriang failed'
    
    n = 10
    print('rothesstri')
    A = matrix(zeros((n, n)))
    for k in range(n):
        for s in range(k + 2):
            if s < n:
                A[s, k] = random.random()
    b = random.random((n,1))
    xexact = linalg.solve(A, b)
    x = rothesstri(A, b)
    assert abs(x - xexact).max() < 1E-4, 'rothesstri failed'
    
if __name__=='__main__':
    _test_chap1()
    _test_chap2()
    _test_chap3()
    _test_chap4()
