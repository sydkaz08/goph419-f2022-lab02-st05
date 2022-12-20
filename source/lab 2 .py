#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

def gauss_iter_solve(A,b,x0 = 'None',tol = 1e-8,alg = 'seidel'):
    """Solves a matrix using an iterative Gauss-Seidel approach.
    
    Parameters
    ----------
    A: array_like
        Coefficient matrix

    b: array_like
        Array of RHS vectors. 

    x0: array_like
        Optional: Array of initial guess(es)
    
    tol: float
        Optional: Error tolerance for the stopping criterion
    
    alg: str
        Optional: Algorithm to be used for solving the system. Can also be jacobi. 
    Returns
    -------
    solved_arr: np.ndarray
        The solution vector. 
    """
    
    #Input sanitization toi check for correct and solvable inputs of this method
    if np.shape(A)[0] != np.shape(A)[1]:
        raise ValueError("Matrix A is not square!")
    
    #if np.shape(b)[1] != np.shape(A)[1]:
        #raise ValueError("RHS is not same length as A!")
    
    if x0 != 'None' and np.shape(x0)[1] != np.shape(A)[1]:
        raise ValueError("x0 not the same length as A!")
    
    #checks flags stripped of case and leading/trailing whitespace
    if alg.lower().strip() != 'jacobi' and alg.lower().strip() != 'seidel':
        raise ValueError("You must use Seidel or Jacobi for the algorithm type!")
    
    if alg.lower().strip() == 'jacobi':
        return jacobi(A,b,x0 = 'None',tol = 1e-8)

    #Creates array of initial guesses starting with all zeroes if one isn't provided
    if x0 == 'None':
        x0 = np.zeros_like(b,dtype=np.double)
    
    k=0
    max_iterations = 1e3
    x=x0
    while max_iterations >= k:
        x_new = np.zeros_like(x,dtype=np.double)
        print("Iteration {0}: {1}".format(k, x))
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1 :], x[i + 1 :])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if np.allclose(x, x_new, rtol=1e-8):
            break
        x = x_new
        k+=1    
    return x

def jacobi(A,b,x0 = 'None',tol = 1e-8):
    x = np.zeros_like(b, dtype=np.double)
    
    T = A - np.diag(np.diagonal(A))
    k=0
    max_iterations = 1e3
    while max_iterations >= k:
        
        x_old  = x.copy()
        
        x[:] = (b - np.dot(T, x)) / np.diagonal(A)
        
        if np.linalg.norm(x - x_old, ord=np.inf) / np.linalg.norm(x, ord=np.inf) < tol:
            break
            
    return x

  
def spline_function(xd,yd,order = 3):
    """Description of function.
    Parameters
    ----------
    xd: array_like
        Array of float data of increasing value

    yd: array_like
        Array of float data of increasing value

    order: int
        Optional: Function returns that order of interpolated polynomial. Linear = 1, Quad = 2, Cube = 3. 
    Returns
    -------
    array
        Interpolated y values.
    """
    if xd.shape != yd.shape:
        raise ValueError("Arrays are not the same length!")

    xd_unique = np.unique(xd)

    if xd.shape != xd_unique.shape:
        raise ValueError("Non-unique (repeated) values in xd array!")

    import numpy as np

def cubic_spline_interpolation(x, y):
    """
    Interpolate the given data points using a cubic spline.

    Args:
        x (array-like): The x-coordinates of the data points.
        y (array-like): The y-coordinates of the data points.

    Returns:
        callable: A function that evaluates the interpolated cubic spline at any point.
    """
    # Create arrays for the coefficients of the cubic polynomials
    n = len(x)
    a = np.copy(y)
    b = np.zeros(n)
    d = np.zeros(n)
    h = np.zeros(n)
    alpha = np.zeros(n)

    # Compute the differences between the x-coordinates
    for i in range(n - 1):
        h[i] = x[i + 1] - x[i]

    # Compute the coefficients of the tridiagonal matrix
    for i in range(1, n - 1):
        alpha[i] = 3 / h[i] * (a[i + 1] - a[i]) - 3 / h[i - 1] * (a[i] - a[i - 1])
    alpha[0] = 0
    alpha[-1] = 0

    # Solve for the coefficients of the cubic polynomials using the tridiagonal matrix algorithm
    c = np.zeros(n + 1)
    l = np.zeros(n)
    mu = np.zeros(n)
    z = np.zeros(n)
    l[0] = 1
    mu[0] = 0
    z[0] = 0

    for i in range(1, n - 1):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
    l[-1] = 1
    z[-1] = 0
    c[-1] = 0

    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    # Define a function that evaluates the interpolated cubic spline at any point
    def f(xx):
        k = np.searchsorted(x, xx) - 1
        return a[k] + b[k] * (xx - x[k]) + c[k] * (xx - x[k])**2 + d[k] * (xx - x[k])**3

    return f

    
    #return y_arr


# In[ ]:




