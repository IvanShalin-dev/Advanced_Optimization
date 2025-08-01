import numpy as np

def rastrigin(X):
    """
    Rastrigin function for n-dimensional input.
    Global minimum at X = 0 where f(X) = 0
    """
    A = 10
    X = np.array(X)
    return A * len(X) + np.sum(X**2 - A * np.cos(2 * np.pi * X))