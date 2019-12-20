import numpy as np
from sklearn.model_selection import train_test_split

# +
def load_quadratic_data(num=1500, scale=2, test_size=1/6, sort=True):
    ''' Loads the data for our experiments

    Parameters
    ----------
    num : int, optional, default: 1500
        Number of points to generate
    scale : int, optional, default: 2
        Standard deviation used in our generation function
    test_size : float, optional, default: 1/6
        The ratio of test set points to generate. Must be between 0 and 1 (exclusive)
    sort : boolean, default: True
        Whether the points should be sorted or not

    Returns
    -------
    X : np.array
        np.array of (1-test_size)* num points between -10 and 10
    X_test : np.array
        np.array of the test_size*num points between -10 and 10
    y : np.array
        np.array of (1-test_size)* num points that are a function of x and some noise sigma
    y_test : np.array
        np.array of test_size* num points that are a function of x and some noise sigma

    '''
    X = np.linspace(-10, 10, num)
    y = 0.1*X**2 + np.random.normal(size=num, scale=scale)

    # splitting according to train-test split
    X, X_test, y, y_test = train_test_split(X, y, test_size=test_size)

    if sort:
        inds_x, inds_test = np.argsort(X), np.argsort(X_test)
        X, y = X[inds_x], y[inds_x]
        X_test, y_test = X_test[inds_test], y_test[inds_test]

    return X, X_test, y, y_test

def load_sin_data():
    '''Loads data for epistemic uncertainty experiment.
    
    Returns
    -------
    X : np.array
    X_test : np.array
    y : np.array
    y_test : np.array
    '''
    sigma = 0.4
    f = lambda x: np.sin(0.05*x**2) + np.random.normal(0, sigma, size=len(x))
    X = np.concatenate([np.linspace(-10, -5, 4),
                        np.linspace(-5, 5, 500),
                        np.linspace(5, 10, 4)])

    y = f(X)

    X_test = np.linspace(-10, 10, 200)
    y_test = f(X_test)
    
    return X, y, X_test, y_test
