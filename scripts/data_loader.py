import numpy as np
from sklearn.model_selection import train_test_split

def load_quadratic_data(num=1500, scale=2, test_size=1/6, sort=True):
    X = np.linspace(-10, 10, num)
    y = 0.1*X**2 + np.random.normal(size=num, scale=scale)
    
    # splitting according to train-test split
    X, X_test, y, y_test = train_test_split(X, y, test_size=test_size)
    
    if sort:
        inds_x, inds_test = np.argsort(X), np.argsort(X_test)
        X, y = X[inds_x], y[inds_x]
        X_test, y_test = X_test[inds_test], y_test[inds_test]
    
    return X, X_test, y, y_test