import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    n_samples, n_features = X.shape

    # Initialize parameters
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(steps):
        # Linear combination
        z = np.dot(X, w) + b
        
        # Prediction
        p = _sigmoid(z)

        # Gradients
        dw = (1/n_samples) * np.dot(X.T, (p - y))
        db = (1/n_samples) * np.sum(p - y)

        # Update parameters
        w -= lr * dw
        b -= lr * db

    return w, b