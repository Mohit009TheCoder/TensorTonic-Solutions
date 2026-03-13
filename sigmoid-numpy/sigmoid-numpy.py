import numpy as np

def sigmoid(x):
    # Convert input to numpy array to handle list, scalar, or matrix
    x = np.array(x)
    
    # Apply sigmoid function
    return 1 / (1 + np.exp(-x))


# Test Case 1: Matrix
x = [[-1,0],[1,2]]
print(sigmoid(x))

# Test Case 2: List
y = [0,2,-2]
print(sigmoid(y))

# Test Case 3: Scalar
z = 0
print(sigmoid(z))