import numpy as np
import sys
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from utils import least_square_loss, least_square_loss_gradient

sys.path.insert(0,'./algorithms_implementation')
import algorithms

# Generate synthetic regression data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.5, random_state=42)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Set the hyperparameters
n_epochs = 50
n_steps = n_epochs*X.shape[0]
learning_rate = 0.001

# Initialize the weights
n_features = X.shape[1]
w_init = np.zeros(n_features)

#Perform SGD
print("SGD:")
w_sgd, obj_sgd= algorithms.sgd(X, y, w_init, learning_rate, n_epochs, least_square_loss, least_square_loss_gradient)
print("SGD weights:", w_sgd)

# # Perform SAGA
print("\nSAGA:")
w_saga, obj_saga = algorithms.saga(X, y, w_init, learning_rate, n_steps, least_square_loss, least_square_loss_gradient)
print("SAGA weights:", w_saga)

# # Perform q_SAGA
print("\nSAGA:")
q = 2
w_q_saga, obj_q_saga = algorithms.saga(X, y, w_init, learning_rate, n_steps, least_square_loss, least_square_loss_gradient, q)
print("SAGA weights:", w_saga)



# Plotting the results
plt.semilogy(range(len(obj_sgd)),obj_sgd , label='SGD')
plt.semilogy(range(len(obj_saga)), obj_saga, label='SAGA')

plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()

plt.show()
