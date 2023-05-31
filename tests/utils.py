import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logistic_loss(X, y_true, w):
    # Compute the linear combination of features and weights
    z = np.dot(X, w)

    # Compute the predicted probabilities
    y_pred = sigmoid(z)

    # Compute the logistic loss
    loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    return loss


def logistic_loss_gradient(X, y_true, w, point_wise = False):
    # Compute the linear combination of features and weights
    z = np.dot(X, w)

    # Compute the predicted probabilities
    y_pred = sigmoid(z)

    if not point_wise:    
        # Compute the gradient of the logistic loss
        gradient = np.dot(X.T, (y_pred - y_true))
    else:
        # Compute the gradient of logistic loss for every individual data point, return matrix whose rows are the gradients
        gradient = X * (y_pred - y_true)

    return gradient


def least_square_loss(X, y, w):
    # Compute the least square loss
    loss = np.sum((y - X @ w) ** 2)
    return loss


def least_square_loss_gradient(X, y, w, point_wise = False):
    if not point_wise:
        # Compute the gradient of the least square loss
        gradient = 2 * X.T @ (X @ w - y)
    else:
        # Compute the gradient of least square loss for every individual data point, return matrix whose rows are the gradients
        gradient = 2 * X * (X @ w - y)

    return gradient