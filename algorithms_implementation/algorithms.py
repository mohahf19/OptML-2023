import numpy as np
import random as rnd
from sys import stdout

def saga(X, y, w_init, gamma, n_steps, objective, objective_gradient, batch_size = 1):
    n_samples, n_features = X.shape
    obj_vals = []
    
    # Initialize gradients storage
    # Row i of gradients_memory contains the gradient of the i-th function
    gradients_memory = 2 * X *np.expand_dims(X @ w_init - y, axis=1)
    gradient_averages = np.mean(gradients_memory, axis=0)

    # Initialize weights
    w = w_init.copy()

    for step in range(n_steps):
        
        # choose uniformly random index
        index = rnd.randint(0, n_samples-1)
        X_sample = np.expand_dims(X[index], axis = 0)
        y_sample = y[index]

        # Compute the gradients for the current batch
        gradients_batch = objective_gradient(X_sample, y_sample, w)

        # Update the weights
        w -= gamma * (gradients_batch - gradients_memory[index] + gradient_averages)

        # Update the old gradients with the current gradients
        gradient_averages -= gradients_memory[index]/n_samples
        gradient_averages += gradients_batch/n_samples
        gradients_memory[index] = gradients_batch.copy()
        
         # Compute the objective function value for the current epoch
        obj_val = objective(X, y, w)
        obj_vals.append(obj_val)
        if step % (n_steps // 100) == 0:
            print(f"Step {step+1}/{n_steps} - Objective Value: {obj_val:.4f}")
    return w, obj_vals



def sgd(X, y, w_init, gamma, n_epochs, objective, objective_gradient, batch_size = 1):
    n_samples, n_features = X.shape
    n_batches = n_samples // batch_size
    print(n_batches)
    obj_vals = []

    # Initialize weights
    w = w_init.copy()

    for epoch in range(n_epochs):
        # Shuffle the data
        permutation = np.random.permutation(n_samples)
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]

        for batch in range(n_batches):
            # Select the batch
            batch_start = batch * batch_size
            batch_end = (batch + 1) * batch_size
            X_batch = X_shuffled[batch_start:batch_end]
            y_batch = y_shuffled[batch_start:batch_end]

            # Compute the gradient for the current batch

            gradient = objective_gradient(X_batch, y_batch, w)

            # Update the weights
            w -= gamma * gradient
            # Compute the objective function value for the current epoch
            obj_val = objective(X, y, w)
            obj_vals.append(obj_val)
        print(f"Epoch {epoch+1}/{n_epochs} - Objective Value: {obj_val:.4f}")

    return w, obj_vals
