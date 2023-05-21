import random as rnd
from sys import stdout
import copy

import numpy as np


def saga(X, y, w_init, gamma, n_steps, objective, objective_gradient, batch_size=1):
    n_samples, n_features = X.shape
    obj_vals = []

    # Initialize gradients storage
    # Row i of gradients_memory contains the gradient of the i-th function
    gradients_memory = objective_gradient(X, y, w_init)
    gradient_averages = np.mean(gradients_memory, axis=0)

    # Initialize weights
    w = copy.deepcopy(w_init)

    for step in range(n_steps):
        # choose uniformly random index
        index = rnd.randint(0, n_samples - 1)
        X_sample = np.expand_dims(X[index], axis=0)
        y_sample = y[index]
        # Compute the gradients for the current batch
        gradients_batch = objective_gradient(X_sample, y_sample, w)

        # Update the weights
        w -= gamma * (gradients_batch - gradients_memory[index] + gradient_averages)

        # Update the old gradients with the current gradients
        gradient_averages -= gradients_memory[index] / n_samples
        gradient_averages += gradients_batch / n_samples
        gradients_memory[index] = copy.deepcopy(gradients_batch)

        # Compute the objective function value for the current epoch
        obj_val = objective(X, y, w)
        obj_vals.append(obj_val)
        if step % (n_steps // 100) == 0:
            print(f"Step {step+1}/{n_steps} - Objective Value: {obj_val:.4f}")
    return w, obj_vals


def sgd(X, y, w_init, gamma, n_epochs, objective, objective_gradient, batch_size=1):
    n_samples, n_features = X.shape
    n_batches = n_samples // batch_size
    print(n_batches)
    obj_vals = []

    # Initialize weights
    w = copy.deepcopy(w_init)

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


def q_saga(X, y, w_init, gamma, n_steps, objective, objective_gradient, q, batch_size = 1):
    
    n_samples, n_features = X.shape
    obj_vals = []

    # Initialize gradients storage
    # Row i of gradients_memory contains the gradient of the i-th function
    gradients_memory = objective_gradient(X, y, w_init)
    gradient_averages = np.mean(gradients_memory, axis=0)

    # Initialize weights
    w = copy.deepcopy(w_init)

    for step in range(n_steps):
        
        # choose uniformly random indices to update memory table
        #using rand.sample to have unique values
        indices = rnd.sample(range(0, n_samples), q)
        X_sample_grad = X[indices]
        y_sample_grad = y[indices]
        
        #choose uniformely random index to perform SGD step with noise
        index = np.random.randint(0, len(indices)-1)
        X_sample = np.expand_dims(X[indices[index]], axis = 0)
        y_sample = y[indices[index]]

        # Compute the gradient for the SGD step
        gradients_batch_sgd = objective_gradient(X_sample, y_sample, w)

        #copy by value of w to perform the memory table update
        w_old = copy.deepcopy(w)
        # Update the weights
        w -= gamma * (gradients_batch_sgd - gradients_memory[indices[index]] + gradient_averages)

        # Update the old gradients with the current gradients
        for j, i in enumerate(indices):
            
            gradients_batch_mem = objective_gradient(np.expand_dims(X_sample_grad[j,:], axis = 0), y_sample_grad[j], w_old)
            
            gradient_averages -= gradients_memory[i]/n_samples
            gradient_averages += gradients_batch_mem/n_samples
            gradients_memory[i] = copy.deepcopy(gradients_batch_mem)
            
         # Compute the objective function value for the current epoch
        obj_val = objective(X, y, w)
        obj_vals.append(obj_val)
        if step % (n_steps // 100) == 0:
            print(f"Step {step+1}/{n_steps} - Objective Value: {obj_val:.4f}")
    
    return w, obj_vals


def batch_q_saga(X, y, w_init, gamma, n_steps, objective, objective_gradient, q, batch_size = 1):
    
    n_samples, n_features = X.shape
    obj_vals = []

    # Initialize gradients storage
    # Row i of gradients_memory contains the gradient of the i-th function
    gradients_memory = objective_gradient(X, y, w_init)
    gradient_averages = np.mean(gradients_memory, axis=0)

    # Initialize weights
    w = copy.deepcopy(w_init)

    for step in range(n_steps):
        
        # choose uniformly random indices to update memory table
        #using rand.sample to have unique values
        indices = rnd.sample(range(0, n_samples), q)
        X_sample_grad = X[indices]
        y_sample_grad = y[indices]
        gradients_batch = np.zeros((q, n_features))
        
        # Compute the gradient 
        for row, index in enumerate(indices):
            
            gradients_batch[row,:] = objective_gradient(np.expand_dims(X_sample_grad[row], axis = 0), y_sample_grad[row], w)
            

        #copy by value of w to perform the memory table update
        w_old = copy.deepcopy(w)

        # Update the weights
        w -= gamma * (np.mean(gradients_batch, axis = 0) - np.mean(gradients_memory[indices,:], axis = 0) + gradient_averages)


        # Update the old gradients with the current gradients
        gradient_averages -= np.sum(gradients_memory[indices, :], axis = 0)/n_samples
        gradient_averages += np.sum(gradients_batch, axis = 0) /n_samples
        gradients_memory[indices,:] = copy.deepcopy(gradients_batch)
            

         # Compute the objective function value for the current epoch
        obj_val = objective(X, y, w)
        obj_vals.append(obj_val)
        if step % (n_steps // 100) == 0:
            print(f"Step {step+1}/{n_steps} - Objective Value: {obj_val:.4f}")
    
    return w, obj_vals
