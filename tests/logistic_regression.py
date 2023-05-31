import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import logistic_loss, logistic_loss_gradient

sys.path.insert(0, "./algorithms_implementation")
import algorithms

# Generate a synthetic dataset
X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=5, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the logistic regression model
lr_model = LogisticRegression(solver="saga", max_iter=1000)

# Train the model using scikit-learn's implementation
lr_model.fit(X_train, y_train)

# Make predictions on the test set using scikit-learn's implementation
lr_preds = lr_model.predict(X_test)

# Calculate accuracy using scikit-learn's implementation
lr_accuracy = accuracy_score(y_test, lr_preds)

# Initialize the initial weight vector and learning rate for custom SAGA and SGD algorithms
w_init = np.zeros(X.shape[1])
learning_rate = 0.001
n_epochs = 100
# each epoch SGD algorithm iterates on all the data points; to have a comparable number of step with SAGA
# we need to multiply the number of epochs for the number of data points
n_steps = X_train.shape[0] * n_epochs

# Train the model using the SAGA algorithm
print("\nSAGA:")
w_saga, obj_saga = algorithms.saga(
    X_train,
    y_train,
    w_init,
    learning_rate,
    n_steps,
    logistic_loss,
    logistic_loss_gradient,
)
print("SAGA weights:", w_saga)


# Make predictions on the test set using the trained SAGA model
saga_preds = np.sign(X_test.dot(w_saga))

# Calculate accuracy for SAGA algorithm
saga_accuracy = accuracy_score(y_test, saga_preds)

# Train the model using the custom SGD algorithm
print("\nSGD:")
w_sgd, obj_sgd = algorithms.sgd(
    X_train,
    y_train,
    w_init,
    learning_rate,
    n_epochs,
    logistic_loss,
    logistic_loss_gradient,
)
print("SGD weights:", w_sgd)

# Make predictions on the test set using the trained SGD model
sgd_preds = np.sign(X_test.dot(w_sgd))

# Calculate accuracy for SGD algorithm
sgd_accuracy = accuracy_score(y_test, sgd_preds)

# Perform q_SAGA
print("\nQ-SAGA:")
# q is the number of updates in the memory table
q = 5
w_q_saga, obj_q_saga = algorithms.q_saga(
    X,
    y,
    w_init,
    learning_rate,
    n_steps,
    logistic_loss,
    logistic_loss_gradient, q)
print("Q-SAGA weights:", w_q_saga)

# Make predictions on the test set using the trained SGD model
q_saga_preds = np.sign(X_test.dot(w_q_saga))

# Calculate accuracy for SGD algorithm
q_saga_accuracy = accuracy_score(y_test, q_saga_preds)

# Perform q_SAGA
print("\nBATCH Q-SAGA:")
# q is both the number of sample to be used and the number of updates in the memory table
q = 5
w_batch_q_saga, obj_batch_q_saga = algorithms.batch_q_saga(
    X,
    y,
    w_init,
    learning_rate,
    n_steps,
    logistic_loss,
    logistic_loss_gradient, q)
print("BATCH Q-SAGA weights:", w_batch_q_saga)

# Make predictions on the test set using the trained SGD model
batch_q_saga_preds = np.sign(X_test.dot(w_batch_q_saga))

# Calculate accuracy for SGD algorithm
batch_q_saga_accuracy = accuracy_score(y_test, batch_q_saga_preds)



# Print the accuracies
print("Logistic Regression Accuracy: {:.4f}".format(lr_accuracy))
print("SAGA Accuracy: {:.4f}".format(saga_accuracy))
print("SGD Accuracy: {:.4f}".format(sgd_accuracy))
print("Q-SAGA Accuracy: {:.4f}".format(q_saga_accuracy))
print("BATCH Q-SAGA Accuracy: {:.4f}".format(batch_q_saga_accuracy))

# Plotting the results
plt.plot(range(len(obj_sgd)), obj_sgd, label="SGD")
plt.plot(range(len(obj_saga)), obj_saga, label="SAGA")
plt.plot(range(len(obj_q_saga)), obj_q_saga, label="Q-SAGA")
plt.plot(range(len(obj_batch_q_saga)), obj_batch_q_saga, label="BATCH Q-SAGA")

plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()

plt.show()