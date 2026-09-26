import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# Generating simple synthetic data
np.random.seed(42)
X = np.linspace(-1, 1, 200)
y = X**2 + np.random.normal(0, 0.1, size=X.shape)  # A simple quadratic function with noise
# Splitting into training and testing sets
X_train, X_test = X[:150], X[150:]
y_train, y_test = y[:150], y[150:]
X_train
# Building a simple neural network model
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(1,)),  # Hidden layer with 10 neurons
    keras.layers.Dense(1)])  # Output layer
# Compiling the model
model.compile(optimizer='adam', loss='mse')
# Training the model
history = model.fit(X_train, y_train, epochs=100, verbose=0)  # 100 epochs
# Making predictions
X_test = np.array(X_test).reshape(-1, 1)  # Ensure consistent shape
y_pred = model.predict(X_test, verbose=0)  # Disable unnecessary verbosity
# Plotting results
plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='red', label='Testing Data')
plt.plot(X_test, y_pred, color='black', linewidth=2, label='Predictions')
plt.xlabel('Input X')
plt.ylabel('Output y')
plt.title('Simple Neural Network Regression')
plt.legend()
plt.show()
# Explanation of the model
print("\nDeep Learning Model Explanation:")
print("1. We created a simple neural network with one hidden layer (10 neurons) and ReLU activation.")
print("2. The model was trained on synthetic quadratic data.")
print("3. Adam optimizer was used for efficient learning.")
print("4. After training, the model is learning the pattern in the data.")