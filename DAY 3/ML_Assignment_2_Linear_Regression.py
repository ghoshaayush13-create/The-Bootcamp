import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Experimental data
# Input voltage (V)
X = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14]).reshape(-1, 1)

# Measured output voltage (V)
y = np.array([4.80, 5.10, 5.35, 5.65, 5.90,
              6.20, 6.45, 6.70, 7.00, 7.25])

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict output voltage
y_pred = model.predict(X)

# Display the regression equation
slope = model.coef_[0]
intercept = model.intercept_

print("Linear Regression Equation:")
print(f"Output Voltage = {slope:.3f} × Input Voltage + {intercept:.3f}")

# Model evaluation
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"\nMean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Predict output voltage for a new input voltage
new_input = np.array([[15]])
predicted_output = model.predict(new_input)

print(f"\nPredicted output voltage for 15 V input: "
      f"{predicted_output[0]:.2f} V")

# Plot experimental data and regression line
plt.scatter(X, y, color="blue", label="Measured Data")
plt.plot(X, y_pred, color="red", label="Regression Line")

plt.xlabel("Input Voltage (V)")
plt.ylabel("Output Voltage (V)")
plt.title("Linear Regression for Voltage Regulator")
plt.legend()
plt.grid(True)
plt.show()