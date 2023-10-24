# Import necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # Independent variable (features)
y = 4 + 3 * X + np.random.randn(100, 1)  # Dependent variable (target)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Make predictions
X_new = np.array([[0], [2]])  # New data points for prediction
y_pred = model.predict(X_new)

# Visualize the data and the regression line
plt.scatter(X, y, label='Data')
plt.plot(X_new, y_pred, color='red', linewidth=3, label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Print the model's parameters
print("Intercept (Theta 0):", model.intercept_[0])
print("Coefficient (Theta 1):", model.coef_[0][0])
