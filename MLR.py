import numpy as np
import matplotlib.pyplot as plt

# User-provided data
X1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
X2 = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
y = np.array([5, 8, 11, 13, 15, 17, 19, 22, 24, 26])

# Concatenate the features and add a column of ones for the intercept
X = np.vstack((X1, X2, np.ones(X1.shape[0]))).T

# Calculate the coefficients using OLS
coefficients = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# Extract the coefficients
coef1, coef2, intercept = coefficients[0], coefficients[1], coefficients[2]

print("Coefficient 1:", coef1)
print("Coefficient 2:", coef2)
print("Intercept:", intercept)

# Create prediction function
def predict(X1, X2):
    return coef1 * X1 + coef2 * X2 + intercept

# Calculate Root Mean Square Error (RMSE)
predictions = predict(X1, X2)
rmse = np.sqrt(np.mean((y - predictions) ** 2))

print("Root Mean Square Error (RMSE):", rmse)

# Create a grid for plotting
x1 = np.linspace(min(X1), max(X1), 100)
x2 = np.linspace(min(X2), max(X2), 100)
X1, X2 = np.meshgrid(x1, x2)
Y = predict(X1, X2)

# Plot the 3D surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1, X2, y, color='red', marker='o', label='Data Points')
ax.plot_surface(X1, X2, Y, color='blue', alpha=0.5, label='Regression Plane')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.legend()
plt.show()
