import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Generate a more complex dataset with noise
np.random.seed(42)
X = np.arange(1, 101).reshape(-1, 1)  # Feature: 1 to 100
y = 3.5 * X.squeeze() + np.random.normal(0, 25, size=X.shape[0])  # Linear relationship with noise

# Create a DataFrame for visualization and data manipulation
df = pd.DataFrame({'Feature': X.squeeze(), 'Target': y})

# Visualize the data distribution
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Feature', y='Target', data=df, color='blue', label='Data Points')
plt.title('Scatter Plot of Feature vs Target')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.show()

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate model performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the performance metrics
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Plot the regression line with data points
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Feature', y='Target', data=df, color='blue', label='Data Points')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
plt.title('Linear Regression with Feature vs Target')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.show()

# Visualize residuals
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, color='purple')
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()
