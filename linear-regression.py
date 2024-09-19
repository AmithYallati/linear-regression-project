import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Creating a simple dataset
# Feature: Years of Experience, Target: Salary
data = {
    'YearsExperience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [35000, 37000, 39000, 41000, 43000, 45000, 47000, 49000, 51000, 53000]
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Split the data into features (X) and target (y)
X = df[['YearsExperience']]  # Feature
y = df['Salary']  # Target

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions using the model
y_pred = model.predict(X)

# Calculate performance metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Plotting the regression line along with the data points
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression: Years of Experience vs. Salary')
plt.legend()
plt.grid(True)
plt.show()

# Outputting the performance metrics
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
