
# Linear Regression Project

This repository contains Python scripts for implementing both basic and advanced Linear Regression models. The project demonstrates how to build and evaluate Linear Regression models and visualize the results with a fitted regression line.

## Contents

- `linear_regression.py`: Basic implementation of Linear Regression using a single feature to predict a target variable.
- `advanced_linear_regression_task.py`: Advanced Linear Regression with detailed evaluation metrics and visualizations for a more complex dataset.

## Project Overview

### Objective

The goal of this project is to build both basic and advanced Linear Regression models that predict a target variable based on one or more features. The models are evaluated for performance using standard metrics like Mean Squared Error (MSE) and R-squared score, with visualizations to display the relationship between the features and the target.

### Dataset

The datasets used for this project consist of at least two columns:
- **Feature(s)**: The independent variable(s) used to predict the target.
- **Target**: The dependent variable, i.e., the value that is being predicted.

### Basic Linear Regression

In the basic Linear Regression model:
- **Input**: One feature.
- **Task**: Predict the target variable using a single feature.
- **Visualization**: Scatter plot with the data points and the regression line.

### Advanced Linear Regression

In the advanced Linear Regression model:
- **Input**: Multiple features.
- **Task**: Predict the target variable using multiple features.
- **Evaluation**: The model is evaluated using multiple metrics, such as:
  - Mean Squared Error (MSE)
  - R-squared score
  - Cross-validation results
- **Visualization**: Regression line, residuals, and feature importance plots.

## Model Implementation

### Basic Linear Regression

In the `linear_regression.py` script:
- **Library Used**: Scikit-learn
- **Steps**:
  1. Load and preprocess the data.
  2. Split the data into training and testing sets.
  3. Train a Linear Regression model.
  4. Evaluate the model using MSE and R-squared score.
  5. Plot the regression line with data points.
  
- **Metrics**:
  - **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted values.
  - **R-squared Score**: Indicates the proportion of the variance in the target variable explained by the feature.

### Advanced Linear Regression

In the `advanced_linear_regression_task.py` script:
- **Library Used**: Scikit-learn
- **Steps**:
  1. Load a more complex dataset with multiple features.
  2. Feature scaling and preprocessing.
  3. Perform train-test split.
  4. Train the Linear Regression model.
  5. Use cross-validation to evaluate the model performance.
  6. Visualize the regression line and residuals.
  
- **Additional Metrics**:
  - **Cross-validation**: Evaluate the model across different folds of the dataset to ensure robustness.
  - **Residual Analysis**: Visualize the residuals to check for homoscedasticity (equal variance).

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/AmithYallati/linear-regression-project.git
    ```

2. Navigate to the project folder:
    ```bash
    cd linear-regression-project
    ```

3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

4. To run the basic Linear Regression:
    ```bash
    python linear_regression.py
    ```

5. To run the advanced Linear Regression:
    ```bash
    python advanced_linear_regression_task.py
    ```

## Results

### Basic Linear Regression

- **Mean Squared Error (MSE)**: `value`
- **R-squared Score**: `value`

### Advanced Linear Regression

- **Mean Squared Error (MSE)**: `value`
- **R-squared Score**: `value`
- **Cross-validation MSE (average)**: `value`
- **Residual Analysis**: See plots in the output for residual patterns.

## Visualizations

### Basic Linear Regression

The scatter plot with the fitted regression line shows the relationship between the feature and the target variable.

![Basic Linear Regression](images/basic_regression.png)

### Advanced Linear Regression

Visualizations include:
- Residual plot to assess model performance.
- Feature importance plot (if applicable).
- Regression line plot with actual data points.

![Advanced Linear Regression](images/advanced_regression.png)
