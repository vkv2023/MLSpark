from sklearn.linear_model import Ridge, Lasso
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25])

# High degree polynomial (overfitting)
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

print("Training score:", model.score(X_poly, y))

# Regularization = adding a penalty term to the loss function to prevent overfitting.
# It helps to reduce the complexity of the model and improve generalization to new data.

# Using ridge method(): L2 Regularization (Ridge) - Shirnk weights towards zero but not exactly zero.

model = Ridge(alpha=1.0)  # alpha = regularization strength, increasing alpha, reduces overfitting
model.fit(X_poly, y)

print("\nTraining score with regularization, Ridge:", model.score(X_poly, y))

# Using Lasso (): # L1 Regularization (Lasso) : Shrinks some coefficients to exactly zero, effectively performing feature selection.

model = Lasso(alpha=0.1) # alpha = regularization strength, increasing alpha, reduces overfitting amd maked curve smoother
model.fit(X_poly, y)

print("\nTraining score with regularization, Lasso:", model.score(X_poly, y))