import numpy as np
import matplotlib.pyplot as plt  # Fixed: Usually imported as plt, not plot
from sklearn.preprocessing import PolynomialFeatures # Fixed syntax
# from mpmath import degree     # Removed: Not needed and will cause errors here
from sklearn.linear_model import LinearRegression

# High degree polynomial = overfitting
# Noise in data can lead to overfitting when using high degree polynomials, as the model tries to fit the noise rather than the underlying pattern. This can result in poor generalization to new data.

# Data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81])

# High degree polynomial regression
poly = PolynomialFeatures(degree=4)

"""
       Fit to data, then transform it.

       Fits transformer to `X` and `y` with optional parameters `fit_params`
       and returns a transformed version of `X`.

       Parameters
       ----------
       X : array-like of shape (n_samples, n_features)
           Input samples.

       y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \
               default=None
           Target values (None for unsupervised transformations).
"""

x_poly = poly.fit_transform(x)
model = LinearRegression()

"""
        Fit linear model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.
            If not provided, then each sample will be given unit weight.
"""

model.fit(x_poly, y)

# Predicting values
x_test = np.array([10, 11, 12]).reshape(-1, 1)
x_test_poly = poly.transform(x_test)

"""
        Predict using the linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
"""
y_pred = model.predict(x_test_poly)

print(f"Training Score: {model.score(x_poly, y)}")
print(f"Predicted values for 10, 11, 12: {y_pred}")

import matplotlib.pyplot as plt

# Generate a smooth line for plotting
x_range = np.linspace(1, 12, 100).reshape(-1, 1)
x_range_poly = poly.transform(x_range)
y_range_pred = model.predict(x_range_poly)

plt.scatter(x, y, color='red', label='Actual Data')
plt.plot(x_range, y_range_pred, color='blue', label='Model Prediction')
plt.legend()
plt.show()