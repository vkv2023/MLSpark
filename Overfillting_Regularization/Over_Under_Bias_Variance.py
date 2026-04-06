import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

# 1. Simple dataset: Study hours vs score
np.random.seed(0)
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([50, 55, 65, 70, 80, 85]) + np.random.normal(0, 3, 6)

# Smooth range for plotting
X_test = np.linspace(1, 6, 100).reshape(-1, 1)

# Function to plot
def plot_model(degree, model, title):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    X_test_poly = poly.transform(X_test)

    model.fit(X_poly, y)
    y_pred = model.predict(X_test_poly)

    plt.scatter(X, y, color='black', label="Data")
    plt.plot(X_test, y_pred, label=f"{title} (deg={degree})")
    plt.title(title)
    plt.xlabel("Study Hours")
    plt.ylabel("Score")
    plt.legend()
    plt.show()

# 2. Underfitting (High Bias)
plot_model(1, LinearRegression(), "Underfitting (High Bias)")

# 3. Good Fit
plot_model(2, LinearRegression(), "Good Fit")

# 4. Overfitting (High Variance)
plot_model(5, LinearRegression(), "Overfitting (High Variance)")

# 5. Regularization (Fix Overfitting)
plot_model(5, Ridge(alpha=10), "Regularized (Ridge)")