from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


# 1. Dataset (same as before)
np.random.seed(0)
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([50, 55, 65, 70, 80, 85]) + np.random.normal(0, 3, 6)

# 2. K-Fold setup
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# 3. Model (polynomial regression)
model = Pipeline([
    ("poly", PolynomialFeatures(degree=3)),
    ("lr", LinearRegression())
])

scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')

print("MSE per fold:", -scores)
print("Average MSE:", -scores.mean())