import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

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

# 4. Run K-Fold
fold = 1
errors = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    error = mean_squared_error(y_test, y_pred)
    errors.append(error)

    print(f"Fold {fold} MSE:", error)
    fold += 1

# 5. Final result
print("\nAverage MSE:", np.mean(errors))