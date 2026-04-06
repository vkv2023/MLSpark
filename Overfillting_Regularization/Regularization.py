from sklearn.linear_model import Ridge, Lasso


# Regularization = adding a penalty term to the loss function to prevent overfitting.
# It helps to reduce the complexity of the model and improve generalization to new data.

# Using ridge method(): L2 Regularization (Ridge) - Shirnk weights towards zero but not exactly zero.

model = Ridge(alpha=1.0)  # alpha = regularization strength
model.fit(X_poly, y)

print("\nTraining score with regularization, Ridge:", model.score(X_poly, y))

# Using Lasso (): # L1 Regularization (Lasso) : Shrinks some coefficients to exactly zero, effectively performing feature selection.

model = Lasso(alpha=0.1)
model.fit(X_poly, y)

print("\nTraining score with regularization, Lasso:", model.score(X_poly, y))