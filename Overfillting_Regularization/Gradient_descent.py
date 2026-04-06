import numpy as np
import matplotlib.pyplot as plt

"""
Loss keeps decreasing 
m and b improve gradually
Model becomes more accurate
"""

"""
A ball rolling downhill

Loss = height
Gradient = slope
Learning rate = step size
"""

# Data: Study hours vs score
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([50, 55, 65, 70, 80], dtype=float)

# Normalize X (important)
X = (X - X.mean()) / X.std()

# Initialize parameters
m = 0.0
b = 0.0

learning_rate = 0.1
epochs = 50
n = len(X)

losses = []

# Gradient Descent
for epoch in range(epochs):
    y_pred = m * X + b

    # Loss (Mean Squared Error)
    loss = np.mean((y - y_pred) ** 2)
    losses.append(loss)

    # Gradients
    dm = (-2 / n) * np.sum(X * (y - y_pred))
    db = (-2 / n) * np.sum(y - y_pred)

    # Update parameters
    m = m - learning_rate * dm
    b = b - learning_rate * db

    print(f"Epoch {epoch}: Loss={loss:.2f}, m={m:.2f}, b={b:.2f}")

# Plot loss
plt.plot(losses)
plt.title("Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

"""
learning rate = 0.1 -> smooth convergence
learning rate = 1.0 -> divergence (loss increases)
"""