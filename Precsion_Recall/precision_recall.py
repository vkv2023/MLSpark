from sklearn.metrics import precision_score, recall_score, f1_score

# Actual (1 = denied, 0 = approved)
y_true = [1, 1, 0, 1, 0, 1]

# Predicted
y_pred = [1, 0, 1, 1, 0, 1]

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)