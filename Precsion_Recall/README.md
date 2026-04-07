Scenario (RCM – Denial Prediction Model)

Your AI model predicts whether a claim will be Denied (1) or Approved (0).

Actual vs Predicted:
Claim	Actual	Predicted
C1	Denied (1)	Denied (1) 
C2	Denied (1)	Approved (0) 
C3	Approved (0)	Denied (1) 
C4	Denied (1)	Denied (1) 
C5	Approved (0)	Approved (0) 
C6	Denied (1)	Denied (1) 

Step 1: Confusion Matrix
|                     | Predicted Denied | Predicted Approved |
| ------------------- | ---------------- | ------------------ |
| **Actual Denied**   | TP = 3           | FN = 1             |
| **Actual Approved** | FP = 1           | TN = 1             |

Step 2: Precision - “Out of all claims predicted as denied, how many were actually denied?”
Precision = TP / (TP + FP) = 3 / (3 + 1) = 0.75
75% of flagged claims were correct

Step 3: Recall - “Out of all actually denied claims, how many were correctly identified?”
Recall = TP / (TP + FN) = 3 / (3 + 1) = 0.75        
75% of denied claims were identified

step 4: F1 Score - “Harmonic mean of Precision and Recall”
F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
F1 Score = 2 * (0.75 * 0.75) / (0.75 + 0.75) = 0.75
Balanced performance between precision and recall

Business Interpretation (VERY IMPORTANT)    
- Precision (75%): 25% of claims are false alarms
- Recall (75%): 25% of all denied claims are missed, revenue loss 
- F1 Score (75%): The balanced F1 score suggests that the model has a good trade-off between precision and recall, but there is room for improvement in both metrics to reduce false positives and false negatives.
