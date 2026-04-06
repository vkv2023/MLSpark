Underfitting
    Straight line
    Misses pattern
Good Fit
    Smooth curve
    Best prediction
Overfitting
    Curve bends too much
    Fits noise
Regularization
    Smooths curve
    Controls complexity

===============================
K-Fold {Why K-Fold is important}
===============================
Without K-Fold:
    Result depends on one random split
With K-Fold:
    More stable + reliable evaluation

===============================
| Fold | Train Data  | Test Data   |
| ---- | ----------- | ----------- |
| 1    | 4 points    | 2 points    |
| 2    | different 4 | different 2 |
| 3    | different 4 | different 2 |

Note : Evry data point is used in train and test at least once.