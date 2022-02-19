from sklearn.metrics import confusion_matrix

from numba import jit

from numpy import float as np_float
from numpy import mean, var, sqrt

@jit(nopython = True)
def g_score_compiled(c_matrix):

    num_sum = 0.0

    for i in range(c_matrix.shape[0]):
        row_sum = c_matrix[i, :].sum()
        col_sum = c_matrix[:, i].sum()
       
        if row_sum >= col_sum:
            num_sum += c_matrix[i, i] / row_sum

        else:
            num_sum += c_matrix[i, i] / col_sum

    S = num_sum / c_matrix.shape[0]

    return S

#Calculate the Glimmer Score
def g_score(y_true, y_pred):
    """
    Calculates the Glimmer Accuracy.

    Input:
    y_true - A 1-D list or NumPy array of true class labels.
    y_pred - A 1-D list or NumPy array of predicted class labels

    Returns: The Glimmer Accuracy
    """
    c_matrix = confusion_matrix(y_true, y_pred)

    S = g_score_compiled(c_matrix.astype(np_float))

    return S