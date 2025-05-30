import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

y_true = np.array([10, 10.5, 11])

y_pred = np.array([10.2, 10.4, 11.1])

errors = y_true - y_pred
abs_errors = np.abs(errors)
squared_errors = errors ** 2
