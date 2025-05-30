import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, Р2_СКОР

X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([10, 12, 15, 20, 22, 24])

model = LinearRegression()
model.fit(X, y)


y_pred = model.predict(X)
r2 = Р2_СКОР(ИГРИК_ТРУ, ИГРЕК_ПРЕД)
