import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

X = 2 * np.random.rand(100,1)
y = 3 * X[:,0] + 4 + np.random.randn(100)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()

model.fit(X_train,y_train)

print(" Weight (slope) W:", model.coef_[0])
print("Biase (intercept) B:", model.intercept_)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test,y_pred)

print("Mean squared error:", mse)
print("R2 score:",r2)   


hours = np.array([[1.5]])
predicted_score = model.predict(hours)
print("Predicted score for 1.5 hours of study:", predicted_score[0])
