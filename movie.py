import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
data = pd.read_csv(r'D:\Programs\ML\Movies revenue\Movies_data.csv', encoding='latin1')

# Define Features (X) and Target (y)
X = data[['budget']]
y = data['revenue']

# Handle missing or invalid values (if any)
X = X.fillna(0)
y = y.fillna(0)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Results
print("Mean Squared Error:", mse)
print("R-Squared Value:", r2)
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Plot: Actual vs Predicted Revenue
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted Revenue")
plt.show()
