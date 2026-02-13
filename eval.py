import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


model = joblib.load('regression_model.pkl')
scaler = joblib.load('scaler.pkl')
df = pd.read_csv('Housing.csv')

df['furnishingstatus'] = df['furnishingstatus'].map({'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2})
X = df[['area', 'bedrooms', 'bathrooms', 'parking', 'furnishingstatus']]
y = df['price']

from sklearn.model_selection import train_test_split
_, X_test_raw, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_test_scaled = scaler.transform(X_test_raw)
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- Model Evaluation Metrics ---")
print(f"R-Squared (Accuracy): {r2:.4f}")
print(f"Mean Absolute Error: ${mae:,.2f}")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.title('Actual Prices vs Predicted Prices')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()