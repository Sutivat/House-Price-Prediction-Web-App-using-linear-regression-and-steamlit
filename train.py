import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import joblib
import numpy as np

df = pd.read_csv('Housing.csv')

ranking = [['unfurnished', 'semi-furnished', 'furnished']]

encoder = OrdinalEncoder(categories=ranking)
sizes = np.array(df['furnishingstatus']).reshape(-1,1)

encoded = encoder.fit_transform(sizes) 
df['furnishingstatus'] = encoded

X = df[['area', 'bedrooms', 'bathrooms', 'parking', 'furnishingstatus']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, 'regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')