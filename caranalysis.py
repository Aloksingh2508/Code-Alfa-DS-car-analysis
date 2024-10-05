# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'C:\\Users\\Mr.Alok\\Desktop\\code alfa\\car data.csv'
 
car_df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(car_df.head())

# Step 1: Data Preprocessing

# Check for missing values
print("Missing values:\n", car_df.isnull().sum())

# Handling categorical data using Label Encoding (for simplicity)
le = LabelEncoder()
for column in car_df.select_dtypes(include=['object']).columns:
    car_df[column] = le.fit_transform(car_df[column])

# Step 2: Define Features and Target
X = car_df.drop(columns=['Selling_Price'])  # Features (drop target)
y = car_df['Selling_Price']  # Target (car price)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a regression model
# Using RandomForestRegressor for better accuracy in price prediction
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

# Optional: Feature Importance
import matplotlib.pyplot as plt
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances in Car Price Prediction")
plt.show()