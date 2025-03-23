import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('house.csv')

# Define features and target (target variable is "price")
X = data.drop("price", axis=1)
y = data["price"]

# Split data into training (80%) and validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up hyperparameter grid for tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Initialize RandomForestRegressor and perform GridSearchCV
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

# Use the best estimator for predictions
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_val)

# Compute evaluation metrics
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

# Print only the method and accuracy results
print("Method: Tuned RandomForestRegressor")
print("Best Parameters:", grid_search.best_params_)
print("Validation RMSE:", rmse)
print("Validation R^2:", r2)
