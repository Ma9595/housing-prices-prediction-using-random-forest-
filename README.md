# housing-prices-prediction-using-random-forest-
House Price Prediction
 
 
 Summary
Built a machine learning model to predict house prices based on features like size, location proximity, age, and more. After hyperparameter tuning, the model achieved a validation RMSE of ~2104 and R² of ~0.70, indicating strong performance relative to the average house price of ~95,700.



📈 Objective
To accurately estimate house sale prices using a regression model trained on numerical property attributes such as:

bedroom_count: Number of bedrooms

net_sqm: Net area in square meters

center_distance: Distance from city center (m)

metro_distance: Distance to nearest metro station (m)

floor: Floor level of the property

age: Age of the building



 Machine Learning Approach
Model Used: Tuned RandomForestRegressor
Tuning Method: GridSearchCV with cross-validation
Best Parameters:

    'max_depth': 20,
    'min_samples_leaf': 2,
    'min_samples_split': 5,
    'n_estimators': 100




Final Performance
Metric	Value
RMSE	~2104
R² (Validation)	~0.70
Average Price	~95,700


Tools & Libraries
Python
Pandas, NumPy
Scikit-learn
Matplotlib, Seaborn

📂 Files in This Repository
main.py — Cleaned, tuned, and trained the regression model

house.csv — Input dataset (4308 rows × 7 columns)

README.md — This project summary

