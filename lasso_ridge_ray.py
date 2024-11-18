import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import ray
import matplotlib.pyplot as plt

# Initialize Ray
ray.init(ignore_reinit_error=True, dashboard_port=8506)

# Print the link to the Ray dashboard
print("You can access the Ray dashboard at: http://localhost:8506/")

# Load data
train = pd.read_csv('/home/centos/python/chetan/train.csv')
test = pd.read_csv('/home/centos/python/chetan/test.csv')

# Drop the 'ID' column as it's not useful
train = train.drop(columns=['ID'])
test = test.drop(columns=['ID'])

# Separate features (X) and target (y)
X = train.drop(columns=['y'])
y = train['y']

# Handle categorical features (using one-hot encoding)
X = pd.get_dummies(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for Lasso and Ridge)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train a basic linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Evaluate the model
y_val_pred = lr_model.predict(X_val_scaled)
print("Linear Regression Performance:")
print("MAE:", mean_absolute_error(y_val, y_val_pred))
print("MSE:", mean_squared_error(y_val, y_val_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_val, y_val_pred)))
print("R²:", r2_score(y_val, y_val_pred))

# Hyperparameter tuning for Lasso and Ridge using GridSearchCV
param_grid = {'alpha': np.logspace(-4, 1, 6)}  # Exploring a smaller range

# Remote function to perform grid search
@ray.remote
def run_grid_search(model, param_grid, X_train_scaled, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


# Lasso Regression (parallelized)
lasso = Lasso(max_iter=10000, alpha=0.01)
lasso_future = run_grid_search.remote(lasso, param_grid, X_train_scaled, y_train)

# Ridge Regression (parallelized)
ridge = Ridge(solver='auto', max_iter=10000, tol=1e-4)
ridge_future = run_grid_search.remote(ridge, param_grid, X_train_scaled, y_train)

# Wait for the results of both grid searches
lasso_best, lasso_best_params, lasso_best_score = ray.get(lasso_future)
ridge_best, ridge_best_params, ridge_best_score = ray.get(ridge_future)

# Evaluate Lasso
y_val_lasso_pred = lasso_best.predict(X_val_scaled)
print("\nLasso Regression Performance:")
print("Best Params:", lasso_best_params)
print("Best CV Score:", lasso_best_score)
print("MAE:", mean_absolute_error(y_val, y_val_lasso_pred))
print("MSE:", mean_squared_error(y_val, y_val_lasso_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_val, y_val_lasso_pred)))
print("R²:", r2_score(y_val, y_val_lasso_pred))

# Evaluate Ridge
y_val_ridge_pred = ridge_best.predict(X_val_scaled)
print("\nRidge Regression Performance:")
print("Best Params:", ridge_best_params)
print("Best CV Score:", ridge_best_score)
print("MAE:", mean_absolute_error(y_val, y_val_ridge_pred))
print("MSE:", mean_squared_error(y_val, y_val_ridge_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_val, y_val_ridge_pred)))
print("R²:", r2_score(y_val, y_val_ridge_pred))

# Optionally, plot the learning curves
# You can plot the error as a function of alpha to visualize the model's performance under different regularization strengths.

# Shut down Ray when done
ray.shutdown()
