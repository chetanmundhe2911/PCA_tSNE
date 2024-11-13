import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import time  # To track the time taken for each model
import joblib  # For saving and loading models
from sklearn.model_selection import GridSearchCV

# Assuming train_df is already loaded
train_df = pd.read_csv("/home/ec2-user/python/modeltraining/project/data/raw/train.csv")
test_df = pd.read_csv("/home/ec2-user/python/modeltraining/project/data/raw/test.csv")
print("Train shape : ", train_df.shape)
print("Test shape : ", test_df.shape)

# Step 1: Label Encoding for categorical features
for f in ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[f].values)) 
    train_df[f] = lbl.transform(list(train_df[f].values))

# Step 2: Prepare your features and target
train_y = train_df['y'].values
train_X = train_df.drop(["ID", "y"], axis=1)

# Step 3: Standardize the features (important for PCA)
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)

# Step 4: Define the variance thresholds and create the models to compare
variance_thresholds = [0.90]  # Variance levels to check

# Defining models with hyperparameters
models = {
    'XGBoost': xgb.XGBRegressor(eta=0.05, max_depth=6, subsample=0.7, colsample_bytree=0.7, objective='reg:squarederror'),
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf', C=1, epsilon=0.1)
}

# Define hyperparameter grids for GridSearchCV
param_grids = {
    'XGBoost': {
        'eta': [0.01, 0.05],
        'max_depth': [4, 6],
        'subsample': [0.6, 0.7],
        'colsample_bytree': [0.6, 0.7]
    },
    'Random Forest': {
        'n_estimators': [50, 100],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'SVR': {
        'C': [0.1, 1],
        'epsilon': [0.01, 0.1],
        'kernel': ['rbf', 'linear']
    }
}

# Function to perform GridSearchCV and return the best model
def tune_hyperparameters(model, param_grid, train_X, train_y):
    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(train_X, train_y)
    best_model = grid_search.best_estimator_
    return best_model, grid_search.best_params_

# Function to train and evaluate models
def train_and_evaluate(model, train_X, train_y):
    start_time = time.time()  # Start time tracking
    model.fit(train_X, train_y)  # Train the model
    training_time = time.time() - start_time  # Calculate the training time
    
    # Make predictions
    preds = model.predict(train_X)
    
    # Calculate evaluation metrics
    r2 = r2_score(train_y, preds)
    mae = mean_absolute_error(train_y, preds)
    mse = mean_squared_error(train_y, preds)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(train_y, preds)
    
    # Calculate Adjusted R2
    n = len(train_y)  # Number of data points
    p = train_X.shape[1]  # Number of features
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    return r2, adj_r2, mae, mse, rmse, mape, training_time

# Initialize a dictionary to store results
results = []

# Step 5: Evaluate each model at different variance levels (PCA transformation)
for threshold in variance_thresholds:
    pca = PCA(n_components=threshold)  # Apply PCA with specified variance threshold
    train_X_pca = pca.fit_transform(train_X_scaled)
    
    print(f"\nEvaluating models at {threshold * 100}% variance retention (retained {train_X_pca.shape[1]} features)...")
    
    for model_name, model in models.items():
        # Tune the model's hyperparameters using GridSearchCV
        print(f"Tuning {model_name} hyperparameters...")
        if model_name in param_grids:
            best_model, best_params = tune_hyperparameters(model, param_grids[model_name], train_X_pca, train_y)
            print(f"Best parameters for {model_name}: {best_params}")
        else:
            best_model = model  # No tuning for Linear Regression
        
        # Train and evaluate the model on PCA-transformed features
        r2, adj_r2, mae, mse, rmse, mape, training_time = train_and_evaluate(best_model, train_X_pca, train_y)
        
        # Store results
        results.append({
            'Variance Threshold': threshold,
            'Model': model_name,
            'Best Parameters': best_params if model_name in param_grids else 'N/A',
            'R2 Score': r2,
            'Adjusted R2': adj_r2,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'Time Taken (s)': training_time
        })
        
        # Print model performance and time taken
        print(f"{model_name}: R² = {r2:.4f}, Adjusted R² = {adj_r2:.4f}, MAE = {mae:.4f}, MSE = {mse:.4f}, RMSE = {rmse:.4f}, MAPE = {mape:.4f}, Time Taken = {training_time:.4f} seconds")

        # Step 6: Save the trained model to the specified path
        model_path = f"/home/ec2-user/python/modeltraining/pca/models/{model_name}_{threshold}.pkl"
        joblib.dump(best_model, model_path)
        print(f"Model saved to {model_path}")

        # Save PCA transformation for future use on the test set
        pca_path = f"/home/ec2-user/python/modeltraining/pca/pca_{threshold}.pkl"
        joblib.dump(pca, pca_path)
        print(f"PCA transformation saved to {pca_path}")

# Step 7: Display results as a DataFrame
results_df = pd.DataFrame(results)
print("\nModel Performance Results:")
print(results_df)

# Step 8: Save the results to a CSV file
results_csv_path = "/home/ec2-user/python/modeltraining/pca/model_performance_results.csv"
results_df.to_csv(results_csv_path, index=False)  # Save to CSV (no index column)
print(f"\nModel performance results exported to {results_csv_path}")

# Step 9: Visualize the results using a bar plot
import seaborn as sns
plt.figure(figsize=(12, 6))
sns.barplot(x='Variance Threshold', y='R2 Score', hue='Model', data=results_df, ci=None)
plt.title('Model Comparison at Different Variance Thresholds')
plt.ylabel('R-squared')
plt.xlabel('Variance Threshold')
plt.tight_layout()
plt.show()

# ------------------------------
# Loading the models and PCA and making predictions on the test set
# ------------------------------

# Assuming test_df is loaded and preprocessed similarly
# Apply Label Encoding to test set
for f in ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[f].values) + list(test_df[f].values))  # Fit on both train and test
    test_df[f] = lbl.transform(list(test_df[f].values))

# Prepare test features
test_X = test_df.drop(["ID"], axis=1)
test_X_scaled = scaler.transform(test_X)  # Use the same scaler from training

# Load the models and PCA transformation
all_predictions = []

for threshold in variance_thresholds:
    for model_name in models.keys():
        # Load the saved model
        model_path = f"/home/ec2-user/python/modeltraining/pca/models/{model_name}_{threshold}.pkl"
        model = joblib.load(model_path)
        
        # Load the saved PCA transformation
        pca_path = f"/home/ec2-user/python/modeltraining/pca/pca_{threshold}.pkl"
        pca = joblib.load(pca_path)

        # Apply the loaded PCA to test data
        test_X_pca = pca.transform(test_X_scaled)  # Apply the same PCA transformation
        
        # Make predictions
        preds = model.predict(test_X_pca)
        
        # Store the predictions with the corresponding ID and model details
        for i, pred in enumerate(preds):
            all_predictions.append({
                'ID': test_df['ID'].iloc[i],
                'Model': model_name,
                'Variance Threshold': threshold,
                'Prediction': pred
            })

# Convert all predictions to a DataFrame
predictions_df = pd.DataFrame(all_predictions)

# Step 10: Export the predictions to a CSV file
predictions_csv_path = "/home/ec2-user/python/modeltraining/pca/all_model_predictions.csv"
predictions_df.to_csv(predictions_csv_path, index=False)

print(f"\nPredictions exported to {predictions_csv_path}")
