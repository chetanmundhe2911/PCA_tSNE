import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import joblib
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Configure logging
logging.basicConfig(
    filename='/home/ec2-user/python/modeltraining/pca/training.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load data
train_df = pd.read_csv("/home/ec2-user/python/modeltraining/project/data/raw/train.csv")
test_df = pd.read_csv("/home/ec2-user/python/modeltraining/project/data/raw/test.csv")
logging.info(f"Train shape : {train_df.shape}")
logging.info(f"Test shape : {test_df.shape}")

# Step 1: Label Encoding for categorical features on training data
label_encoders = {}
categorical_features = ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]

for f in categorical_features:
    lbl = LabelEncoder()
    lbl.fit(list(train_df[f].astype(str).values) + ['Unknown'])  # Include 'Unknown'
    train_df[f] = lbl.transform(train_df[f].astype(str))
    label_encoders[f] = lbl
    logging.info(f"Encoded feature {f} with {len(lbl.classes_)} classes (including 'Unknown').")

# Step 2: Prepare features and target
train_y = train_df['y'].values
train_X = train_df.drop(["ID", "y"], axis=1)

# Step 3: Standardize the features
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)
logging.info("Features standardized.")

# Step 4: Define variance thresholds and models
variance_thresholds = [0.90]

models = {
    'XGBoost': xgb.XGBRegressor(eta=0.05, max_depth=6, subsample=0.7, colsample_bytree=0.7, objective='reg:squarederror'),
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf', C=1, epsilon=0.1)
}

param_grids = {
    'XGBoost': {
        'eta': [0.01, 0.05],
        'max_depth': [4, 6],
        'subsample': [0.6, 0.7],
        'colsample_bytree': [0.6, 0.7]
    },
    'RandomForest': {
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

def tune_hyperparameters(model, param_grid, train_X, train_y):
    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(train_X, train_y)
    return grid_search.best_estimator_, grid_search.best_params_

def train_and_evaluate(model, train_X, train_y):
    start_time = time.time()
    model.fit(train_X, train_y)
    training_time = time.time() - start_time
    
    # Validation: Check if the model has been fitted
    if hasattr(model, 'coef_') or hasattr(model, 'feature_importances_'):
        logging.info(f"{model.__class__.__name__} has been fitted.")
    else:
        logging.warning(f"{model.__class__.__name__} might not be fitted correctly.")
    
    preds = model.predict(train_X)
    r2 = r2_score(train_y, preds)
    mae = mean_absolute_error(train_y, preds)
    mse = mean_squared_error(train_y, preds)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(train_y, preds)
    
    n = len(train_y)
    p = train_X.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    return r2, adj_r2, mae, mse, rmse, mape, training_time

results = []

for threshold in variance_thresholds:
    pca = PCA(n_components=threshold)
    train_X_pca = pca.fit_transform(train_X_scaled)
    logging.info(f"PCA applied with {threshold*100}% variance retention. {train_X_pca.shape[1]} components retained.")
    
    for model_name, model in models.items():
        logging.info(f"Processing model: {model_name}")
        
        if model_name in param_grids:
            logging.info(f"Tuning hyperparameters for {model_name}...")
            best_model, best_params = tune_hyperparameters(model, param_grids[model_name], train_X_pca, train_y)
            logging.info(f"Best parameters for {model_name}: {best_params}")
        else:
            best_model = model
            best_params = 'N/A'
        
        # Train and evaluate
        r2, adj_r2, mae, mse, rmse, mape, training_time = train_and_evaluate(best_model, train_X_pca, train_y)
        
        # Store results
        results.append({
            'Variance Threshold': threshold,
            'Model': model_name,
            'Best Parameters': best_params,
            'R2 Score': r2,
            'Adjusted R2': adj_r2,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'Time Taken (s)': training_time
        })
        
        logging.info(f"{model_name}: R²={r2:.4f}, Adjusted R²={adj_r2:.4f}, MAE={mae:.4f}, "
                     f"MSE={mse:.4f}, RMSE={rmse:.4f}, MAPE={mape:.4f}, Time Taken={training_time:.4f} seconds")
        
        # Save the model
        model_path = f"/home/ec2-user/python/modeltraining/pca/models/{model_name}_{threshold}.pkl"
        joblib.dump(best_model, model_path)
        logging.info(f"Model saved to {model_path}")
        
        # Save PCA
        pca_path = f"/home/ec2-user/python/modeltraining/pca/pca_{threshold}.pkl"
        joblib.dump(pca, pca_path)
        logging.info(f"PCA saved to {pca_path}")

# Save results
results_df = pd.DataFrame(results)
results_csv_path = "/home/ec2-user/python/modeltraining/pca/model_performance_results.csv"
results_df.to_csv(results_csv_path, index=False)
logging.info(f"Model performance results exported to {results_csv_path}")

# Visualization
plt.figure(figsize=(12, 6))
sns.barplot(x='Variance Threshold', y='R2 Score', hue='Model', data=results_df, errorbar=None)
plt.title('Model Comparison at Different Variance Thresholds')
plt.ylabel('R-squared')
plt.xlabel('Variance Threshold')
plt.tight_layout()
plt.show()

# ------------------------------
# Loading the models and PCA and making predictions on the test set
# ------------------------------

# Apply Label Encoding to test set using the same encoders
for f in categorical_features:
    lbl = label_encoders.get(f)
    if lbl:
        # Replace unseen labels with 'Unknown'
        test_df[f] = test_df[f].astype(str).apply(lambda x: x if x in lbl.classes_ else 'Unknown')
        
        # Transform the test data
        test_df[f] = lbl.transform(test_df[f])
    else:
        # Assign a default value if the encoder is missing
        test_df[f] = -1
        logging.warning(f"No encoder found for feature {f}. Assigned default value.")
        continue

# Prepare test features
test_X = test_df.drop(["ID"], axis=1)
test_X_scaled = scaler.transform(test_X)

all_predictions = []

for threshold in variance_thresholds:
    for model_name in models.keys():
        model_path = f"/home/ec2-user/python/modeltraining/pca/models/{model_name}_{threshold}.pkl"
        pca_path = f"/home/ec2-user/python/modeltraining/pca/pca_{threshold}.pkl"
        
        if not os.path.exists(model_path):
            logging.warning(f"Model file {model_path} does not exist. Skipping...")
            continue
        
        if not os.path.exists(pca_path):
            logging.warning(f"PCA file {pca_path} does not exist. Skipping...")
            continue
        
        # Load model
        try:
            model = joblib.load(model_path)
            logging.info(f"Loaded {model_name} from {model_path}.")
        except Exception as e:
            logging.error(f"Error loading {model_name} from {model_path}: {e}")
            continue
        
        # Verify model is fitted
        if not (hasattr(model, 'coef_') or hasattr(model, 'feature_importances_') or hasattr(model, 'n_features_in_')):
            logging.error(f"{model_name} loaded from {model_path} is not fitted. Skipping predictions.")
            continue
        
        # Load PCA
        try:
            pca = joblib.load(pca_path)
            logging.info(f"Loaded PCA from {pca_path}.")
        except Exception as e:
            logging.error(f"Error loading PCA from {pca_path}: {e}")
            continue
        
        # Apply PCA to test data
        try:
            test_X_pca = pca.transform(test_X_scaled)
        except Exception as e:
            logging.error(f"Error applying PCA for {model_name}: {e}")
            continue
        
        # Make predictions
        try:
            preds = model.predict(test_X_pca)
            logging.info(f"Made predictions using {model_name}.")
        except Exception as e:
            logging.error(f"Error making predictions with {model_name}: {e}")
            continue
        
        # Store predictions
        for i, pred in enumerate(preds):
            all_predictions.append({
                'ID': test_df['ID'].iloc[i],
                'Model': model_name,
                'Variance Threshold': threshold,
                'Prediction': pred
            })

# Convert predictions to DataFrame
predictions_df = pd.DataFrame(all_predictions)

# Save predictions
predictions_csv_path = "/home/ec2-user/python/modeltraining/pca/all_model_predictions.csv"
predictions_df.to_csv(predictions_csv_path, index=False)
logging.info(f"Predictions exported to {predictions_csv_path}")

print(f"\nPredictions exported to {predictions_csv_path}")
