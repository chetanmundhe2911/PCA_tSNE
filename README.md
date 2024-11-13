# Model Evaluation with Outliers: Best Metrics for Robust Assessment

## Overview
This repository discusses model evaluation in the presence of outliers and how to select the best performing model. We focus on adjusting our evaluation metrics to handle the impact of outliers, which can severely distort certain metrics like R², MSE, and RMSE. The goal is to evaluate models using more robust metrics that provide a realistic picture of model performance when outliers are present.

## Key Topics

### Outliers and Their Impact
Outliers can distort many commonly used performance metrics in machine learning models. Some metrics are more sensitive to extreme values, while others are more robust.

### Evaluation Metrics
Different metrics behave differently when outliers are present. The following metrics were considered:

- **R² (R-squared):** Sensitive to outliers, typically overestimates model performance.
- **Adjusted R²:** Adjusts R² for model complexity, but still impacted by outliers.
- **MAE (Mean Absolute Error):** Best for outliers — robust to extreme values.
- **MSE (Mean Squared Error):** Highly sensitive to outliers.
- **RMSE (Root Mean Squared Error):** Also sensitive to outliers due to the squaring of errors.
- **MAPE (Mean Absolute Percentage Error):** Sensitive to outliers, especially for values near zero.

### Best Metric for Outliers
MAE is the most robust metric when dealing with outliers. It gives a direct, unweighted measure of the model's average error magnitude and is less influenced by large errors compared to MSE or RMSE. Adjusted R² may still be used for understanding model variance, but should be secondary to MAE when outliers are a concern.

## Implementation in Python
The process includes applying PCA (Principal Component Analysis) to reduce dimensionality and evaluating various models with hyperparameter tuning. Evaluation metrics (including MAE, MSE, RMSE, R², Adjusted R², and MAPE) are computed for each model, with the results saved to a CSV for further analysis.

How to Choose the Optimal Number of Components
You can use explained variance ratio to find the ideal number of components that captures, say, 90-95% of the variance in the data.
