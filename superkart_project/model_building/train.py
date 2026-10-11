# for data manipulation
import pandas as pd
# for building the preprocessing and modeling pipeline
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# for model serialization and experiment tracking
import joblib
import mlflow

# Configure MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("SuperKart-Sales-Prediction-Experiment")

# Xtrain/Xtest/ytrain/ytest are downloaded from the previous job's artifact
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

numeric_features = [
    'Product_Weight', 
    'Product_Allocated_Area', 
    'Product_MRP', 
    'Store_Establishment_Year'
]
categorical_features = [
    'Product_Id', 
    'Product_Sugar_Content', 
    'Product_Type', 
    'Store_Size',
    'Store_Location_City_Type', 
    'Store_Type'
]

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost regressor model for predicting sales totals
xgb_model = xgb.XGBRegressor(random_state=42)

# Define hyperparameter grid for regression
param_grid = {
    'xgbregressor__n_estimators': [50, 100, 150],
    'xgbregressor__max_depth': [3, 4, 5],
    'xgbregressor__learning_rate': [0.01, 0.05, 0.1],
    'xgbregressor__reg_lambda': [0.1, 0.5, 1.0],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning with GridSearchCV
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, scoring='r2')
    grid_search.fit(Xtrain, ytrain)

    # Log every parameter combination tried during the search as a nested run
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_r2", results["mean_test_score"][i])
            mlflow.log_metric("std_test_r2", results["std_test_score"][i])

    # Log the best hyperparameters in the main run
    mlflow.log_params(grid_search.best_params_)

    # Store the best model
    best_model = grid_search.best_estimator_

    # Make predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Evaluation metrics
    train_r2 = r2_score(ytrain, y_pred_train)
    test_r2 = r2_score(ytest, y_pred_test)
    train_mae = mean_absolute_error(ytrain, y_pred_train)
    test_mae = mean_absolute_error(ytest, y_pred_test)

    # Log metrics
    mlflow.log_metrics({
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_mae": train_mae,
        "test_mae": test_mae
    })

    # Save the model so the Streamlit app can load it directly
    model_dir = "superkart_project/deployment"
    import os
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/best_superkart_sales_model_v1.joblib"
    
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")
