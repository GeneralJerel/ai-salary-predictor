"""
Model Training Module
Trains and evaluates multiple ML models for salary prediction.
Uses RandomizedSearchCV for hyperparameter tuning (matching notebook workflow).
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_linear_regression(X_train, y_train):
    """Train Linear Regression model with scaled features."""
    print("Training Linear Regression (with scaled features)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    return model, scaler


def train_random_forest(X_train, y_train, n_iter=50, cv=5, random_state=42):
    """
    Train Random Forest model with RandomizedSearchCV hyperparameter tuning.
    Matches notebook: 50 iterations, 5-fold CV.
    """
    print("Training Random Forest with hyperparameter tuning...")
    print(f"Running RandomizedSearchCV ({n_iter} iterations, {cv}-fold CV)...")

    param_distributions = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [5, 10, 15, 20, 30, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
    }

    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    search = RandomizedSearchCV(
        rf,
        param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring='r2',
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    search.fit(X_train, y_train)

    best_params = search.best_params_
    print(f"\nBest Random Forest Hyperparameters:")
    for param, val in best_params.items():
        print(f"  {param}: {val}")

    return search.best_estimator_, search.best_score_


def train_xgboost(X_train, y_train, n_iter=50, cv=5, random_state=42):
    """
    Train XGBoost model with RandomizedSearchCV hyperparameter tuning.
    Matches notebook: 50 iterations, 5-fold CV.
    """
    print("Training XGBoost with hyperparameter tuning...")
    print(f"Running RandomizedSearchCV ({n_iter} iterations, {cv}-fold CV)...")

    param_distributions = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'reg_alpha': [0, 0.01, 0.1, 1.0],
        'reg_lambda': [0.5, 1.0, 2.0, 5.0],
    }

    xgb = XGBRegressor(random_state=random_state, verbosity=0)
    search = RandomizedSearchCV(
        xgb,
        param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring='r2',
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    search.fit(X_train, y_train)

    best_params = search.best_params_
    print(f"\nBest XGBoost Hyperparameters:")
    for param, val in best_params.items():
        print(f"  {param}: {val}")

    return search.best_estimator_, search.best_score_


def evaluate_model(model, X_test, y_test, model_name, scaler=None):
    """
    Evaluate model performance.
    If scaler is provided, applies it to X_test before prediction.
    Returns: dict with RMSE, MAE, R²
    """
    X_eval = scaler.transform(X_test) if scaler is not None else X_test
    y_pred = model.predict(X_eval)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'predictions': y_pred
    }


def save_model(model, model_name, output_dir='models/trained_models'):
    """Save trained model."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'{model_name}.joblib')
    joblib.dump(model, filepath)
    print(f"Model saved: {filepath}")
    return filepath


def train_and_evaluate(X, y, test_size=0.2, random_state=42):
    """
    Complete training pipeline with hyperparameter tuning.
    Matches notebook workflow: 80/20 split, RandomizedSearchCV tuning.
    """
    print(f"\nDataset shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Train models
    print("\n" + "=" * 50)
    print("MODEL TRAINING (with hyperparameter tuning)")
    print("=" * 50)

    lr_model, lr_scaler = train_linear_regression(X_train, y_train)

    # Cross-validation for Linear Regression
    from sklearn.preprocessing import StandardScaler as SS
    scaler_cv = SS()
    X_train_scaled_cv = scaler_cv.fit_transform(X_train)
    lr_cv_scores = cross_val_score(
        LinearRegression(), X_train_scaled_cv, y_train, cv=5, scoring='r2'
    )
    print(f"Linear Regression CV R² (5-fold): {lr_cv_scores.mean():.4f} ± {lr_cv_scores.std():.4f}")

    rf_model, rf_best_cv = train_random_forest(X_train, y_train)
    xgb_model, xgb_best_cv = train_xgboost(X_train, y_train)

    # Evaluate models
    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    results = []
    results.append(evaluate_model(lr_model, X_test, y_test, 'Linear Regression', scaler=lr_scaler))
    results.append(evaluate_model(rf_model, X_test, y_test, 'Random Forest (Tuned)'))
    results.append(evaluate_model(xgb_model, X_test, y_test, 'XGBoost (Tuned)'))

    # Add CV scores
    results[0]['CV R²'] = lr_cv_scores.mean()
    results[1]['CV R²'] = rf_best_cv
    results[2]['CV R²'] = xgb_best_cv

    # Create results dataframe
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    display_cols = ['Model', 'RMSE', 'MAE', 'R²', 'CV R²']
    print(results_df[display_cols].to_string(index=False))

    # Save models
    print("\n" + "=" * 50)
    print("SAVING MODELS")
    print("=" * 50)
    save_model(lr_model, 'linear_regression_model')
    save_model(rf_model, 'random_forest_model')
    save_model(xgb_model, 'xgboost_model')
    save_model(lr_scaler, 'scaler')

    # Find best model by test R²
    best_idx = results_df['R²'].idxmax()
    best_model_info = results_df.iloc[best_idx]
    print(f"\nBest Model: {best_model_info['Model']} (R² = {best_model_info['R²']:.4f})")

    models_list = [lr_model, rf_model, xgb_model]

    return {
        'models': {
            'lr': lr_model,
            'rf': rf_model,
            'xgb': xgb_model
        },
        'scaler': lr_scaler,
        'results': results_df,
        'test_set': (X_test, y_test),
        'best_model_name': best_model_info['Model'],
        'best_model': models_list[best_idx]
    }


def main(X, y):
    """Main model training pipeline."""
    training_results = train_and_evaluate(X, y)
    return training_results


if __name__ == '__main__':
    # This would be called from the notebook or main script
    pass
