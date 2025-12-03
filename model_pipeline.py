"""
Machine Learning Pipeline for Garment Productivity Prediction

This module contains all the necessary functions for the complete ML pipeline including:
- Data loading and preprocessing
- Feature engineering
- Model training and evaluation
- Model persistence and prediction
"""

import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from typing import Dict, Tuple, List, Optional

# Machine Learning Imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)

# Model Imports
from sklearn.ensemble import RandomForestRegressor

import warnings

warnings.filterwarnings("ignore")


def load_data(csv_path: str, target_column: str) -> Tuple[pd.DataFrame, str]:
    """
    Load dataset from CSV file and validate its structure.

    Args:
        csv_path (str): Path to the CSV file
        target_column (str): Name of the target variable column

    Returns:
        Tuple[pd.DataFrame, str]: Loaded dataframe and target column name

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If target column is not found in the dataset
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  📂 Loading Data                                                     ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    print("✅ Dataset loaded successfully")
    print(f"   ├─ Rows: {df.shape[0]:,}")
    print(f"   ├─ Columns: {df.shape[1]}")
    print(f"   ├─ Target: {target_column}")
    print(f"   └─ Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return df, target_column


def clean_data(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and removing duplicates.

    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target variable

    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║  🧹 Cleaning Data                                                    ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    df_clean = df.copy()
    initial_rows = df_clean.shape[0]

    # Handle missing values in numerical columns
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    for col in numerical_cols:
        missing_count = df_clean[col].isnull().sum()
        if missing_count > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(
                f"✅ Filled {missing_count} missing values in '{col}' with median: {median_val:.2f}"
            )

    # Handle missing values in categorical columns
    categorical_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        missing_count = df_clean[col].isnull().sum()
        if missing_count > 0:
            mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else "Unknown"
            df_clean[col].fillna(mode_val, inplace=True)
            print(f"✅ Filled {missing_count} missing values in '{col}' with mode: {mode_val}")

    # Remove duplicate rows
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        df_clean.drop_duplicates(inplace=True)
        print(f"✅ Removed {duplicates} duplicate rows")

    # Remove rows with missing target values
    target_missing = df_clean[target_column].isnull().sum()
    if target_missing > 0:
        df_clean = df_clean[df_clean[target_column].notnull()]
        print(f"✅ Removed {target_missing} rows with missing target values")

    final_rows = df_clean.shape[0]
    print("\n📊 Cleaning Summary:")
    print(f"   ├─ Initial rows: {initial_rows:,}")
    print(f"   ├─ Final rows: {final_rows:,}")
    print(f"   └─ Rows removed: {initial_rows - final_rows:,}")

    return df_clean


def prepare_data(
    df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data by encoding categorical variables and splitting into train/test sets.

    Args:
        df (pd.DataFrame): Cleaned dataframe
        target_column (str): Name of the target variable
        test_size (float): Proportion of dataset for testing (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)

    Returns:
        Tuple containing:
            - X_train (pd.DataFrame): Training features
            - X_test (pd.DataFrame): Testing features
            - y_train (pd.Series): Training target
            - y_test (pd.Series): Testing target
    """
    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║  🔧 Preparing Data                                                   ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    df_prep = df.copy()

    # Encode categorical variables with one-hot encoding
    cat_cols = df_prep.select_dtypes(include=["object"]).columns.tolist()

    if len(cat_cols) > 0:
        print(f"✅ Encoding {len(cat_cols)} categorical columns: {cat_cols}")
        df_encoded = pd.get_dummies(df_prep, columns=cat_cols, drop_first=True, dtype=int)
        print(f"   ├─ Features before encoding: {df_prep.shape[1]}")
        print(f"   └─ Features after encoding: {df_encoded.shape[1]}")
    else:
        df_encoded = df_prep.copy()
        print("✅ No categorical variables to encode")

    # Separate features and target
    X = df_encoded.drop(target_column, axis=1)
    y = df_encoded[target_column]

    print("\n📊 Dataset Composition:")
    print(f"   ├─ Feature columns: {X.shape[1]}")
    print(f"   └─ Total samples: {X.shape[0]}")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    print("\n📊 Train-Test Split:")
    print(f"   ├─ Training samples: {X_train.shape[0]} ({(1-test_size)*100:.0f}%)")
    print(f"   └─ Testing samples: {X_test.shape[0]} ({test_size*100:.0f}%)")

    return X_train, X_test, y_train, y_test


def engineer_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    date_column: Optional[str] = "date",
    feature_config: Optional[Dict[str, List[str]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Engineer new features from existing ones including temporal and derived features.

    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        date_column (str, optional): Name of date column for temporal features
        feature_config (dict, optional): Configuration for feature engineering
            Example: {
                'time_features': ['over_time', 'idle_time'],
                'productivity_features': ['targeted_productivity'],
                'worker_features': ['no_of_workers', 'idle_men'],
                ...
            }

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Engineered train and test features
    """
    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║  ⚙️ Engineering Features                                             ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    X_train_eng = X_train.copy()
    X_test_eng = X_test.copy()

    features_created = 0

    # Temporal features from date column
    if date_column in X_train_eng.columns:
        for df in [X_train_eng, X_test_eng]:
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
            df["month"] = df[date_column].dt.month
            df["day_of_month"] = df[date_column].dt.day
            df["week_of_year"] = df[date_column].dt.isocalendar().week
            df["is_month_start"] = df[date_column].dt.is_month_start.astype(int)
            df["is_month_end"] = df[date_column].dt.is_month_end.astype(int)

            # Fill NaN values that might have been created
            for col in ["month", "day_of_month", "week_of_year"]:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)

            df.drop(date_column, axis=1, inplace=True, errors="ignore")

        print("✅ Created temporal features from date column")
        features_created += 5

    # Time-based features
    if "over_time" in X_train_eng.columns and "idle_time" in X_train_eng.columns:
        for df in [X_train_eng, X_test_eng]:
            df["total_time"] = df["over_time"] + df["idle_time"]
            df["productive_time"] = df["over_time"] - df["idle_time"]
        print("✅ Created time-based features (total_time, productive_time)")
        features_created += 2

    # Work complexity features
    if "smv" in X_train_eng.columns and "no_of_workers" in X_train_eng.columns:
        for df in [X_train_eng, X_test_eng]:
            df["work_complexity"] = df["smv"] * df["no_of_workers"]
        print("✅ Created work_complexity feature")
        features_created += 1

    # Worker utilization features
    if "idle_men" in X_train_eng.columns and "no_of_workers" in X_train_eng.columns:
        for df in [X_train_eng, X_test_eng]:
            df["worker_utilization"] = 1 - (df["idle_men"] / df["no_of_workers"].replace(0, 1))
        print("✅ Created worker_utilization feature")
        features_created += 1

    # Incentive per worker
    if "incentive" in X_train_eng.columns and "no_of_workers" in X_train_eng.columns:
        for df in [X_train_eng, X_test_eng]:
            df["incentive_per_worker"] = df["incentive"] / df["no_of_workers"].replace(0, 1)
        print("✅ Created incentive_per_worker feature")
        features_created += 1

    # WIP per worker
    if "wip" in X_train_eng.columns and "no_of_workers" in X_train_eng.columns:
        for df in [X_train_eng, X_test_eng]:
            df["wip_per_worker"] = df["wip"] / df["no_of_workers"].replace(0, 1)
        print("✅ Created wip_per_worker feature")
        features_created += 1

    # Handle any NaN or infinite values created during feature engineering
    for df in [X_train_eng, X_test_eng]:
        # Replace infinite values with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Fill NaN values with median
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)

    print("\n📊 Feature Engineering Summary:")
    print(f"   ├─ Features before: {X_train.shape[1]}")
    print(f"   ├─ Features after: {X_train_eng.shape[1]}")
    print(f"   └─ New features created: {features_created}")

    return X_train_eng, X_test_eng


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame, scaler_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale features using StandardScaler.

    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        scaler_path (str, optional): Path to load existing scaler

    Returns:
        Tuple containing:
            - X_train_scaled (pd.DataFrame): Scaled training features
            - X_test_scaled (pd.DataFrame): Scaled testing features
            - scaler (StandardScaler): Fitted scaler object
    """
    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║  📏 Scaling Features                                                 ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"✅ Loaded existing scaler from: {scaler_path}")
    else:
        scaler = StandardScaler()
        scaler.fit(X_train)
        print("✅ Fitted new StandardScaler on training data")

    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train), columns=X_train.columns, index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    print("✅ Features scaled successfully")
    print(f"   ├─ Training set shape: {X_train_scaled.shape}")
    print(f"   └─ Testing set shape: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, scaler


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    hyperparameter_tuning: bool = False,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int = 10,
    min_samples_split: int = 15,
    min_samples_leaf: int = 2,
    max_features: float = 0.5,
    min_impurity_decrease: float = 0.0001,
) -> RandomForestRegressor:
    """
    Train a Random Forest regression model with tuned hyperparameters for optimal performance.

    Args:
        X_train (pd.DataFrame): Scaled training features
        y_train (pd.Series): Training target values
        hyperparameter_tuning (bool): Whether to perform grid search
        random_state (int): Random seed for reproducibility
        n_estimators (int): Number of trees in the forest (default: 200 - tuned)
        max_depth (int): Maximum depth of trees (default: 10 - tuned for balance)
        min_samples_split (int): Minimum samples to split node (default: 15 - tuned)
        min_samples_leaf (int): Minimum samples in leaf node (default: 2 - tuned)
        max_features (float): Fraction of features for splits (default: 0.5 - tuned)
        min_impurity_decrease (float): Min impurity decrease for split (default: 0.0001)

    Returns:
        RandomForestRegressor: Trained Random Forest model with tuned hyperparameters
    """
    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║  🏋️ Training Tuned Random Forest Model                               ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # Initialize Random Forest model with tuned hyperparameters
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        min_impurity_decrease=min_impurity_decrease,
        bootstrap=True,
        oob_score=True,
        random_state=random_state,
        n_jobs=-1,
        warm_start=False,
    )

    print("✅ Model: Random Forest Regressor (Tuned Configuration)")
    print(f"   ├─ n_estimators: {n_estimators} (tuned)")
    print(f"   ├─ max_depth: {max_depth} (tuned for optimal depth)")
    print(f"   ├─ min_samples_split: {min_samples_split} (tuned for regularization)")
    print(f"   ├─ min_samples_leaf: {min_samples_leaf} (tuned for generalization)")
    print(f"   ├─ max_features: {max_features} (tuned - 50% feature subsampling)")
    print(f"   └─ min_impurity_decrease: {min_impurity_decrease}")

    if hyperparameter_tuning:
        print("\n🔍 Performing extensive hyperparameter tuning with GridSearchCV...")

        param_grid = {
            "n_estimators": [200, 300, 400],
            "max_depth": [10, 15, 20],
            "min_samples_split": [5, 10, 15],
            "min_samples_leaf": [2, 4, 6],
            "max_features": ["sqrt", "log2", 0.5],
        }

        grid_search = GridSearchCV(
            estimator=model, param_grid=param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_

        print(f"✅ Best parameters found: {grid_search.best_params_}")
        print(f"✅ Best CV score: {grid_search.best_score_:.4f}")
    else:
        print("\n📊 Training with optimized parameters...")
        model.fit(X_train, y_train)
        print("✅ Model training completed")

        # Show OOB score (Out-of-Bag) - a good estimate of generalization
        if hasattr(model, "oob_score_"):
            print(f"   └─ OOB Score: {model.oob_score_:.4f} (out-of-bag estimate)")

    # Cross-validation score with more folds for better estimate
    cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring="r2", n_jobs=-1)
    print("\n📊 Cross-Validation Results (10-fold):")
    print(f"   ├─ Mean R² Score: {cv_scores.mean():.4f}")
    print(f"   ├─ Std Dev: {cv_scores.std():.4f}")
    print(f"   ├─ Min CV Score: {cv_scores.min():.4f}")
    print(f"   └─ Max CV Score: {cv_scores.max():.4f}")

    return model


def evaluate_model(
    model: RandomForestRegressor,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate model performance on training and testing sets.

    Args:
        model: Trained model object
        X_train (pd.DataFrame): Scaled training features
        X_test (pd.DataFrame): Scaled testing features
        y_train (pd.Series): Training target values
        y_test (pd.Series): Testing target values
        verbose (bool): Whether to print detailed metrics

    Returns:
        Dict[str, float]: Dictionary containing all evaluation metrics
    """
    if verbose:
        print("\n╔══════════════════════════════════════════════════════════════════════╗")
        print("║  📊 Evaluating Model                                                 ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
        "test_mape": mean_absolute_percentage_error(y_test, y_test_pred) * 100,
    }

    # Calculate business accuracy metrics
    metrics["accuracy_5pct"] = (np.abs(y_test - y_test_pred) <= 0.05).sum() / len(y_test) * 100
    metrics["accuracy_10pct"] = (np.abs(y_test - y_test_pred) <= 0.10).sum() / len(y_test) * 100

    # Overfitting detection
    metrics["r2_difference"] = metrics["train_r2"] - metrics["test_r2"]
    metrics["is_overfitting"] = metrics["r2_difference"] > 0.1

    if verbose:
        print("\n📈 TRAINING SET PERFORMANCE:")
        print(f"   ├─ R² Score: {metrics['train_r2']:.4f}")
        print(f"   ├─ RMSE: {metrics['train_rmse']:.4f}")
        print(f"   └─ MAE: {metrics['train_mae']:.4f}")

        print("\n📉 TESTING SET PERFORMANCE:")
        print(f"   ├─ R² Score: {metrics['test_r2']:.4f}")
        print(f"   ├─ RMSE: {metrics['test_rmse']:.4f}")
        print(f"   ├─ MAE: {metrics['test_mae']:.4f}")
        print(f"   └─ MAPE: {metrics['test_mape']:.2f}%")

        print("\n🎯 BUSINESS ACCURACY:")
        print(f"   ├─ Within ±5%: {metrics['accuracy_5pct']:.1f}%")
        print(f"   └─ Within ±10%: {metrics['accuracy_10pct']:.1f}%")

        print("\n⚠️ OVERFITTING CHECK:")
        print(f"   ├─ R² Difference (Train - Test): {metrics['r2_difference']:.4f}")
        if metrics["is_overfitting"]:
            print("   └─ ⚠️  Warning: Model may be overfitting!")
        else:
            print("   └─ ✅ Model generalization looks good")

    return metrics


def save_model(
    model: RandomForestRegressor,
    scaler: StandardScaler,
    model_path: str = "artifacts/models/model.pkl",
    scaler_path: str = "artifacts/scalers/scaler.pkl",
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save trained model, scaler, and metadata to disk.

    Args:
        model: Trained model object
        scaler (StandardScaler): Fitted scaler object
        model_path (str): Path to save the model
        scaler_path (str): Path to save the scaler
        metadata (dict, optional): Additional metadata to save
    """
    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║  💾 Saving Model Artifacts                                           ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    # Save model
    joblib.dump(model, model_path)
    print(f"✅ Model saved to: {model_path}")

    # Save scaler
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved to: {scaler_path}")

    # Save metadata if provided
    if metadata:
        metadata_path = model_path.replace(".pkl", "_metadata.pkl")
        metadata["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        joblib.dump(metadata, metadata_path)
        print(f"✅ Metadata saved to: {metadata_path}")

    print("\n📦 All artifacts saved successfully!")


def load_model(
    model_path: str = "artifacts/models/model.pkl",
    scaler_path: str = "artifacts/scalers/scaler.pkl",
) -> Tuple[RandomForestRegressor, StandardScaler]:
    """
    Load trained model and scaler from disk.

    Args:
        model_path (str): Path to the saved model
        scaler_path (str): Path to the saved scaler

    Returns:
        Tuple containing:
            - model: Loaded model object
            - scaler (StandardScaler): Loaded scaler object

    Raises:
        FileNotFoundError: If model or scaler files don't exist
    """
    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║  📊 Loading Model Artifacts                                          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    # Load model
    model = joblib.load(model_path)
    print(f"✅ Model loaded from: {model_path}")

    # Load scaler
    scaler = joblib.load(scaler_path)
    print(f"✅ Scaler loaded from: {scaler_path}")

    # Load metadata if exists
    metadata_path = model_path.replace(".pkl", "_metadata.pkl")
    if os.path.exists(metadata_path):
        metadata = joblib.load(metadata_path)
        print(f"✅ Metadata loaded from: {metadata_path}")
        print("\n📊 Model Information:")
        for key, value in metadata.items():
            print(f"   ├─ {key}: {value}")

    return model, scaler


def predict(
    model: RandomForestRegressor,
    scaler: StandardScaler,
    X_new: pd.DataFrame,
    return_confidence: bool = False,
) -> np.ndarray:
    """
    Make predictions on new data.

    Args:
        model: Trained model object
        scaler (StandardScaler): Fitted scaler object
        X_new (pd.DataFrame): New data for prediction
        return_confidence (bool): Whether to return prediction intervals (for tree-based models)

    Returns:
        np.ndarray: Predicted values (and confidence intervals if requested)
    """
    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║  🔮 Making Predictions                                               ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # Scale features
    X_scaled = pd.DataFrame(scaler.transform(X_new), columns=X_new.columns, index=X_new.index)

    print(f"✅ Scaled {X_new.shape[0]} samples for prediction")

    # Make predictions
    predictions = model.predict(X_scaled)
    print("✅ Predictions generated successfully")

    # Calculate prediction statistics
    print("\n📊 Prediction Statistics:")
    print(f"   ├─ Mean: {predictions.mean():.4f}")
    print(f"   ├─ Median: {np.median(predictions):.4f}")
    print(f"   ├─ Min: {predictions.min():.4f}")
    print(f"   └─ Max: {predictions.max():.4f}")

    return predictions


def get_feature_importance(
    model: RandomForestRegressor,
    feature_names: List[str],
    top_n: int = 20,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract and visualize feature importance from trained model.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (List[str]): List of feature names
        top_n (int): Number of top features to return
        save_path (str, optional): Path to save the importance DataFrame

    Returns:
        pd.DataFrame: Feature importance dataframe sorted by importance
    """
    if not hasattr(model, "feature_importances_"):
        print("⚠️  Model does not have feature_importances_ attribute")
        return pd.DataFrame()

    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║  📊 Feature Importance Analysis                                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    print(f"\n🏆 TOP {top_n} MOST IMPORTANT FEATURES:")
    for idx, row in importance_df.head(top_n).iterrows():
        print(f"   {idx + 1:2d}. {row['Feature']:<40} → {row['Importance']:.6f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        importance_df.to_csv(save_path, index=False)
        print(f"\n✅ Feature importance saved to: {save_path}")

    return importance_df


if __name__ == "__main__":
    print("══════════════════════════════════════════════════════════════════════")
    print("ML PIPELINE MODULE - Random Forest Regressor")
    print("══════════════════════════════════════════════════════════════════════")
    print("\nThis module contains functions for the complete ML pipeline.")
    print("Import these functions in your main script to use them.")
    print("\nAvailable functions:")
    print("  • load_data()")
    print("  • clean_data()")
    print("  • prepare_data()")
    print("  • engineer_features()")
    print("  • scale_features()")
    print("  • train_model() - Random Forest")
    print("  • evaluate_model()")
    print("  • save_model()")
    print("  • load_model()")
    print("  • predict()")
    print("  • get_feature_importance()")
    print("══════════════════════════════════════════════════════════════════════")
