"""
Main Script for Garment Productivity Prediction Pipeline

This script provides a command-line interface to execute the complete ML pipeline
using Random Forest Regressor for predicting garment productivity.

Usage:
    python main.py --mode train --data data.csv --target actual_productivity
    python main.py --mode predict --data new_data.csv --model artifacts/models/model.pkl
    python main.py --mode full_pipeline --data data.csv --target actual_productivity
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np

# Import pipeline functions
from model_pipeline import (
    load_data,
    clean_data,
    prepare_data,
    engineer_features,
    scale_features,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    predict,
    get_feature_importance,
)


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Garment Productivity Prediction ML Pipeline (Random Forest)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
      # Run full training pipeline
      python main.py --mode full_pipeline --data data.csv --target actual_productivity

      # Train model with hyperparameter tuning
      python main.py --mode train --data data.csv --target actual_productivity --tuning

      # Make predictions on new data
      python main.py --mode predict --data new_data.csv --model artifacts/models/model.pkl

      # Evaluate existing model
      python main.py --mode evaluate --data data.csv --target actual_productivity --model artifacts/models/model.pkl
        """,
    )

    # Required arguments
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["full_pipeline", "train", "evaluate", "predict", "feature_importance"],
        help="Execution mode: full_pipeline, train, evaluate, predict, or feature_importance",
    )

    parser.add_argument("--data", type=str, required=True, help="Path to the CSV data file")

    # Optional arguments
    parser.add_argument(
        "--target",
        type=str,
        default="actual_productivity",
        help="Name of the target column (default: actual_productivity)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="artifacts/models/model.pkl",
        help="Path to save/load the trained model (default: artifacts/models/model.pkl)",
    )

    parser.add_argument(
        "--scaler",
        type=str,
        default="artifacts/scalers/scaler.pkl",
        help="Path to save/load the scaler (default: artifacts/scalers/scaler.pkl)",
    )

    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Proportion of data for testing (default: 0.2)"
    )

    parser.add_argument(
        "--random_state", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )

    parser.add_argument(
        "--tuning", action="store_true", help="Enable hyperparameter tuning with GridSearchCV"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Path to save predictions (default: predictions.csv)",
    )

    parser.add_argument(
        "--feature_importance_path",
        type=str,
        default="artifacts/results/feature_importance.csv",
        help="Path to save feature importance (default: artifacts/results/feature_importance.csv)",
    )

    parser.add_argument(
        "--no_feature_engineering", action="store_true", help="Skip feature engineering step"
    )

    return parser.parse_args()


def run_full_pipeline(args):
    """
    Execute the complete ML pipeline from data loading to model saving.

    Args:
        args: Parsed command-line arguments
    """
    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║  🚀 Running Full ML Pipeline                                         ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # Step 1: Load data
    df, target = load_data(args.data, args.target)

    # Step 2: Clean data
    df_clean = clean_data(df, target)

    # Step 3: Prepare data (encode and split)
    X_train, X_test, y_train, y_test = prepare_data(
        df_clean, target, test_size=args.test_size, random_state=args.random_state
    )

    # Step 4: Feature engineering (optional)
    if not args.no_feature_engineering:
        X_train, X_test = engineer_features(X_train, X_test)

    # Step 5: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Step 6: Train model
    model = train_model(
        X_train_scaled, y_train, hyperparameter_tuning=args.tuning, random_state=args.random_state
    )

    # Step 7: Evaluate model
    metrics = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, verbose=True)

    # Step 8: Save model artifacts
    metadata = {
        "model_type": "random_forest",
        "test_r2": metrics["test_r2"],
        "test_rmse": metrics["test_rmse"],
        "test_mae": metrics["test_mae"],
        "test_mape": metrics["test_mape"],
        "target_column": target,
        "n_features": X_train_scaled.shape[1],
        "feature_names": list(X_train_scaled.columns),  # Save feature names for API
        "hyperparameter_tuning": args.tuning,
        "random_state": args.random_state,
    }

    save_model(model, scaler, args.model, args.scaler, metadata)

    # Step 9: Feature importance (if available)
    if hasattr(model, "feature_importances_"):
        get_feature_importance(
            model, X_train_scaled.columns.tolist(), top_n=20, save_path=args.feature_importance_path
        )

    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║  ✅ Full Pipeline Completed Successfully!                            ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print("\n📊 Final Model Performance:")
    print(f"   ├─ Test R² Score: {metrics['test_r2']:.4f}")
    print(f"   ├─ Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"   ├─ Test MAE: {metrics['test_mae']:.4f}")
    print(f"   └─ Test MAPE: {metrics['test_mape']:.2f}%")
    print("\n💾 Artifacts Saved:")
    print(f"   ├─ Model: {args.model}")
    print(f"   └─ Scaler: {args.scaler}")

    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║  🎉 Execution Completed Successfully!                                ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print("")


def run_train_only(args):
    """
    Train a model on prepared data.

    Args:
        args: Parsed command-line arguments
    """
    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║  🏋️ Training Model                                                   ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # Load and prepare data
    df, target = load_data(args.data, args.target)
    df_clean = clean_data(df, target)
    X_train, X_test, y_train, y_test = prepare_data(
        df_clean, target, args.test_size, args.random_state
    )

    # Feature engineering
    if not args.no_feature_engineering:
        X_train, X_test = engineer_features(X_train, X_test)

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Train model
    model = train_model(
        X_train_scaled, y_train, hyperparameter_tuning=args.tuning, random_state=args.random_state
    )

    # Evaluate model
    metrics = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, verbose=True)

    # Save model
    metadata = {
        "model_type": "random_forest",
        "test_r2": metrics["test_r2"],
        "target_column": target,
        "n_features": X_train_scaled.shape[1],
        "feature_names": list(X_train_scaled.columns),  # Save feature names for API
    }
    save_model(model, scaler, args.model, args.scaler, metadata)

    print(f"\n✅ Training completed! Model saved to: {args.model}")


def run_evaluate(args):
    """
    Evaluate an existing model on new data.

    Args:
        args: Parsed command-line arguments
    """
    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║  📊 Evaluating Model                                                 ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # Load model and scaler
    model, scaler = load_model(args.model, args.scaler)

    # Load and prepare data
    df, target = load_data(args.data, args.target)
    df_clean = clean_data(df, target)
    X_train, X_test, y_train, y_test = prepare_data(
        df_clean, target, args.test_size, args.random_state
    )

    # Feature engineering
    if not args.no_feature_engineering:
        X_train, X_test = engineer_features(X_train, X_test)

    # Scale features
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    # Evaluate (printed via verbose=True)
    evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, verbose=True)

    print("\n✅ Evaluation completed!")


def run_predict(args):
    """
    Make predictions on new data using a trained model.

    Args:
        args: Parsed command-line arguments
    """
    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║  🔮 Making Predictions                                               ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # Load model and scaler
    model, scaler = load_model(args.model, args.scaler)

    # Load new data (without target column requirement)
    print(f"\n📂 Loading data from: {args.data}")
    df_new = pd.read_csv(args.data)
    print(f"✅ Loaded {df_new.shape[0]} samples with {df_new.shape[1]} features")

    # Check if target column exists (for evaluation)
    has_target = args.target in df_new.columns
    if has_target:
        y_true = df_new[args.target].copy()
        df_new = df_new.drop(args.target, axis=1)
        print("✅ Target column found - will evaluate predictions")

    # Apply same preprocessing as training
    # Note: Categorical encoding should match training data
    cat_cols = df_new.select_dtypes(include=["object"]).columns.tolist()
    if len(cat_cols) > 0:
        df_new = pd.get_dummies(df_new, columns=cat_cols, drop_first=True, dtype=int)

    # Feature engineering (if not disabled)
    if not args.no_feature_engineering:
        # Create a dummy train set for feature engineering function
        X_dummy = df_new.copy()
        df_new, _ = engineer_features(df_new, X_dummy)

    # Make predictions
    predictions = predict(model, scaler, df_new)

    # Save predictions
    output_df = pd.DataFrame({"prediction": predictions})

    if has_target:
        output_df["actual"] = y_true.values
        output_df["error"] = output_df["actual"] - output_df["prediction"]
        output_df["abs_error"] = np.abs(output_df["error"])
        output_df["pct_error"] = (output_df["abs_error"] / output_df["actual"] * 100).round(2)

        # Calculate metrics
        r2 = 1 - (
            np.sum(output_df["error"] ** 2)
            / np.sum((output_df["actual"] - output_df["actual"].mean()) ** 2)
        )
        rmse = np.sqrt(np.mean(output_df["error"] ** 2))
        mae = np.mean(output_df["abs_error"])
        mape = np.mean(output_df["pct_error"])

        print("\n📊 Prediction Metrics:")
        print(f"   ├─ R² Score: {r2:.4f}")
        print(f"   ├─ RMSE: {rmse:.4f}")
        print(f"   ├─ MAE: {mae:.4f}")
        print(f"   └─ MAPE: {mape:.2f}%")

    output_df.to_csv(args.output, index=False)
    print(f"\n✅ Predictions saved to: {args.output}")


def run_feature_importance(args):
    """
    Display feature importance from a trained model.

    Args:
        args: Parsed command-line arguments
    """
    print("\n" + "=" * 80)
    print("📊 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)

    # Load model
    model, scaler = load_model(args.model, args.scaler)

    # Load data to get feature names
    df, target = load_data(args.data, args.target)
    df_clean = clean_data(df, target)
    X_train, _, _, _ = prepare_data(df_clean, target, args.test_size, args.random_state)

    # Feature engineering to get final feature names
    if not args.no_feature_engineering:
        X_train, X_dummy = engineer_features(X_train, X_train.copy())

    # Get feature importance
    get_feature_importance(
        model, X_train.columns.tolist(), top_n=20, save_path=args.feature_importance_path
    )

    print("\n✅ Feature importance analysis completed!")


def main():
    """
    Main execution function.
    """
    try:
        # Parse arguments
        args = parse_arguments()

        # Validate file paths
        if not os.path.exists(args.data):
            print(f"❌ Error: Data file not found: {args.data}")
            sys.exit(1)

        # Execute based on mode
        if args.mode == "full_pipeline":
            run_full_pipeline(args)
        elif args.mode == "train":
            run_train_only(args)
        elif args.mode == "evaluate":
            run_evaluate(args)
        elif args.mode == "predict":
            run_predict(args)
        elif args.mode == "feature_importance":
            run_feature_importance(args)
        else:
            print(f"❌ Error: Unknown mode: {args.mode}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
