"""
Training Script with Monitoring Integration

This enhanced version of train.py integrates MLflow + Elasticsearch monitoring
to track model performance, data drift, and system metrics in real-time.

Usage:
    python scripts/train_monitored.py --mode full_pipeline --data data/raw/data.csv
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pipeline functions from src package
from src.model_pipeline import (
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

# Import monitoring module
try:
    from src.monitoring import MLOpsMonitor, DataDriftDetector
    MONITORING_AVAILABLE = True
except ImportError:
    print("⚠️  Monitoring module not available. Install with: pip install elasticsearch psutil")
    MONITORING_AVAILABLE = False

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "garment_productivity_pipeline")


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Garment Productivity Prediction with Monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["full_pipeline", "train", "evaluate", "predict", "feature_importance"],
        help="Execution mode",
    )

    parser.add_argument("--data", type=str, required=True, help="Path to the CSV data file")

    parser.add_argument(
        "--target",
        type=str,
        default="actual_productivity",
        help="Name of the target column",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="artifacts/models/model.pkl",
        help="Path to save/load the trained model",
    )

    parser.add_argument(
        "--scaler",
        type=str,
        default="artifacts/scalers/scaler.pkl",
        help="Path to save/load the scaler",
    )

    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Proportion of data for testing"
    )

    parser.add_argument(
        "--random_state", type=int, default=42, help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--tuning", action="store_true", help="Enable hyperparameter tuning"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Path to save predictions",
    )

    parser.add_argument(
        "--no_monitoring",
        action="store_true",
        help="Disable monitoring (use standard pipeline)",
    )

    return parser.parse_args()


def run_full_pipeline_monitored(args):
    """
    Execute the complete ML pipeline with monitoring
    """
    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║  🚀 Running Full ML Pipeline with Monitoring                         ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    # Initialize monitoring
    use_monitoring = MONITORING_AVAILABLE and not args.no_monitoring
    monitor = None

    if use_monitoring:
        try:
            monitor = MLOpsMonitor(mlflow_tracking_uri=TRACKING_URI)
            run_id = monitor.start_monitored_run(
                run_name="full_pipeline_monitored", experiment_name=EXPERIMENT_NAME
            )
            print(f"✅ Monitoring enabled (Run ID: {run_id[:8]}...)")
        except Exception as e:
            print(f"⚠️  Monitoring initialization failed: {str(e)}")
            print("   Continuing without monitoring...")
            use_monitoring = False

    # Configure MLflow (fallback if monitoring not available)
    if not use_monitoring:
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        mlflow.start_run(run_name="full_pipeline")

    # Step 1: Load data
    df, target = load_data(args.data, args.target)

    # Step 2: Clean data
    df_clean = clean_data(df, target)

    # Step 3: Prepare data
    X_train, X_test, y_train, y_test = prepare_data(
        df_clean, target, test_size=args.test_size, random_state=args.random_state
    )

    # Step 4: Feature engineering
    X_train_eng, X_test_eng = engineer_features(X_train, X_test)

    # Step 5: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_eng, X_test_eng)

    # Log parameters
    params = {
        "test_size": args.test_size,
        "random_state": args.random_state,
        "hyperparameter_tuning": args.tuning,
        "n_features": int(X_train_scaled.shape[1]),
        "target": str(target),
    }

    if use_monitoring and monitor:
        monitor.log_params_monitored(params)
    else:
        mlflow.log_params(params)

    # Step 6: Train model
    model = train_model(
        X_train_scaled, y_train, hyperparameter_tuning=args.tuning, random_state=args.random_state
    )

    # Step 7: Evaluate model
    metrics = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, verbose=True)

    # Log metrics
    if use_monitoring and monitor:
        monitor.log_metrics_monitored(
            {
                "train_r2": metrics["train_r2"],
                "test_r2": metrics["test_r2"],
                "train_rmse": metrics["train_rmse"],
                "test_rmse": metrics["test_rmse"],
                "train_mae": metrics["train_mae"],
                "test_mae": metrics["test_mae"],
                "test_mape": metrics["test_mape"],
                "accuracy_5pct": metrics["accuracy_5pct"],
                "accuracy_10pct": metrics["accuracy_10pct"],
            },
            step=1,
        )
    else:
        mlflow.log_metrics(
            {
                "train_r2": metrics["train_r2"],
                "test_r2": metrics["test_r2"],
                "train_rmse": metrics["train_rmse"],
                "test_rmse": metrics["test_rmse"],
                "train_mae": metrics["train_mae"],
                "test_mae": metrics["test_mae"],
                "test_mape": metrics["test_mape"],
            },
            step=1,
        )

    # Log model
    X_df = pd.DataFrame(X_train_scaled)
    preds = model.predict(X_df)
    signature = infer_signature(X_df, preds)
    input_example = X_df.head(2)

    try:
        mlflow.sklearn.log_model(
            model, artifact_path="model", signature=signature, input_example=input_example
        )
    except TypeError:
        mlflow.sklearn.log_model(model, "model", signature=signature, input_example=input_example)

    # Step 8: Data drift detection (if monitoring enabled)
    if use_monitoring and monitor:
        try:
            detector = DataDriftDetector(X_train_eng)
            drift_results = detector.detect_drift(X_test_eng, threshold=0.05)

            print("\n🔍 Data Drift Analysis:")
            print(f"   Overall drift score: {drift_results['overall_drift_score']:.4f}")
            print(f"   Drifted features: {drift_results['drifted_features']}")

            # Log drift metrics
            monitor.log_metrics_monitored(
                {
                    "drift_score": drift_results["overall_drift_score"],
                    "n_drifted_features": len(drift_results["drifted_features"]),
                },
                step=1,
            )
        except Exception as e:
            print(f"⚠️  Drift detection failed: {str(e)}")

    # Step 9: Save model artifacts
    metadata = {
        "model_type": "random_forest",
        "test_r2": metrics["test_r2"],
        "test_rmse": metrics["test_rmse"],
        "test_mae": metrics["test_mae"],
        "test_mape": metrics["test_mape"],
        "target_column": target,
        "n_features": X_train_scaled.shape[1],
        "feature_names": list(X_train_scaled.columns),
        "hyperparameter_tuning": args.tuning,
        "random_state": args.random_state,
        "monitoring_enabled": use_monitoring,
    }

    save_model(model, scaler, args.model, args.scaler, metadata)

    # End run
    if use_monitoring and monitor:
        monitor.end_monitored_run()
    else:
        mlflow.end_run()

    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║  ✅ Full Pipeline Completed Successfully with Monitoring!            ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print("\n📊 Final Model Performance:")
    print(f"   ├─ Test R² Score: {metrics['test_r2']:.4f}")
    print(f"   ├─ Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"   ├─ Test MAE: {metrics['test_mae']:.4f}")
    print(f"   └─ Test MAPE: {metrics['test_mape']:.2f}%")
    print("\n💾 Artifacts Saved:")
    print(f"   ├─ Model: {args.model}")
    print(f"   └─ Scaler: {args.scaler}")

    if use_monitoring:
        print("\n🔍 Monitoring:")
        print("   ├─ MLflow UI: http://localhost:5000")
        print("   ├─ Kibana: http://localhost:5601")
        print("   └─ Elasticsearch: http://localhost:9200")

    print("")


def main():
    """Main execution function"""
    try:
        args = parse_arguments()

        if not os.path.exists(args.data):
            print(f"❌ Error: Data file not found: {args.data}")
            sys.exit(1)

        # Execute based on mode
        if args.mode == "full_pipeline":
            run_full_pipeline_monitored(args)
        else:
            # For other modes, use original main.py logic
            print("⚠️  Other modes not yet migrated to monitored version")
            print("   Use: python main.py for standard execution")
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
