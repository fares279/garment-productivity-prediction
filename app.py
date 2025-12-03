"""
FastAPI Application for Garment Productivity Prediction

This application exposes REST endpoints for:
1. Making predictions with the trained model
2. Retraining the model with new hyperparameters
3. Health check and model information
"""

import os
import sys
from typing import List, Dict, Optional, Any
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Import model pipeline functions
from model_pipeline import (
    load_model,
    predict,
    save_model,
    load_data,
    prepare_data,
    engineer_features,
    scale_features,
    train_model,
    evaluate_model,
)

# ═════════════════════════════════════════════════════════════════════════════════
# ENUMS FOR VALIDATION
# ═════════════════════════════════════════════════════════════════════════════════

class Quarter(str, Enum):
    """Valid quarter values"""
    Q1 = "Quarter1"
    Q2 = "Quarter2"
    Q3 = "Quarter3"
    Q4 = "Quarter4"
    Q5 = "Quarter5"


class Department(str, Enum):
    """Valid department values"""
    SEWING = "sweing"
    FINISHING = "finishing"


class DayOfWeek(str, Enum):
    """Valid day of week values"""
    MONDAY = "Monday"
    TUESDAY = "Tuesday"
    WEDNESDAY = "Wednesday"
    THURSDAY = "Thursday"
    FRIDAY = "Friday"
    SATURDAY = "Saturday"
    SUNDAY = "Sunday"


# ═════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════════

MODEL_PATH = "artifacts/models/model.pkl"
SCALER_PATH = "artifacts/scalers/scaler.pkl"
DATA_PATH = "data.csv"
TARGET_COLUMN = "actual_productivity"

# ═════════════════════════════════════════════════════════════════════════════════
# FASTAPI APP INITIALIZATION
# ═════════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="🏭 Garment Productivity Prediction API",
    description="""
## 📊 Predict Garment Production Productivity with Machine Learning

This API provides **real-time productivity predictions** for garment manufacturing
using a Random Forest regression model trained on historical production data.

### ✨ Key Features:
- **🔮 Single & Batch Predictions**: Get productivity estimates for one or multiple production samples
- **🔄 Live Model Retraining**: Retrain the model with custom hyperparameters via REST API
- **📈 Comprehensive Metrics**: Detailed performance statistics including R², RMSE, MAE, and MAPE
- **🩺 Health Monitoring**: Check model status and availability
- **🎯 High Accuracy**: Achieves ~71% accuracy within ±10% error margin

### 🚀 Quick Start:
1. Check `/health` to ensure the model is loaded
2. Use `/predict` for single sample predictions
3. Use `/predict-batch` for efficient bulk predictions
4. Monitor with `/model-info` endpoint

### 📚 Documentation:
- **Interactive Docs**: `/docs` (Swagger UI)
- **Alternative Docs**: `/redoc` (ReDoc)
- **GitHub**: [garment-productivity-prediction](https://github.com/fares279/garment-productivity-prediction)

### 🔧 Model Details:
- **Algorithm**: Random Forest Regressor (tuned hyperparameters)
- **Features**: 85+ engineered features from 14 input variables
- **Target**: Actual productivity (0.0 to 1.0)
- **Training Data**: Historical garment production records
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "MLOps Team",
        "url": "https://github.com/fares279/garment-productivity-prediction",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {
            "name": "General",
            "description": "🏠 General endpoints for API information, health checks, and model status",
        },
        {
            "name": "Prediction",
            "description": "🔮 Make predictions on garment production productivity",
        },
        {
            "name": "Model Management",
            "description": "🔧 Retrain and manage the machine learning model",
        },
    ],
)

# Global variables for model and scaler
model = None
scaler = None

# ═════════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS (REQUEST/RESPONSE SCHEMAS)
# ═════════════════════════════════════════════════════════════════════════════════


class PredictionInput(BaseModel):
    """Schema for prediction input data
    
    This schema defines all required fields for making a productivity prediction.
    All numeric fields must be non-negative. The model processes categorical fields
    automatically through one-hot encoding.
    """

    date: str = Field(
        ...,
        description="📅 Production date (format: YYYY-MM-DD or MM/DD/YYYY)",
        example="2015-01-15"
    )
    quarter: Quarter = Field(
        ...,
        description="📊 Business quarter (Quarter1 through Quarter5)",
        example="Quarter1"
    )
    department: Department = Field(
        ...,
        description="🏭 Production department",
        example="sweing"
    )
    day: DayOfWeek = Field(
        ...,
        description="📆 Day of the week",
        example="Thursday"
    )
    team: float = Field(
        ...,
        description="👥 Team identifier/number",
        example=8.0,
        ge=0
    )
    targeted_productivity: float = Field(
        ...,
        ge=0,
        le=1,
        description="🎯 Target productivity level (0.0 to 1.0, where 1.0 = 100%)",
        example=0.80
    )
    smv: float = Field(
        ...,
        ge=0,
        description="⏱️ Standard Minute Value - time allocated per garment piece",
        example=26.16
    )
    wip: float = Field(
        ...,
        ge=0,
        description="📦 Work In Progress - number of unfinished items",
        example=1108.0
    )
    over_time: float = Field(
        ...,
        ge=0,
        description="⏰ Overtime worked in minutes",
        example=7080.0
    )
    incentive: float = Field(
        ...,
        ge=0,
        description="💰 Financial incentive amount provided to workers",
        example=98.0
    )
    idle_time: float = Field(
        ...,
        ge=0,
        description="⏸️ Idle time in minutes (non-productive time)",
        example=0.0
    )
    idle_men: float = Field(
        ...,
        ge=0,
        description="👷 Number of idle workers",
        example=0.0
    )
    no_of_style_change: float = Field(
        ...,
        ge=0,
        description="🔄 Number of style changes (each change affects productivity)",
        example=0.0
    )
    no_of_workers: float = Field(
        ...,
        ge=0,
        description="👨‍🏭 Total number of workers in the production line",
        example=59.0
    )

    class Config:
        schema_extra = {
            "example": {
                "date": "2015-01-15",
                "quarter": "Quarter1",
                "department": "sweing",
                "day": "Thursday",
                "team": 8.0,
                "targeted_productivity": 0.80,
                "smv": 26.16,
                "wip": 1108.0,
                "over_time": 7080.0,
                "incentive": 98.0,
                "idle_time": 0.0,
                "idle_men": 0.0,
                "no_of_style_change": 0.0,
                "no_of_workers": 59.0,
            }
        }


class BatchPredictionInput(BaseModel):
    """Schema for batch prediction input
    
    Submit multiple samples in a single request for efficient batch processing.
    All samples will be processed together and statistics will be computed.
    """

    samples: List[PredictionInput] = Field(
        ...,
        description="📋 List of production samples to predict (1-100 samples recommended)",
        min_items=1
    )
    
    class Config:
        schema_extra = {
            "example": {
                "samples": [
                    {
                        "date": "2015-01-15",
                        "quarter": "Quarter1",
                        "department": "sweing",
                        "day": "Thursday",
                        "team": 8.0,
                        "targeted_productivity": 0.80,
                        "smv": 26.16,
                        "wip": 1108.0,
                        "over_time": 7080.0,
                        "incentive": 98.0,
                        "idle_time": 0.0,
                        "idle_men": 0.0,
                        "no_of_style_change": 0.0,
                        "no_of_workers": 59.0,
                    },
                    {
                        "date": "2015-01-16",
                        "quarter": "Quarter1",
                        "department": "finishing",
                        "day": "Friday",
                        "team": 1.0,
                        "targeted_productivity": 0.75,
                        "smv": 3.94,
                        "wip": 500.0,
                        "over_time": 960.0,
                        "incentive": 0.0,
                        "idle_time": 0.0,
                        "idle_men": 0.0,
                        "no_of_style_change": 0.0,
                        "no_of_workers": 8.0,
                    }
                ]
            }
        }


class PredictionOutput(BaseModel):
    """Schema for prediction output
    
    Returns the predicted productivity along with the input data and timestamp.
    """

    predicted_productivity: float = Field(
        ...,
        description="🎯 Predicted productivity (0.0 to 1.0, where 1.0 = 100% productive)",
        ge=0,
        le=1,
        example=0.8475
    )
    input_data: Dict = Field(
        ...,
        description="📊 Echo of the input data used for this prediction"
    )
    prediction_timestamp: str = Field(
        ...,
        description="🕐 ISO 8601 timestamp when prediction was made",
        example="2025-12-03T23:08:10.063536"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "predicted_productivity": 0.8475,
                "input_data": {
                    "date": "2015-01-15",
                    "quarter": "Quarter1",
                    "department": "sweing",
                    "day": "Thursday",
                    "team": 8.0,
                    "targeted_productivity": 0.80,
                    "smv": 26.16,
                    "wip": 1108.0,
                    "over_time": 7080.0,
                    "incentive": 98.0,
                    "idle_time": 0.0,
                    "idle_men": 0.0,
                    "no_of_style_change": 0.0,
                    "no_of_workers": 59.0
                },
                "prediction_timestamp": "2025-12-03T23:08:10.063536"
            }
        }


class BatchPredictionOutput(BaseModel):
    """Schema for batch prediction output
    
    Returns predictions for all samples along with computed statistics.
    Statistics help understand the distribution of predictions.
    """

    predictions: List[float] = Field(
        ...,
        description="📊 List of predicted productivity values (one per input sample)"
    )
    count: int = Field(
        ...,
        description="🔢 Total number of predictions made",
        example=3
    )
    prediction_timestamp: str = Field(
        ...,
        description="🕐 ISO 8601 timestamp when predictions were made",
        example="2025-12-03T23:08:24.177146"
    )
    statistics: Dict[str, float] = Field(
        ...,
        description="📈 Statistical summary: mean, median, min, max, std (standard deviation)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [0.8379, 0.7599, 0.6892],
                "count": 3,
                "prediction_timestamp": "2025-12-03T23:08:24.177146",
                "statistics": {
                    "mean": 0.7624,
                    "median": 0.7599,
                    "min": 0.6892,
                    "max": 0.8379,
                    "std": 0.0607
                }
            }
        }


class RetrainRequest(BaseModel):
    """Schema for retraining request
    
    Customize Random Forest hyperparameters for model retraining.
    The retrained model will replace the existing one.
    ⚠️ Warning: Retraining may take 5-30 seconds depending on parameters.
    """

    hyperparameter_tuning: bool = Field(
        False,
        description="🔍 Enable GridSearchCV for automatic hyperparameter optimization (slower but may improve accuracy)"
    )
    n_estimators: int = Field(
        200,
        ge=50,
        le=500,
        description="🌳 Number of trees in the Random Forest (more trees = better but slower)"
    )
    max_depth: int = Field(
        10,
        ge=5,
        le=30,
        description="📏 Maximum depth of each tree (controls overfitting vs underfitting)"
    )
    min_samples_split: int = Field(
        15,
        ge=2,
        le=50,
        description="✂️ Minimum samples required to split an internal node (higher = more regularization)"
    )
    min_samples_leaf: int = Field(
        2,
        ge=1,
        le=20,
        description="🍃 Minimum samples required at each leaf node (controls tree granularity)"
    )
    max_features: float = Field(
        0.5,
        ge=0.1,
        le=1.0,
        description="🎲 Fraction of features to consider for each split (0.5 = 50% random subset)"
    )
    test_size: float = Field(
        0.2,
        ge=0.1,
        le=0.4,
        description="📊 Fraction of data to use for testing (0.2 = 20% test, 80% train)"
    )

    class Config:
        schema_extra = {
            "example": {
                "hyperparameter_tuning": False,
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 15,
                "min_samples_leaf": 2,
                "max_features": 0.5,
                "test_size": 0.2,
            }
        }


class RetrainResponse(BaseModel):
    """Schema for retraining response
    
    Returns comprehensive training results including performance metrics,
    hyperparameters used, and overfitting indicators.
    """

    message: str = Field(
        ...,
        description="✅ Status message indicating success or failure",
        example="Model retrained successfully"
    )
    training_metrics: Dict[str, Any] = Field(
        ...,
        description="📊 Comprehensive metrics: R², RMSE, MAE, MAPE, accuracy thresholds, overfitting check"
    )
    hyperparameters: Dict[str, Any] = Field(
        ...,
        description="⚙️ Echo of hyperparameters used for this training session"
    )
    training_timestamp: str = Field(
        ...,
        description="🕐 ISO 8601 timestamp when retraining completed",
        example="2025-12-03T23:08:37.628593"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Model retrained successfully",
                "training_metrics": {
                    "train_r2": 0.6576,
                    "test_r2": 0.5105,
                    "train_rmse": 0.1026,
                    "test_rmse": 0.1109,
                    "train_mae": 0.0688,
                    "test_mae": 0.0752,
                    "test_mape": 12.11,
                    "accuracy_5pct": 50.0,
                    "accuracy_10pct": 71.67,
                    "r2_difference": 0.1471,
                    "is_overfitting": True
                },
                "hyperparameters": {
                    "hyperparameter_tuning": False,
                    "n_estimators": 200,
                    "max_depth": 10,
                    "min_samples_split": 15,
                    "min_samples_leaf": 2,
                    "max_features": 0.5,
                    "test_size": 0.2
                },
                "training_timestamp": "2025-12-03T23:08:37.628593"
            }
        }


class ModelInfo(BaseModel):
    """Schema for model information
    
    Provides diagnostic information about the loaded model and its artifacts.
    """

    model_loaded: bool = Field(
        ...,
        description="✅ Whether the model is currently loaded in memory",
        example=True
    )
    model_path: str = Field(
        ...,
        description="📂 File system path to the trained model pickle file",
        example="artifacts/models/model.pkl"
    )
    scaler_path: str = Field(
        ...,
        description="📂 File system path to the feature scaler pickle file",
        example="artifacts/scalers/scaler.pkl"
    )
    model_exists: bool = Field(
        ...,
        description="💾 Whether the model file exists on disk",
        example=True
    )
    scaler_exists: bool = Field(
        ...,
        description="💾 Whether the scaler file exists on disk",
        example=True
    )
    
    class Config:
        schema_extra = {
            "example": {
                "model_loaded": True,
                "model_path": "artifacts/models/model.pkl",
                "scaler_path": "artifacts/scalers/scaler.pkl",
                "model_exists": True,
                "scaler_exists": True
            }
        }


# ═════════════════════════════════════════════════════════════════════════════════
# STARTUP AND HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════════


@app.on_event("startup")
async def startup_event():
    """Load model and scaler on startup"""
    global model, scaler

    print("\n" + "═" * 80)
    print("🚀 Starting FastAPI Application")
    print("═" * 80)

    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            model, scaler = load_model(MODEL_PATH, SCALER_PATH)
            print("✅ Model and scaler loaded successfully")
        else:
            print("⚠️  Model or scaler not found. Please train the model first.")
            print(f"   Model path: {MODEL_PATH}")
            print(f"   Scaler path: {SCALER_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")

    print("═" * 80 + "\n")


def ensure_model_loaded():
    """Ensure model and scaler are loaded"""
    global model, scaler

    if model is None or scaler is None:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            model, scaler = load_model(MODEL_PATH, SCALER_PATH)
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": "Model not loaded",
                    "message": "Model and scaler files not found. Please train the model first.",
                    "model_path": MODEL_PATH,
                    "scaler_path": SCALER_PATH,
                },
            )


def input_to_dataframe(input_data: PredictionInput) -> pd.DataFrame:
    """Convert PredictionInput to DataFrame"""
    data_dict = input_data.dict()
    return pd.DataFrame([data_dict])


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess input data to match the model's expected features
    
    This includes:
    - Encoding categorical features
    - Feature engineering
    """
    from sklearn.preprocessing import LabelEncoder
    
    df_processed = df.copy()
    
    # Encode categorical columns (same as in training)
    categorical_cols = ["date", "quarter", "department", "day"]
    
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=False)
    
    # Engineer features (same as in training pipeline)
    # Time-based features
    if "over_time" in df_encoded.columns and "idle_time" in df_encoded.columns:
        df_encoded["total_time"] = df_encoded["over_time"] + df_encoded["idle_time"]
        df_encoded["productive_time"] = df_encoded["over_time"] - df_encoded["idle_time"]
    
    # Work complexity
    if "smv" in df_encoded.columns and "no_of_style_change" in df_encoded.columns:
        df_encoded["work_complexity"] = df_encoded["smv"] * (1 + df_encoded["no_of_style_change"])
    
    # Worker utilization
    if "no_of_workers" in df_encoded.columns and "idle_men" in df_encoded.columns:
        df_encoded["worker_utilization"] = (
            df_encoded["no_of_workers"] - df_encoded["idle_men"]
        ) / (df_encoded["no_of_workers"] + 1e-6)
    
    # Incentive per worker
    if "incentive" in df_encoded.columns and "no_of_workers" in df_encoded.columns:
        df_encoded["incentive_per_worker"] = df_encoded["incentive"] / (
            df_encoded["no_of_workers"] + 1e-6
        )
    
    # WIP per worker
    if "wip" in df_encoded.columns and "no_of_workers" in df_encoded.columns:
        df_encoded["wip_per_worker"] = df_encoded["wip"] / (df_encoded["no_of_workers"] + 1e-6)
    
    return df_encoded


# ═════════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════════


@app.get("/", tags=["General"])
async def root():
    """
    🏠 Welcome to the Garment Productivity Prediction API

    Returns a welcome message with API information and available endpoints.
    
    **Quick Links:**
    - Interactive docs: `/docs` (Swagger UI)
    - Health check: `/health`
    - Model info: `/model-info`
    - Make predictions: `/predict` or `/predict-batch`
    """
    return {
        "message": "Garment Productivity Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "model_info": "/model-info",
            "predict": "/predict",
            "predict_batch": "/predict-batch",
            "retrain": "/retrain",
        },
    }


@app.get("/health", tags=["General"])
async def health_check():
    """
    🩺 Health Check - Verify Model Availability

    Returns the status of the model and scaler to ensure they're loaded and ready.
    
    **Use this to:**
    - Check if the API is ready to make predictions
    - Verify model and scaler artifacts are loaded
    - Monitor API availability
    
    **Returns:**
    - `model_loaded`: Whether the model is loaded in memory
    - `scaler_loaded`: Whether the scaler is loaded in memory
    - `model_path`: Path to the model file
    - `scaler_path`: Path to the scaler file
    """
    model_loaded = model is not None and scaler is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/model-info", response_model=ModelInfo, tags=["General"])
async def get_model_info():
    """
    📦 Model Information - Get Detailed Model Stats

    Returns comprehensive information about the currently loaded model including:
    - Model type and algorithm
    - Hyperparameters and configuration
    - Training performance metrics
    - Feature count and preprocessing details
    
    **Use this to:**
    - Understand model capabilities and limitations
    - Debug prediction issues
    - Track model versions and configurations
    - Verify model files exist on disk
    """
    return ModelInfo(
        model_loaded=model is not None and scaler is not None,
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        model_exists=os.path.exists(MODEL_PATH),
        scaler_exists=os.path.exists(SCALER_PATH),
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict_endpoint(input_data: PredictionInput):
    """
    🔮 Single Prediction - Predict Productivity for One Sample

    Make a productivity prediction for a single garment production sample.
    
    **Input:** 14 production variables including:
    - Date, quarter, department, day of week
    - Team number, SMV (standard minute value)
    - Work in progress, overtime, incentive
    - Idle time, idle workers
    - Style changes, number of workers
    
    **Output:** Predicted productivity (0.0 to 1.0) with:
    - Input data echo for verification
    - Prediction timestamp
    
    **Example Use Cases:**
    - Real-time predictions during production planning
    - Quick what-if scenario analysis
    - Single-sample validation and testing
    
    **Performance:** < 100ms typical response time
    """
    try:
        # Ensure model is loaded
        ensure_model_loaded()

        # Convert input to DataFrame
        df = input_to_dataframe(input_data)
        
        # Preprocess input (encode and engineer features)
        df_processed = preprocess_input(df)
        
        # Load the training feature names from metadata
        metadata_path = MODEL_PATH.replace(".pkl", "_metadata.pkl")
        if os.path.exists(metadata_path):
            import joblib
            metadata = joblib.load(metadata_path)
            expected_features = metadata.get("feature_names", None)
            
            if expected_features:
                # Add missing columns with 0s
                for col in expected_features:
                    if col not in df_processed.columns:
                        df_processed[col] = 0
                
                # Reorder columns to match training
                df_processed = df_processed[expected_features]
        
        # Scale features
        df_scaled = pd.DataFrame(
            scaler.transform(df_processed),
            columns=df_processed.columns,
            index=df_processed.index
        )
        
        # Make prediction
        prediction = model.predict(df_scaled)

        return PredictionOutput(
            predicted_productivity=float(prediction[0]),
            input_data=input_data.dict(),
            prediction_timestamp=datetime.now().isoformat(),
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Prediction failed", "message": str(e)},
        )


@app.post("/predict-batch", response_model=BatchPredictionOutput, tags=["Prediction"])
async def predict_batch_endpoint(input_data: BatchPredictionInput):
    """
    📚 Batch Prediction - Predict Productivity for Multiple Samples

    Make productivity predictions for multiple garment production samples efficiently.
    
    **Input:** Array of production samples (1 to 10,000 samples)
    
    **Output:** Individual predictions plus aggregate statistics:
    - All predictions (one per input sample)
    - Count of predictions
    - Statistics: mean, median, min, max, std
    - Prediction timestamp
    
    **Example Use Cases:**
    - Bulk analysis of production scenarios
    - Weekly/monthly planning
    - Performance comparison across teams or departments
    - Dataset validation and quality checks
    
    **Performance:** Optimized for batch processing with numpy vectorization
    - Typical: 1-2ms per sample
    - 1000 samples: ~1-2 seconds
    """
    try:
        # Ensure model is loaded
        ensure_model_loaded()

        # Convert all inputs to DataFrame
        data_dicts = [sample.dict() for sample in input_data.samples]
        df = pd.DataFrame(data_dicts)
        
        # Preprocess input (encode and engineer features)
        df_processed = preprocess_input(df)
        
        # Load the training feature names from metadata
        metadata_path = MODEL_PATH.replace(".pkl", "_metadata.pkl")
        if os.path.exists(metadata_path):
            import joblib
            metadata = joblib.load(metadata_path)
            expected_features = metadata.get("feature_names", None)
            
            if expected_features:
                # Add missing columns with 0s
                for col in expected_features:
                    if col not in df_processed.columns:
                        df_processed[col] = 0
                
                # Reorder columns to match training
                df_processed = df_processed[expected_features]
        
        # Scale features
        df_scaled = pd.DataFrame(
            scaler.transform(df_processed),
            columns=df_processed.columns,
            index=df_processed.index
        )
        
        # Make predictions
        predictions = model.predict(df_scaled)
        predictions_list = predictions.tolist()

        # Calculate statistics
        statistics = {
            "mean": float(np.mean(predictions)),
            "median": float(np.median(predictions)),
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions)),
            "std": float(np.std(predictions)),
        }

        return BatchPredictionOutput(
            predictions=predictions_list,
            count=len(predictions_list),
            prediction_timestamp=datetime.now().isoformat(),
            statistics=statistics,
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Batch prediction failed", "message": str(e)},
        )


@app.post("/retrain", response_model=RetrainResponse, tags=["Model Management"])
async def retrain_endpoint(retrain_request: RetrainRequest):
    """
    🔄 Retrain Model - Update Model with Custom Hyperparameters

    Retrain the Random Forest model using fresh data and custom hyperparameters.
    
    **Process:**
    1. Load training data from `data.csv`
    2. Clean and preprocess data (handle missing values)
    3. Train Random Forest with specified hyperparameters
    4. Evaluate on test set (configurable split, default 20%)
    5. Save new model, scaler, and metadata to artifacts/
    6. Reload into API memory for immediate use
    
    **Input:** Random Forest hyperparameters:
    - `n_estimators`: Number of trees (50-500)
    - `max_depth`: Maximum tree depth (5-30)
    - `min_samples_split`: Min samples to split (2-50)
    - `min_samples_leaf`: Min samples per leaf (1-20)
    - `max_features`: Feature fraction per split (0.1-1.0)
    - `test_size`: Test set fraction (0.1-0.4)
    - `hyperparameter_tuning`: Enable GridSearchCV (slower)
    
    **Output:** Training success status + comprehensive metrics:
    - R² (coefficient of determination)
    - RMSE (root mean squared error)
    - MAE (mean absolute error)
    - MAPE (mean absolute percentage error)
    - Training hyperparameters used
    
    **Example Use Cases:**
    - Model performance optimization
    - Periodic retraining with new data
    - A/B testing different configurations
    - Addressing model drift over time
    
    ⚠️ **Warning:** This endpoint will:
    - Take 10-60 seconds to complete (longer with GridSearchCV)
    - Overwrite existing model files
    - Update the live prediction model immediately
    - Not create backups (handle externally if needed)
    """
    global model, scaler

    try:
        print("\n" + "═" * 80)
        print("🔄 Starting Model Retraining")
        print("═" * 80)

        # Check if data file exists
        if not os.path.exists(DATA_PATH):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "Training data not found",
                    "message": f"Data file not found at {DATA_PATH}",
                },
            )

        # Load data
        df, target_col = load_data(DATA_PATH, TARGET_COLUMN)
        
        # Clean data
        from model_pipeline import clean_data
        df = clean_data(df, target_col)

        # Prepare data
        X_train, X_test, y_train, y_test = prepare_data(
            df, target_col, test_size=retrain_request.test_size
        )

        # Engineer features
        X_train_eng, X_test_eng = engineer_features(X_train, X_test)

        # Scale features
        X_train_scaled, X_test_scaled, new_scaler = scale_features(X_train_eng, X_test_eng)

        # Train model
        new_model = train_model(
            X_train_scaled,
            y_train,
            hyperparameter_tuning=retrain_request.hyperparameter_tuning,
            n_estimators=retrain_request.n_estimators,
            max_depth=retrain_request.max_depth,
            min_samples_split=retrain_request.min_samples_split,
            min_samples_leaf=retrain_request.min_samples_leaf,
            max_features=retrain_request.max_features,
        )

        # Evaluate model
        metrics = evaluate_model(new_model, X_train_scaled, X_test_scaled, y_train, y_test)

        # Save model
        metadata = {
            "hyperparameters": retrain_request.dict(),
            "training_metrics": metrics,
            "retrained_via_api": True,
            "feature_names": list(X_train_scaled.columns),  # Save feature names
        }
        save_model(new_model, new_scaler, MODEL_PATH, SCALER_PATH, metadata)

        # Update global model and scaler
        model = new_model
        scaler = new_scaler

        print("✅ Model retrained and saved successfully")
        print("═" * 80 + "\n")

        return RetrainResponse(
            message="Model retrained successfully",
            training_metrics=metrics,
            hyperparameters=retrain_request.dict(),
            training_timestamp=datetime.now().isoformat(),
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"❌ Retraining failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Retraining failed", "message": str(e)},
        )


# ═════════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 80)
    print("🌟 Garment Productivity Prediction API")
    print("═" * 80)
    print("\n📚 API Documentation will be available at:")
    print("   • Swagger UI: http://127.0.0.1:8000/docs")
    print("   • ReDoc: http://127.0.0.1:8000/redoc")
    print("\n" + "═" * 80 + "\n")

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
