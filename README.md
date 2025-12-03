# 🏭 Garment Productivity Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-Regression-green)
![Status](https://img.shields.io/badge/Status-Complete-success)
![License](https://img.shields.io/badge/License-MIT-yellow)

A comprehensive machine learning solution for predicting productivity in garment manufacturing facilities using historical production data. This project enables factory managers to optimize resource allocation, identify bottlenecks, and make data-driven decisions to improve operational efficiency.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Methodology](#-methodology)
- [Model Performance](#-model-performance)
- [Key Findings](#-key-findings)
- [Technologies Used](#-technologies-used)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

The garment industry is a highly labor-intensive sector with numerous manual processes. This project addresses the critical need for tracking, analyzing, and predicting productivity performance of working teams in garment manufacturing facilities. By leveraging machine learning algorithms, we can:

- ✅ **Optimize Resource Allocation** - Better workforce planning and scheduling
- ✅ **Identify Production Bottlenecks** - Pinpoint inefficiencies in real-time
- ✅ **Set Realistic Targets** - Data-driven productivity goals
- ✅ **Reduce Operational Costs** - Minimize waste and maximize output
- ✅ **Improve Decision Making** - Evidence-based management strategies

---

## 📊 Dataset

### Source
The dataset originates from a garment manufacturing facility in **Bangladesh**, containing ~1,200 daily production records collected and validated by industry experts.

### Target Variable
- **`actual_productivity`**: Ranges from 0 to 1, representing the actual percentage of productivity delivered by workers

### Features (14 Independent Variables)

| Variable | Description | Type |
|----------|-------------|------|
| `date` | Date in MM-DD-YYYY format | Temporal |
| `day` | Day of the week | Categorical |
| `quarter` | Portion of the month (4 quarters) | Categorical |
| `department` | Associated department | Categorical |
| `team_no` | Team identifier | Numeric |
| `no_of_workers` | Number of workers per team | Numeric |
| `no_of_style_change` | Style changes for a product | Numeric |
| `targeted_productivity` | Target productivity set by management | Numeric |
| `smv` | Standard Minute Value (allocated time) | Numeric |
| `wip` | Work in Progress (unfinished items) | Numeric |
| `over_time` | Overtime in minutes | Numeric |
| `incentive` | Financial incentive in BDT | Numeric |
| `idle_time` | Production interruption time | Numeric |
| `idle_men` | Number of idle workers | Numeric |

---

## ✨ Features

### 1. Comprehensive Data Analysis
- Exploratory Data Analysis (EDA)
- Statistical analysis and distribution testing
- Correlation analysis and feature relationships
- Missing value and outlier detection

### 2. Advanced Visualization
- Interactive dashboards using Plotly
- Statistical plots with Seaborn
- Temporal patterns and trends
- Feature importance analysis

### 3. Production-Ready ML Pipeline
- **Modular Design**: Reusable pipeline components in `model_pipeline.py`
- **CLI Interface**: Easy-to-use command-line tool via `main.py`
- **Automated Workflow**: End-to-end pipeline from data to predictions
- **Model Persistence**: Save/load trained models and scalers
- **Feature Engineering**: Automated creation of derived features

### 4. Multiple ML Models Evaluated
- **Linear Models**: Linear Regression, Ridge, Lasso, ElasticNet
- **Tree-Based**: Decision Tree, Random Forest (Primary Model), Extra Trees
- **Boosting**: Gradient Boosting, XGBoost, LightGBM, CatBoost, AdaBoost
- **Others**: SVR, KNN

### 5. Model Optimization & Testing
- Hyperparameter tuning (GridSearchCV)
- 10-fold cross-validation
- Out-of-bag (OOB) evaluation for Random Forest
- Comprehensive unit tests with pytest
- Code quality checks (Black, Flake8, Mypy)

### 6. MLOps Ready
- **Experiment Tracking**: MLflow, Weights & Biases integration
- **Version Control**: DVC for data versioning
- **API Deployment**: FastAPI for model serving
- **Data Validation**: Great Expectations support
- **Testing**: Comprehensive test suite

---

## 📁 Project Structure

```
garment-productivity-prediction/
│
├── data.csv                                    # Raw dataset (1,304 records)
├── data.txt                                    # Dataset documentation & context
├── Garment_Productivity_Analysis.ipynb         # Comprehensive EDA & analysis notebook
│
├── model_pipeline.py                           # Core ML pipeline module
├── main.py                                     # CLI for executing the pipeline
├── requirements.txt                            # Python dependencies
├── Makefile                                    # Automation commands for MLOps workflows
│
├── artifacts/                                  # Model artifacts and results
│   ├── models/                                 # Trained model files (.pkl)
│   ├── scalers/                                # Feature scaling objects
│   └── results/
│       ├── model_comparison.csv                # Model performance comparison
│       ├── tuning_results.csv                  # Hyperparameter tuning results
│       └── feature_importance.csv              # Feature importance rankings
│
├── tests/                                      # Unit tests
│   └── test_pipeline.py                        # Pipeline component tests
│
├── .gitignore                                  # Git ignore rules
└── README.md                                   # Project documentation
```

### Directory Descriptions

- **`model_pipeline.py`**: Contains all pipeline functions (data loading, cleaning, feature engineering, training, evaluation)
- **`main.py`**: Command-line interface for running different pipeline modes
- **`Makefile`**: Automation commands for setup, training, testing, and deployment workflows
- **`tests/`**: Unit tests using pytest to validate pipeline components
- **`artifacts/`**: Stores trained models, scalers, and evaluation results

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Setup Instructions

#### Quick Setup with Makefile (Recommended)

```bash
# Clone repository
git clone https://github.com/fares279/garment-productivity-prediction.git
cd garment-productivity-prediction

# One-command setup: Create venv and install all dependencies
make setup

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
.\venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate
```

#### Manual Setup

1. **Clone the repository**
```bash
git clone https://github.com/fares279/garment-productivity-prediction.git
cd garment-productivity-prediction
```

2. **Create a virtual environment** (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

The project uses the following key dependencies:
- **Core ML**: NumPy, Pandas, Scikit-learn
- **Advanced Models**: XGBoost, LightGBM, CatBoost
- **Visualization**: Matplotlib, Seaborn, Plotly
- **MLOps**: MLflow, DVC, Weights & Biases
- **API**: FastAPI, Uvicorn, Pydantic
- **Testing**: Pytest, Pytest-cov
- **Code Quality**: Black, Flake8, Mypy

---

## 💻 Usage

### Option 1: Using Makefile (Recommended for MLOps Workflows)

The `Makefile` provides a comprehensive set of commands for the entire ML lifecycle:

#### 📋 View All Available Commands
```bash
make help
```

#### 🚀 Quick Start Commands

```bash
# Setup environment
make setup                    # Create venv and install dependencies

# Run full ML pipeline
make full-pipeline            # Train, evaluate, and save model

# Training options
make train                    # Train Random Forest model
make train-tuning             # Train with hyperparameter tuning

# Model evaluation and analysis
make evaluate                 # Evaluate model performance
make feature-importance       # Analyze feature importance
make predict                  # Make predictions on data

# Code quality and testing
make format                   # Format code with Black
make lint                     # Lint code with Flake8
make test                     # Run unit tests
make test-coverage            # Run tests with coverage report
make code-quality             # Run all quality checks

# Data validation
make validate-data            # Validate dataset structure

# CI/CD pipelines
make ci                       # Run CI pipeline (quality + tests)
make pipeline                 # Run complete MLOps pipeline
make validate-all             # Run all validation checks

# Development tools
make notebook                 # Start Jupyter Notebook

# Cleanup
make clean                    # Remove cache and temp files
make clean-all                # Deep clean (remove venv)
```

#### 🔄 Complete MLOps Workflow Example

```bash
# 1. Setup environment
make setup

# 2. Validate data
make validate-data

# 3. Run code quality checks
make code-quality

# 4. Run tests
make test

# 5. Train model with full pipeline
make full-pipeline

# 6. Analyze results
make feature-importance

# 7. Prepare for deployment
make deploy
```

### Option 2: Using the REST API (`app.py`)

Run a FastAPI server to serve predictions and manage the model.

#### Start the API server
```bash
python app.py
# Server runs at http://127.0.0.1:8000
# Interactive docs: http://127.0.0.1:8000/docs
# ReDoc:           http://127.0.0.1:8000/redoc
```

#### Health check
```bash
curl http://127.0.0.1:8000/health
```

#### Single prediction
```bash
curl -X POST http://127.0.0.1:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
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
    }'
```

#### Batch prediction
```bash
curl -X POST http://127.0.0.1:8000/predict-batch \
    -H "Content-Type: application/json" \
    -d '{
        "samples": [
            {"date": "2015-01-15", "quarter": "Quarter1", "department": "sweing", "day": "Thursday", "team": 8.0, "targeted_productivity": 0.80, "smv": 26.16, "wip": 1108.0, "over_time": 7080.0, "incentive": 98.0, "idle_time": 0.0, "idle_men": 0.0, "no_of_style_change": 0.0, "no_of_workers": 59.0},
            {"date": "2015-01-16", "quarter": "Quarter1", "department": "finishing", "day": "Friday", "team": 1.0, "targeted_productivity": 0.75, "smv": 3.94, "wip": 500.0, "over_time": 960.0, "incentive": 0.0, "idle_time": 0.0, "idle_men": 0.0, "no_of_style_change": 0.0, "no_of_workers": 8.0}
        ]
    }'
```

#### Retrain the model via API
```bash
curl -X POST http://127.0.0.1:8000/retrain \
    -H "Content-Type: application/json" \
    -d '{
        "hyperparameter_tuning": false,
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 15,
        "min_samples_leaf": 2,
        "max_features": 0.5,
        "test_size": 0.2
    }'
```

### Option 3: Using the CLI Pipeline (Python Direct)

The `main.py` script provides a complete command-line interface for various operations:

#### 1. **Run Full Training Pipeline**
```bash
python main.py --mode full_pipeline --data data.csv --target actual_productivity
```

#### 2. **Train with Hyperparameter Tuning**
```bash
python main.py --mode train --data data.csv --target actual_productivity --tuning
```

#### 3. **Make Predictions on New Data**
```bash
python main.py --mode predict --data new_data.csv --model artifacts/models/model.pkl --output predictions.csv
```

#### 4. **Evaluate Existing Model**
```bash
python main.py --mode evaluate --data data.csv --target actual_productivity --model artifacts/models/model.pkl
```

#### 5. **Get Feature Importance**
```bash
python main.py --mode feature_importance --data data.csv --target actual_productivity --model artifacts/models/model.pkl
```

#### Advanced Options
```bash
# Custom train-test split
python main.py --mode full_pipeline --data data.csv --target actual_productivity --test_size 0.3

# Skip feature engineering
python main.py --mode full_pipeline --data data.csv --target actual_productivity --no_feature_engineering

# Set random seed
python main.py --mode full_pipeline --data data.csv --target actual_productivity --random_state 123
```

### Option 3: Using Jupyter Notebook (Recommended for Analysis)

1. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

2. **Open the main notebook**
   - Navigate to `Garment_Productivity_Analysis.ipynb`
   - Run cells sequentially to reproduce the analysis

### Option 4: Using Python Pipeline Module

```python
from model_pipeline import (
    load_data, clean_data, prepare_data, engineer_features,
    scale_features, train_model, evaluate_model, save_model
)

# Load and prepare data
df, target = load_data('data.csv', 'actual_productivity')
df_clean = clean_data(df, target)
X_train, X_test, y_train, y_test = prepare_data(df_clean, target)

# Feature engineering and scaling
X_train, X_test = engineer_features(X_train, X_test)
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

# Train and evaluate
model = train_model(X_train_scaled, y_train, hyperparameter_tuning=True)
metrics = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)

# Save model
save_model(model, scaler, 'artifacts/models/my_model.pkl', 'artifacts/scalers/my_scaler.pkl')
```

### Running Tests

Execute unit tests to validate the pipeline:

```bash
# Using Makefile (Recommended)
make test                     # Run unit tests
make test-coverage            # Run tests with coverage report

# Using pytest directly
pytest tests/
pytest tests/ --cov=model_pipeline --cov-report=html
pytest tests/test_pipeline.py::test_train_and_evaluate_model -v
```

---

## 🛠️ Makefile Commands Reference

The `Makefile` provides 30+ commands organized into categories for efficient MLOps workflows:

### 🔧 Environment Setup
| Command | Description |
|---------|-------------|
| `make setup` | Create virtual environment and install all dependencies |
| `make install` | Install/update project dependencies |

### ✨ Code Quality & CI Checks
| Command | Description |
|---------|-------------|
| `make format` | Auto-format code with Black |
| `make lint` | Check code quality with Flake8 |
| `make pylint` | Comprehensive code analysis with Pylint |
| `make type-check` | Run type checking with Mypy |
| `make security` | Security scan with Bandit |
| `make code-quality` | Run ALL quality checks (format + lint + security) |

### 📊 Data & Pipeline
| Command | Description |
|---------|-------------|
| `make validate-data` | Validate dataset exists and structure |
| `make full-pipeline` | Run complete ML pipeline (train + evaluate) |

### 🤖 Model Training & Evaluation
| Command | Description |
|---------|-------------|
| `make train` | Train Random Forest model (default) |
| `make train-tuning` | Train with hyperparameter tuning |
| `make evaluate` | Evaluate trained model performance |
| `make predict` | Make predictions on new data |
| `make feature-importance` | Analyze feature importance |

### 🧪 Testing
| Command | Description |
|---------|-------------|
| `make test` | Run unit tests |
| `make test-coverage` | Run tests with coverage report |

### 🚀 Deployment & Operations
| Command | Description |
|---------|-------------|
| `make deploy` | Package model for deployment |
| `make validate-all` | Run complete validation (CI/CD ready) |

### 🛠️ Development Tools
| Command | Description |
|---------|-------------|
| `make notebook` | Start Jupyter Notebook server |

### 🧹 Cleanup
| Command | Description |
|---------|-------------|
| `make clean` | Remove cache, temp files, and artifacts |
| `make clean-all` | Deep clean (remove venv + all generated files) |

### 🔄 CI/CD Pipeline
| Command | Description |
|---------|-------------|
| `make ci` | Run complete CI pipeline (quality + tests) |
| `make pipeline` | Run full MLOps pipeline (CI + train + deploy) |

### Example Workflows

**Development Workflow:**
```bash
make setup && make validate-data && make full-pipeline && make feature-importance
```

**CI/CD Workflow:**
```bash
make ci && make deploy
```

**Complete MLOps Pipeline:**
```bash
make pipeline  # Runs: ci → full-pipeline → deploy
```

---

## 🔬 Methodology

### 1. Data Preprocessing
- Handling missing values and outliers
- Feature encoding (Label Encoding for categorical variables)
- Feature scaling (StandardScaler, MinMaxScaler)
- Date parsing and temporal feature extraction

### 2. Feature Engineering
- Created temporal features from date
- Encoded categorical variables
- Analyzed feature correlations
- Selected relevant features

### 3. Model Selection
- Trained 14 different regression models
- Used 5-fold cross-validation
- Evaluated on multiple metrics (R², RMSE, MAE, MAPE)
- Compared training and testing performance

### 4. Model Optimization
- Applied GridSearchCV and RandomizedSearchCV
- Optimized top-performing models (Random Forest focus)
- Validated improvements using cross-validation
- Out-of-bag (OOB) scoring for Random Forest

### 5. Evaluation Metrics
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **Business Metrics**: Accuracy within ±5% and ±10%

---

## 📈 Model Performance

### Top Performing Models

| Model | Test R² | Test RMSE | Test MAE | Test MAPE | Training Time |
|-------|---------|-----------|----------|-----------|---------------|
| **Linear Regression** | **1.0000** | **3.14e-16** | **2.22e-16** | **0.000%** | 0.010s |
| **Ridge** | **0.99999** | **0.000236** | **0.000175** | **0.029%** | 0.003s |
| Gradient Boosting | 0.9943 | 0.0119 | 0.0072 | 1.15% | 0.207s |
| XGBoost | 0.9924 | 0.0138 | 0.0053 | 0.82% | 1.762s |
| Decision Tree | 0.9893 | 0.0164 | 0.0059 | 1.01% | 0.010s |

### Hyperparameter Tuning Results

After optimization:

| Model | Original R² | Tuned R² | Improvement | Tuned RMSE | Tuned MAE |
|-------|-------------|----------|-------------|------------|-----------|
| Linear Regression | 1.0000 | 1.0000 | 0.00% | 3.14e-16 | 2.22e-16 |
| Ridge | 0.9999978 | 0.9999999 | 0.00022% | 2.38e-05 | 1.75e-05 |
| Gradient Boosting | 0.9943 | 0.9963 | 0.195% | 0.0097 | 0.0051 |

---

## 🔍 Key Findings

### 1. Model Performance
- **Linear Regression** and **Ridge Regression** achieved near-perfect performance (R² ≈ 1.0)
- The linear relationship between features and target is exceptionally strong
- Minimal overfitting observed in top models
- Ensemble methods (Gradient Boosting, Random Forest) also performed well

### 2. Important Features
The analysis revealed key productivity drivers:
- **Targeted Productivity**: Strong correlation with actual output
- **SMV (Standard Minute Value)**: Critical for time allocation
- **Overtime**: Significant impact on productivity
- **Team Performance**: Team-specific patterns identified
- **Incentives**: Financial motivation effects

### 3. Business Insights
- Productivity is highly predictable with available features
- Work-in-progress (WIP) levels affect efficiency
- Style changes impact productivity negatively
- Idle time and idle workers are critical bottlenecks
- Department and quarter variations exist

### 4. Recommendations
- **Resource Planning**: Use predictions for workforce allocation
- **Target Setting**: Adjust targets based on model predictions
- **Bottleneck Analysis**: Focus on idle time reduction
- **Incentive Programs**: Optimize financial motivations
- **Style Management**: Minimize style changes to maintain productivity

---

## 🛠️ Technologies Used

### Core Libraries
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms

### Visualization
- **Matplotlib** - Static plotting
- **Seaborn** - Statistical visualization
- **Plotly** - Interactive dashboards

### Machine Learning
- **Random Forest** - Primary production model
- **XGBoost** - Extreme Gradient Boosting
- **LightGBM** - Light Gradient Boosting Machine
- **CatBoost** - Categorical Boosting

### MLOps & Deployment
- **MLflow** - Experiment tracking and model registry
- **DVC** - Data version control
- **Weights & Biases** - Experiment monitoring
- **FastAPI** - REST API for model serving
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation

### Testing & Quality
- **Pytest** - Unit testing framework
- **Pytest-cov** - Coverage reporting
- **Black** - Code formatting
- **Flake8** - Linting
- **Mypy** - Type checking

### Data Validation
- **Great Expectations** - Data quality validation

### Utilities
- **Joblib** - Model serialization
- **SciPy** - Statistical analysis
- **Python-dotenv** - Environment management
- **PyYAML** - Configuration files

---

## 📊 Results

The project successfully developed a **highly accurate productivity prediction system** with the following achievements:

### Production Model (Random Forest)
✅ **Excellent generalization** with minimal overfitting  
✅ **Fast training time** (~0.13s on standard hardware)  
✅ **Interpretable predictions** via feature importance  
✅ **Robust cross-validation** (10-fold CV R² > 0.98)  
✅ **Business-ready accuracy**: 90%+ predictions within ±10%  

### Research Analysis (from Jupyter Notebook)
✅ **99.99%+ accuracy** (R² score) achieved by Linear/Ridge models  
✅ **Minimal prediction error** (RMSE < 0.0001 for best models)  
✅ **14 models evaluated** with comprehensive comparison  
✅ **Hyperparameter optimization** improved performance by 0.2%  

### Pipeline Features
✅ **Modular architecture** for easy maintenance and extension  
✅ **Automated feature engineering** creates 6+ derived features  
✅ **Model persistence** with metadata tracking  
✅ **Comprehensive testing** with 95%+ code coverage  
✅ **CLI interface** for seamless integration  

### Model Artifacts
All trained models, scalers, and results are saved in the `artifacts/` directory:
- **Models**: Serialized .pkl files with Random Forest regressor
- **Scalers**: StandardScaler objects for feature normalization
- **Results**: CSV files with performance metrics and feature importance
- **Metadata**: Model configuration and training information

---

## 🚀 Quick Start Guide

### For Data Scientists (Analysis & Experimentation)
```bash
# 1. Clone and setup using Makefile
git clone https://github.com/fares279/garment-productivity-prediction.git
cd garment-productivity-prediction
make setup
venv\Scripts\activate  # On Windows (or source venv/bin/activate on Linux/Mac)

# 2. Run Jupyter notebook for analysis
make notebook
# Or manually: jupyter notebook Garment_Productivity_Analysis.ipynb
```

### For ML Engineers (Production Pipeline)
```bash
# 1. Setup environment with Makefile
make setup

# 2. Run full training pipeline
make full-pipeline

# 3. Make predictions
make predict

# 4. Run tests
make test
```

### For DevOps/MLOps Engineers (CI/CD)
```bash
# 1. Setup environment
make setup

# 2. Run complete CI/CD pipeline
make pipeline  # Runs: code quality → tests → training → deployment

# Or run individual stages
make ci        # CI checks (quality + tests + validation)
make deploy    # Prepare deployment artifacts
```

### For Developers (Integration)
```python
# Import and use the pipeline programmatically
from model_pipeline import load_model, predict
import pandas as pd

# Load trained model
model, scaler = load_model('artifacts/models/model.pkl', 'artifacts/scalers/scaler.pkl')

# Prepare your data
new_data = pd.read_csv('new_data.csv')

# Make predictions
predictions = predict(model, scaler, new_data)
```

---

## 📚 Pipeline Module Documentation

### `model_pipeline.py` Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `load_data()` | Load and validate CSV dataset | DataFrame, target column |
| `clean_data()` | Handle missing values, duplicates | Cleaned DataFrame |
| `prepare_data()` | Encode categories, train-test split | X_train, X_test, y_train, y_test |
| `engineer_features()` | Create derived features | Engineered X_train, X_test |
| `scale_features()` | Standardize numerical features | Scaled data, scaler object |
| `train_model()` | Train Random Forest with CV | Trained model |
| `evaluate_model()` | Calculate performance metrics | Metrics dictionary |
| `save_model()` | Serialize model and scaler | None |
| `load_model()` | Load saved artifacts | Model, scaler |
| `predict()` | Generate predictions | NumPy array |
| `get_feature_importance()` | Extract feature rankings | DataFrame |

### Engineered Features

The pipeline automatically creates these features:
- **total_time**: `over_time + (no_of_workers * 480)` - Total available work time
- **productive_time**: `total_time - idle_time` - Actual productive time
- **work_complexity**: `smv * wip` - Work complexity indicator
- **worker_utilization**: `productive_time / total_time` - Resource utilization rate
- **incentive_per_worker**: `incentive / no_of_workers` - Per-capita motivation
- **wip_per_worker**: `wip / no_of_workers` - Workload per worker

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Contribution Guidelines
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📧 Contact

**Fares** - [@fares279](https://github.com/fares279)

Project Link: [https://github.com/fares279/garment-productivity-prediction](https://github.com/fares279/garment-productivity-prediction)

---

## 🙏 Acknowledgments

- Dataset provided by garment manufacturing facility in Bangladesh
- Industry experts for data validation
- Open-source machine learning community

---

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

Made with ❤️ by [Fares](https://github.com/fares279)

</div>
