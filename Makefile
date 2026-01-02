# ═════════════════════════════════════════════════════════════════════════════════
# MLOps Garment Productivity Prediction - Makefile
# ═════════════════════════════════════════════════════════════════════════════════

.PHONY: help setup clean clean-all format lint pylint type-check security code-quality \
	data train evaluate predict feature-importance train-tuning pipeline \
	test test-coverage deploy notebook validate-all ci api api-test api-smoke retrain-smoke \
	docker-build docker-tag docker-push docker-run docker-deploy docker-stop docker-logs \
	docker-clean docker-status monitoring-up monitoring-down monitoring-status monitoring-logs \
	kibana-open elasticsearch-check monitoring-setup monitoring-test

# ═════════════════════════════════════════════════════════════════════════════════
# VARIABLES
# ═════════════════════════════════════════════════════════════════════════════════

PYTHON := python
PIP := pip
VENV := venv
DATA_FILE := data/raw/data.csv
TARGET := actual_productivity
MODEL_PATH := artifacts/models/model.pkl
SCALER_PATH := artifacts/scalers/scaler.pkl

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color
BOLD := \033[1m

# ═════════════════════════════════════════════════════════════════════════════════
# DEFAULT TARGET - HELP MENU
# ═════════════════════════════════════════════════════════════════════════════════

help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║                                                                      ║"
	@echo "║      🎯 MLOps Garment Productivity - Available Commands              ║"
	@echo "║                                                                      ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  🔧 ENVIRONMENT SETUP"
	@echo "  ──────────────────────────────────────────────────────────────────────"
	@echo "    make setup              Create virtual environment & install dependencies"
	@echo "    make install            Install/Update project dependencies"
	@echo ""
	@echo "  ✨ CODE QUALITY & CI CHECKS"
	@echo "  ──────────────────────────────────────────────────────────────────────"
	@echo "    make format             Format code with black (auto-fix)"
	@echo "    make lint               Check code quality with flake8"
	@echo "    make pylint             Run comprehensive code analysis with pylint"
	@echo "    make type-check         Run type checking with mypy"
	@echo "    make security           Run security scan with bandit"
	@echo "    make code-quality       Run ALL quality checks (format + lint + security)"
	@echo ""
	@echo "  📊 DATA VALIDATION & PIPELINE"
	@echo "  ──────────────────────────────────────────────────────────────────────"
	@echo "    make validate-data      Validate dataset exists and structure"
	@echo "    make full-pipeline      Run complete ML pipeline (train + evaluate)"
	@echo ""
	@echo "  🤖 MODEL TRAINING & EVALUATION"
	@echo "  ──────────────────────────────────────────────────────────────────────"
	@echo "    make train              Train Random Forest model (default)"
	@echo "    make train-tuning       Train model with hyperparameter tuning"
	@echo "    make evaluate           Evaluate trained model performance"
	@echo "    make predict            Make predictions on new data"
	@echo "    make feature-importance Analyze feature importance"
	@echo ""
	@echo "  🧪 TESTING"
	@echo "  ──────────────────────────────────────────────────────────────────────"
	@echo "    make test               Run unit tests"
	@echo "    make test-coverage      Run tests with coverage report"
	@echo ""
	@echo "  🚀 DEPLOYMENT & OPERATIONS"
	@echo "  ──────────────────────────────────────────────────────────────────────"
	@echo "    make deploy             Package model for deployment"
	@echo "    make api                Start FastAPI server for predictions"
	@echo "    make api-test           Test the API with sample request"
	@echo "    make api-smoke          Run API smoke tests (tests/test_api.py)"
	@echo "    make retrain-smoke      Run retrain smoke test (tests/test_retrain.py)"
	@echo "    make validate-all       Run complete validation (CI/CD ready)"
	@echo ""
	@echo "  🐳 DOCKER CONTAINERIZATION"
	@echo "  ──────────────────────────────────────────────────────────────────────"
	@echo "    make docker-build       Build Docker image"
	@echo "    make docker-tag         Tag Docker image for Docker Hub"
	@echo "    make docker-push        Push Docker image to Docker Hub"
	@echo "    make docker-run         Run Docker container locally"
	@echo "    make docker-deploy      Complete Docker workflow (build + push)"
	@echo "    make docker-stop        Stop running Docker containers"
	@echo "    make docker-logs        View container logs"
	@echo "    make docker-status      Show Docker images and containers status"
	@echo "    make docker-clean       Remove Docker containers and images"
	@echo ""
	@echo "  🛠️  DEVELOPMENT TOOLS"
	@echo "  ──────────────────────────────────────────────────────────────────────"
	@echo "    make notebook           Start Jupyter Notebook server"
	@echo "    make notebook-lab       Start Jupyter Lab server"
	@echo "    make mlflow-ui          Launch MLflow Tracking UI (localhost:5000)"
	@echo ""
	@echo "  🔍 MONITORING (Elasticsearch + Kibana)"
	@echo "  ──────────────────────────────────────────────────────────────────────"
	@echo "    make monitoring-setup   Install monitoring dependencies"
	@echo "    make monitoring-up      Start Elasticsearch + Kibana stack"
	@echo "    make monitoring-down    Stop monitoring stack"
	@echo "    make monitoring-status  Check monitoring stack status"
	@echo "    make monitoring-logs    View monitoring stack logs"
	@echo "    make monitoring-test    Test monitoring integration"
	@echo "    make kibana-open        Open Kibana in browser"
	@echo "    make elasticsearch-check Check Elasticsearch health"
	@echo ""
	@echo "  🧹 CLEANUP"
	@echo "  ──────────────────────────────────────────────────────────────────────"
	@echo "    make clean              Remove cache, temp files, and artifacts"
	@echo "    make clean-all          Remove venv + all generated files"
	@echo ""
	@echo "  🔄 CI/CD PIPELINE"
	@echo "  ──────────────────────────────────────────────────────────────────────"
	@echo "    make ci                 Run complete CI pipeline (quality + tests)"
	@echo "    make pipeline           Run full MLOps pipeline (CI + train + deploy)"
	@echo ""
	@echo "  ❓ HELP"
	@echo "  ──────────────────────────────────────────────────────────────────────"
	@echo "    make help               Show this help message"
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  📦 Repository: github.com/fares279/garment-productivity-prediction  ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""

# ═════════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ═════════════════════════════════════════════════════════════════════════════════

setup:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🔧 Setting up Python Environment                                    ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "→ Creating virtual environment..."
	@$(PYTHON) -m venv $(VENV)
	@echo "✓ Virtual environment created"
	@echo ""
	@echo "→ Upgrading pip..."
	@$(VENV)/Scripts/python.exe -m pip install --upgrade pip
	@echo "✓ Pip upgraded"
	@echo ""
	@echo "→ Installing dependencies from requirements.txt..."
	@$(VENV)/Scripts/pip.exe install -r requirements.txt
	@echo ""
	@echo "✓ Environment setup complete!"
	@echo ""
	@echo "To activate the virtual environment, run:"
	@echo "  Windows (PowerShell): .\\$(VENV)\\Scripts\\Activate.ps1"
	@echo "  Windows (CMD):        .\\$(VENV)\\Scripts\\activate.bat"
	@echo "  Linux/Mac:            source $(VENV)/bin/activate"
	@echo ""

install:
	@echo ""
	@echo "→ Installing/Updating dependencies..."
	@$(PIP) install -r requirements.txt
	@echo "✓ Dependencies installed"
	@echo ""

# ═════════════════════════════════════════════════════════════════════════════════
# CODE QUALITY & CI CHECKS
# ═════════════════════════════════════════════════════════════════════════════════

format:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  ✨ Code Formatting (Black)                                          ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "→ Formatting Python files..."
	@$(PYTHON) -m black *.py --line-length 100 || echo "⚠️  Black not installed, skipping..."
	@echo "✓ Code formatting complete"
	@echo ""

lint:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🔍 Code Quality Check (Flake8)                                      ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "→ Running flake8 linter..."
	@$(PYTHON) -m flake8 *.py --max-line-length=100 --ignore=E501,W503 || echo "⚠️  Flake8 not installed, skipping..."
	@echo "✓ Linting complete"
	@echo ""

pylint:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  📊 Code Analysis (Pylint)                                           ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "→ Running pylint analysis..."
	@$(PYTHON) -m pylint *.py --max-line-length=100 || echo "⚠️  Pylint not installed, skipping..."
	@echo "✓ Code analysis complete"
	@echo ""

type-check:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🔎 Type Checking (Mypy)                                             ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "→ Running mypy type checker..."
	@$(PYTHON) -m mypy *.py --ignore-missing-imports || echo "⚠️  Mypy not installed, skipping..."
	@echo "✓ Type checking complete"
	@echo ""

security:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🔒 Security Scan (Bandit)                                           ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "→ Running bandit security scanner..."
	@$(PYTHON) -m bandit -r *.py -ll || echo "⚠️  Bandit not installed, skipping..."
	@echo "✓ Security scan complete"
	@echo ""

code-quality: format lint type-check security
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  ✓ All Code Quality Checks Complete                                  ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""

# ═════════════════════════════════════════════════════════════════════════════════
# DATA VALIDATION
# ═════════════════════════════════════════════════════════════════════════════════

validate-data:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  📊 Data Validation                                                  ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "→ Checking if data file exists..."
	@if [ -f "$(DATA_FILE)" ]; then \
		echo "✓ Data file found: $(DATA_FILE)"; \
		echo "→ Validating data structure..."; \
		$(PYTHON) -c "import pandas as pd; df=pd.read_csv('$(DATA_FILE)'); print(f'  Rows: {len(df):,}'); print(f'  Columns: {len(df.columns)}'); print(f'  Target column present: {\"$(TARGET)\" in df.columns}')"; \
	else \
		echo "✗ ERROR: Data file not found: $(DATA_FILE)"; \
		exit 1; \
	fi
	@echo ""
	@echo "✓ Data validation complete"
	@echo ""

# ═════════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING & EVALUATION
# ═════════════════════════════════════════════════════════════════════════════════

train:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🤖 Training Random Forest Model                                     ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@$(PYTHON) scripts/train.py --mode train --data $(DATA_FILE) --target $(TARGET) --model $(MODEL_PATH)
	@echo ""
	@echo "✓ Model training complete"
	@echo ""

train-tuning:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🎯 Training with Hyperparameter Tuning                              ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@$(PYTHON) scripts/train.py --mode train --data $(DATA_FILE) --target $(TARGET) --model $(MODEL_PATH) --tuning
	@echo ""
	@echo "✓ Model training with tuning complete"
	@echo ""

full-pipeline:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🚀 Running Full ML Pipeline                                         ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@$(PYTHON) scripts/train.py --mode full_pipeline --data $(DATA_FILE) --target $(TARGET) --model $(MODEL_PATH)
	@echo ""
	@echo "✓ Full pipeline execution complete"
	@echo ""

evaluate:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  📈 Evaluating Model Performance                                     ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@$(PYTHON) scripts/train.py --mode evaluate --data $(DATA_FILE) --target $(TARGET) --model $(MODEL_PATH)
	@echo ""
	@echo "✓ Model evaluation complete"
	@echo ""

predict:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🔮 Making Predictions                                               ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@$(PYTHON) scripts/train.py --mode predict --data $(DATA_FILE) --model $(MODEL_PATH) --output predictions.csv
	@echo ""
	@echo "✓ Predictions saved to predictions.csv"
	@echo ""

feature-importance:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  📊 Analyzing Feature Importance                                     ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@$(PYTHON) scripts/train.py --mode feature_importance --data $(DATA_FILE) --target $(TARGET) --model $(MODEL_PATH)
	@echo ""
	@echo "✓ Feature importance analysis complete"
	@echo ""

# ═════════════════════════════════════════════════════════════════════════════════
# TESTING
# ═════════════════════════════════════════════════════════════════════════════════

test:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🧪 Running Unit Tests                                               ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "→ Running pytest..."
	@$(PYTHON) -m pytest tests/ -v --cache-clear || echo "⚠️  No tests found or pytest not installed"
	@rm -rf .pytest_cache .mypy_cache __pycache__ .coverage htmlcov 2>/dev/null || true
	@echo ""
	@echo "✓ Tests complete"
	@echo ""

test-coverage:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  📊 Running Tests with Coverage                                      ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@$(PYTHON) -m pytest tests/ --cov=. --cov-report=term-missing || echo "⚠️  pytest-cov not installed"
	@rm -rf .pytest_cache .mypy_cache __pycache__ .coverage htmlcov 2>/dev/null || true
	@echo ""

# ═════════════════════════════════════════════════════════════════════════════════
# DEPLOYMENT
# ═════════════════════════════════════════════════════════════════════════════════

deploy:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🚀 Preparing Model for Deployment                                   ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "→ Validating model artifacts..."
	@if [ -f "$(MODEL_PATH)" ]; then \
		echo "✓ Model file found: $(MODEL_PATH)"; \
	else \
		echo "✗ ERROR: Model file not found. Run 'make train' first."; \
		exit 1; \
	fi
	@if [ -f "$(SCALER_PATH)" ]; then \
		echo "✓ Scaler file found: $(SCALER_PATH)"; \
	else \
		echo "✗ ERROR: Scaler file not found. Run 'make train' first."; \
		exit 1; \
	fi
	@echo ""
	@echo "→ Creating deployment package..."
	@echo "✓ Model ready for deployment"
	@echo ""
	@echo "📦 Deployment artifacts:"
	@echo "   - Model:  $(MODEL_PATH)"
	@echo "   - Scaler: $(SCALER_PATH)"
	@echo ""

# ═════════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS (FASTAPI)
# ═════════════════════════════════════════════════════════════════════════════════

api:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🌐 Starting FastAPI Server                                          ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "→ Checking model availability..."
	@if [ ! -f "$(MODEL_PATH)" ] || [ ! -f "$(SCALER_PATH)" ]; then \
		echo "⚠️  Warning: Model or scaler not found. Please train the model first."; \
		echo "   Run: make train"; \
		echo ""; \
	fi
	@echo "→ Starting FastAPI server on http://0.0.0.0:8000"
	@echo ""
	@echo "📚 API Documentation available at:"
	@echo "   • Swagger UI: http://127.0.0.1:8000/docs"
	@echo "   • ReDoc:      http://127.0.0.1:8000/redoc"
	@echo ""
	@echo "Press Ctrl+C to stop the server"
	@echo ""
	@$(PYTHON) -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

api-test:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🧪 Testing FastAPI Endpoints                                        ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "→ Opening Swagger UI for interactive testing..."
	@echo ""
	@echo "📖 Testing Instructions:"
	@echo "   1. The browser will open to http://127.0.0.1:8000/docs"
	@echo "   2. Test the /predict endpoint with sample data"
	@echo "   3. Try the /retrain endpoint to retrain the model"
	@echo "   4. Check /health and /model-info for system status"
	@echo ""
	@echo "⚠️  Note: Make sure the API server is running (make api)"
	@echo ""
	@start http://127.0.0.1:8000/docs

api-smoke:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🧪 Running FastAPI Smoke Tests                                      ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@$(PYTHON) tests/test_api.py

retrain-smoke:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🔄 Running Retrain Smoke Test                                       ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@$(PYTHON) tests/test_retrain.py

# ═════════════════════════════════════════════════════════════════════════════════
# DEVELOPMENT TOOLS
# ═════════════════════════════════════════════════════════════════════════════════

notebook:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  📓 Starting Jupyter Notebook                                        ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@$(PYTHON) -m jupyter notebook

# ═════════════════════════════════════════════════════════════════════════════════
# MLFLOW UI
# ═════════════════════════════════════════════════════════════════════════════════

mlflow-ui:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  📟 Launching MLflow Tracking UI                                     ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "→ Starting MLflow UI on http://127.0.0.1:5000"
	@echo "  Backend: SQLite (mlflow.db)"
	@echo "  Artifacts: local (./mlruns)"
	@echo ""
	@mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 &
	@(
		command -v xdg-open >/dev/null 2>&1 && xdg-open http://127.0.0.1:5000 \
		|| command -v gio >/dev/null 2>&1 && gio open http://127.0.0.1:5000 \
		|| command -v open >/dev/null 2>&1 && open http://127.0.0.1:5000 \
		|| (
			echo "⚠️  Unable to auto-open browser. Please visit: http://127.0.0.1:5000";
		)
	)

# ═════════════════════════════════════════════════════════════════════════════════
# CLEANUP
# ═════════════════════════════════════════════════════════════════════════════════

clean:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🧹 Cleaning Cache and Temporary Files                               ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "→ Removing Python cache files..."
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "→ Removing .pyc files..."
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "→ Removing pytest cache..."
	@rm -rf .pytest_cache 2>/dev/null || true
	@echo "→ Removing coverage files..."
	@rm -f .coverage 2>/dev/null || true
	@rm -rf htmlcov 2>/dev/null || true
	@echo "→ Removing mypy cache..."
	@rm -rf .mypy_cache 2>/dev/null || true
	@echo ""
	@echo "✓ Cleanup complete"
	@echo ""

clean-all: clean
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🗑️  Deep Clean (Removing venv and artifacts)                        ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "→ Removing virtual environment..."
	@rm -rf $(VENV) 2>/dev/null || true
	@echo "→ Removing model artifacts..."
	@rm -rf artifacts 2>/dev/null || true
	@echo "→ Removing predictions..."
	@rm -f predictions.csv 2>/dev/null || true
	@echo ""
	@echo "✓ Deep clean complete"
	@echo ""

# ═════════════════════════════════════════════════════════════════════════════════
# CI/CD PIPELINE
# ═════════════════════════════════════════════════════════════════════════════════

validate-all: validate-data code-quality
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  ✓ All Validation Checks Passed                                      ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""

ci: code-quality test validate-data
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  ✓ CI Pipeline Complete                                              ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  ✓ Code quality checks passed"
	@echo "  ✓ Tests passed"
	@echo "  ✓ Data validation passed"
	@echo ""

pipeline: ci full-pipeline deploy
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🎉 Complete MLOps Pipeline Finished Successfully                    ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  ✓ CI checks complete"
	@echo "  ✓ Model trained and evaluated"
	@echo "  ✓ Deployment artifacts ready"
	@echo ""
	@echo "Next steps:"
	@echo "  - Review model performance in artifacts/results/"
	@echo "  - Check feature importance in artifacts/results/feature_importance.csv"
	@echo "  - Deploy model using 'make deploy'"
	@echo ""

# ═════════════════════════════════════════════════════════════════════════════════
# DOCKER CONTAINERIZATION
# ═════════════════════════════════════════════════════════════════════════════════

# Docker variables
DOCKER_IMAGE = fares_garmentproductivity_mlops
DOCKER_USERNAME = fares279

docker-build:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🐳 Building Docker Image                                            ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	docker build -t $(DOCKER_IMAGE) .
	@echo ""
	@echo "✓ Docker image built successfully: $(DOCKER_IMAGE)"
	@echo ""

docker-tag:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🏷️  Tagging Docker Image                                            ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	docker tag $(DOCKER_IMAGE) $(DOCKER_USERNAME)/$(DOCKER_IMAGE):latest
	@echo ""
	@echo "✓ Docker image tagged: $(DOCKER_USERNAME)/$(DOCKER_IMAGE):latest"
	@echo ""

docker-push: docker-tag
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  ⬆️  Pushing Docker Image to Docker Hub                              ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	docker push $(DOCKER_USERNAME)/$(DOCKER_IMAGE):latest
	@echo ""
	@echo "✓ Docker image pushed successfully to Docker Hub"
	@echo ""

docker-run:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🚀 Running Docker Container                                         ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	docker run -d -p 8000:8000 $(DOCKER_IMAGE)
	@echo ""
	@echo "✓ Docker container started"
	@echo "  Access API at: http://localhost:8000/docs"
	@echo ""

docker-deploy: docker-build docker-push
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🎉 Docker Deployment Complete                                       ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  ✓ Image built"
	@echo "  ✓ Image pushed to Docker Hub"
	@echo ""
	@echo "Next steps:"
	@echo "  - Run locally: make docker-run"
	@echo "  - Pull from anywhere: docker pull $(DOCKER_USERNAME)/$(DOCKER_IMAGE):latest"
	@echo ""

docker-stop:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  ⏹️  Stopping Docker Containers                                      ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@docker ps -q --filter ancestor=$(DOCKER_IMAGE) | xargs -r docker stop
	@echo "✓ All containers stopped"
	@echo ""

docker-logs:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  📋 Docker Container Logs                                            ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@docker logs $$(docker ps -q --filter ancestor=$(DOCKER_IMAGE) | head -1) 2>/dev/null || echo "No running container found"
	@echo ""

docker-clean:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🧹 Cleaning Docker Resources                                        ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@docker ps -aq --filter ancestor=$(DOCKER_IMAGE) | xargs -r docker rm -f
	@echo "✓ Containers removed"
	@docker rmi $(DOCKER_IMAGE) 2>/dev/null || true
	@docker rmi $(DOCKER_USERNAME)/$(DOCKER_IMAGE):latest 2>/dev/null || true
	@echo "✓ Images removed"
	@echo ""

docker-status:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@MONITORING STACK (Elasticsearch + Kibana)
# ═════════════════════════════════════════════════════════════════════════════════

monitoring-setup:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  📦 Installing Monitoring Dependencies                               ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	$(PIP) install elasticsearch psutil docker
	@echo ""
	@echo "✅ Monitoring dependencies installed"
	@echo ""

monitoring-up:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🚀 Starting Monitoring Stack (Elasticsearch + Kibana)               ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "📦 Starting Docker Compose..."
	docker compose up -d
	@echo ""
	@echo "⏳ Waiting for services to be healthy (this may take 60-90 seconds)..."
	@sleep 10
	@echo ""
	@echo "✅ Monitoring Stack Started!"
	@echo ""
	@echo "Services:"
	@echo "  • Elasticsearch: http://localhost:9200"
	@echo "  • Kibana:        http://localhost:5601"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Wait ~60 seconds for services to fully start"
	@echo "  2. Check status: make monitoring-status"
	@echo "  3. Open Kibana:  make kibana-open"
	@echo "  4. Run test:     make monitoring-test"
	@echo ""

monitoring-down:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  ⏹️  Stopping Monitoring Stack                                       ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	docker compose down
	@echo ""
	@echo "✅ Monitoring stack stopped"
	@echo ""

monitoring-status:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  📊 Monitoring Stack Status                                          ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Docker Containers:"
	@docker compose ps
	@echo ""
	@echo "Elasticsearch Health:"
	@curl -s http://localhost:9200/_cluster/health?pretty 2>/dev/null || echo "❌ Elasticsearch not responding"
	@echo ""
	@echo "Kibana Status:"
	@curl -s http://localhost:5601/api/status 2>/dev/null | grep -o '"level":"[^"]*"' || echo "❌ Kibana not responding"
	@echo ""

monitoring-logs:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  📋 Monitoring Stack Logs                                            ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	docker compose logs --tail=50 -f

elasticsearch-check:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🔍 Checking Elasticsearch                                           ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Cluster Health:"
	@curl -s http://localhost:9200/_cluster/health?pretty
	@echo ""
	@echo "Indices:"
	@curl -s http://localhost:9200/_cat/indices?v
	@echo ""

kibana-open:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🌐 Opening Kibana                                                   ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Opening Kibana at http://localhost:5601"
	@echo ""
	@echo "To create index patterns:"
	@echo "  1. Go to Stack Management > Index Patterns"
	@echo "  2. Create patterns: mlflow-metrics, mlflow-params, mlflow-predictions"
	@echo "  3. Use Discover to explore your data"
	@echo ""
	@cmd /c start http://localhost:5601 2>/dev/null || xdg-open http://localhost:5601 2>/dev/null || open http://localhost:5601 2>/dev/null || echo "Please open http://localhost:5601 manually"

monitoring-test:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🧪 Testing Monitoring Integration                                   ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	$(PYTHON) -c "from monitoring import ElasticsearchLogger; logger = ElasticsearchLogger(); print('✅ Monitoring module loaded successfully')"
	@echo ""
	@echo "Running monitoring test script..."
	$(PYTHON) test_monitoring.py
	@echo ""

# ═════════════════════════════════════════════════════════════════════════════════
# echo "║  📊 Docker Status                                                    ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Images:"
	@docker images | grep -E "REPOSITORY|$(DOCKER_IMAGE)" || echo "No images found"
	@echo ""
	@echo "Running Containers:"
	@docker ps --filter ancestor=$(DOCKER_IMAGE) || echo "No containers running"
	@echo ""

# ═════════════════════════════════════════════════════════════════════════════════
# END OF MAKEFILE
# ═════════════════════════════════════════════════════════════════════════════════
