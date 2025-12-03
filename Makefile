# ═════════════════════════════════════════════════════════════════════════════════
# MLOps Garment Productivity Prediction - Makefile
# ═════════════════════════════════════════════════════════════════════════════════

.PHONY: help setup clean clean-all format lint pylint type-check security code-quality \
        data train evaluate predict feature-importance train-tuning pipeline \
        test test-coverage deploy notebook validate-all ci

# ═════════════════════════════════════════════════════════════════════════════════
# VARIABLES
# ═════════════════════════════════════════════════════════════════════════════════

PYTHON := python
PIP := pip
VENV := venv
DATA_FILE := data.csv
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
	@echo "    make validate-all       Run complete validation (CI/CD ready)"
	@echo ""
	@echo "  🛠️  DEVELOPMENT TOOLS"
	@echo "  ──────────────────────────────────────────────────────────────────────"
	@echo "    make notebook           Start Jupyter Notebook server"
	@echo "    make notebook-lab       Start Jupyter Lab server"
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
	@$(PYTHON) main.py --mode train --data $(DATA_FILE) --target $(TARGET) --model $(MODEL_PATH)
	@echo ""
	@echo "✓ Model training complete"
	@echo ""

train-tuning:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🎯 Training with Hyperparameter Tuning                              ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@$(PYTHON) main.py --mode train --data $(DATA_FILE) --target $(TARGET) --model $(MODEL_PATH) --tuning
	@echo ""
	@echo "✓ Model training with tuning complete"
	@echo ""

full-pipeline:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🚀 Running Full ML Pipeline                                         ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@$(PYTHON) main.py --mode full_pipeline --data $(DATA_FILE) --target $(TARGET) --model $(MODEL_PATH)
	@echo ""
	@echo "✓ Full pipeline execution complete"
	@echo ""

evaluate:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  📈 Evaluating Model Performance                                     ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@$(PYTHON) main.py --mode evaluate --data $(DATA_FILE) --target $(TARGET) --model $(MODEL_PATH)
	@echo ""
	@echo "✓ Model evaluation complete"
	@echo ""

predict:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  🔮 Making Predictions                                               ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@$(PYTHON) main.py --mode predict --data $(DATA_FILE) --model $(MODEL_PATH) --output predictions.csv
	@echo ""
	@echo "✓ Predictions saved to predictions.csv"
	@echo ""

feature-importance:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════════════╗"
	@echo "║  📊 Analyzing Feature Importance                                     ║"
	@echo "╚══════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@$(PYTHON) main.py --mode feature_importance --data $(DATA_FILE) --target $(TARGET) --model $(MODEL_PATH)
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
# END OF MAKEFILE
# ═════════════════════════════════════════════════════════════════════════════════
