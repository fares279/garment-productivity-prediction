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

### 3. Multiple ML Models Evaluated
- **Linear Models**: Linear Regression, Ridge, Lasso, ElasticNet
- **Tree-Based**: Decision Tree, Random Forest, Extra Trees
- **Boosting**: Gradient Boosting, XGBoost, LightGBM, CatBoost, AdaBoost
- **Others**: SVR, KNN

### 4. Model Optimization
- Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- Cross-validation strategies
- Feature engineering
- Model comparison and selection

---

## 📁 Project Structure

```
garment-productivity-prediction/
│
├── data.csv                                    # Raw dataset
├── data.txt                                    # Dataset documentation
├── Garment_Productivity_Analysis.ipynb         # Main analysis notebook
│
├── artifacts/                                  # Model artifacts and results
│   ├── models/                                 # Trained model files
│   ├── scalers/                                # Feature scaling objects
│   └── results/
│       ├── model_comparison.csv                # Model performance comparison
│       └── tuning_results.csv                  # Hyperparameter tuning results
│
└── README.md                                   # Project documentation
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/fares279/garment-productivity-prediction.git
cd garment-productivity-prediction
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install pandas numpy matplotlib seaborn plotly scipy scikit-learn xgboost lightgbm catboost jupyter
```

Or create a `requirements.txt` with:
```
pandas
numpy
matplotlib
seaborn
plotly
scipy
scikit-learn
xgboost
lightgbm
catboost
jupyter
```

Then install:
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Running the Analysis

1. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

2. **Open the main notebook**
   - Navigate to `Garment_Productivity_Analysis.ipynb`
   - Run cells sequentially to reproduce the analysis

3. **Access the dataset**
   - The dataset is located in `data.csv`
   - Documentation available in `data.txt`

### Notebook Structure

The notebook is organized into the following sections:

1. **Environment Setup** - Library imports and configuration
2. **Data Loading & Inspection** - Initial data exploration
3. **Exploratory Data Analysis** - Statistical analysis and visualization
4. **Data Preprocessing** - Cleaning, encoding, and scaling
5. **Feature Engineering** - Creating new features
6. **Model Training** - Training multiple ML algorithms
7. **Model Evaluation** - Performance comparison
8. **Hyperparameter Tuning** - Optimization of best models
9. **Results & Insights** - Final model selection and conclusions

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

### 4. Hyperparameter Tuning
- Applied GridSearchCV and RandomizedSearchCV
- Optimized top-performing models
- Validated improvements using cross-validation

### 5. Evaluation Metrics
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error

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
- **XGBoost** - Extreme Gradient Boosting
- **LightGBM** - Light Gradient Boosting Machine
- **CatBoost** - Categorical Boosting

### Statistics
- **SciPy** - Statistical analysis and testing

---

## 📊 Results

The project successfully developed a **highly accurate productivity prediction model** with the following achievements:

✅ **99.99%+ accuracy** (R² score) on test data  
✅ **Minimal prediction error** (RMSE < 0.0001)  
✅ **Fast inference time** for real-time predictions  
✅ **Robust cross-validation** performance  
✅ **Actionable insights** for management  

### Model Artifacts
All trained models, scalers, and results are saved in the `artifacts/` directory for deployment and further analysis.

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
