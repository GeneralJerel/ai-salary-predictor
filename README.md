# AI Job Market Salary Prediction - Machine Learning Capstone
## Predictive Modeling for Data Science & AI Compensation Analysis

### Project Overview

This capstone project builds a machine learning model to predict annual salaries for AI and data science professionals based on factors such as experience level, job title, geographic location, company size, and employment type. The project demonstrates end-to-end machine learning workflow including data preprocessing, exploratory analysis, model training, and explainability analysis.

**Objective**: Predict salary in USD for AI/ML professionals to provide market insights for compensation planning and career guidance.

**Model Type**: Supervised Regression (predicting continuous salary values)

**Best Model**: Linear Regression with R² ≈ 0.41 (tuned Random Forest and XGBoost also evaluated)

---

### Dataset

**Source**: [Kaggle - Data Science Job Salaries](https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries)

**Dataset Details**:
- **Total Records**: 607 raw entries (565 after duplicate removal)
- **Time Period**: 2020-2022
- **Salary Range**: $2,859 - $600,000 USD
- **Geographic Coverage**: 50+ countries
- **Features**: 11 original columns (processed to 20+ engineered features)

**Key Columns**:
- `work_year`: Year of employment
- `experience_level`: EN (Entry), MI (Mid), SE (Senior), EX (Executive)
- `employment_type`: FT (Full-time), PT (Part-time), CT (Contractor), FL (Freelancer)
- `job_title`: Position type (50+ unique titles)
- `salary_in_usd`: Target variable (annual salary in USD)
- `remote_ratio`: 0 (on-site), 50 (hybrid), 100 (fully remote)
- `company_location`: Country code of company HQ
- `company_size`: S (Small), M (Medium), L (Large)

---

### Project Structure

```
ai-job-market-ml-capstone/
├── data/
│   ├── raw/
│   │   └── ds_salaries.csv              # Original Kaggle dataset
│   └── processed/
│       └── ds_salaries_clean.csv         # Cleaned data
├── src/
│   ├── data_processing.py               # Data loading & cleaning
│   ├── feature_engineering.py           # Feature transformations
│   └── train_models.py                  # Model training & evaluation
├── notebooks/
│   └── Jerel_Velarde_Pillar5_Capstone_AI_Job_Market.ipynb
├── models/
│   └── trained_models/
│       ├── linear_regression_model.joblib
│       ├── random_forest_model.joblib
│       └── xgboost_model.joblib
├── reports/
│   ├── 01_salary_distribution.png
│   ├── 02_salary_vs_experience.png
│   ├── 03_top_paying_jobs.png
│   ├── 04_remote_ratio_distribution.png
│   ├── 05_correlation_heatmap.png
│   ├── 06_pca_visualization.png
│   ├── 07_umap_visualization.png
│   ├── 08_actual_vs_predicted.png
│   ├── 09_shap_feature_importance.png
│   ├── 10_shap_summary_plot.png
│   └── 11_error_analysis.png
├── for_submission/
│   └── Jerel_Velarde_Pillar5_Capstone_AI_Job_Market.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

---

### How to Reproduce

#### 1. Environment Setup

```bash
# Clone or navigate to the project directory
cd ai-job-market-ml-capstone

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Run Data Processing

```bash
python src/data_processing.py
# Output: data/processed/ds_salaries_clean.csv
```

#### 3. Feature Engineering

```python
from src.feature_engineering import prepare_features
import pandas as pd

df = pd.read_csv('data/processed/ds_salaries_clean.csv')
X, y, df_engineered = prepare_features(df)
```

#### 4. Train Models

```python
from src.train_models import train_and_evaluate
from src.feature_engineering import prepare_features
import pandas as pd

df = pd.read_csv('data/processed/ds_salaries_clean.csv')
X, y, _ = prepare_features(df)
results = train_and_evaluate(X, y)
```

#### 5. Run Full Analysis Notebook

```bash
jupyter notebook notebooks/Jerel_Velarde_Pillar5_Capstone_AI_Job_Market.ipynb
```

The notebook contains all code cells for:
- Data loading and cleaning
- Exploratory data analysis
- Feature engineering
- Model training and evaluation
- SHAP explainability analysis
- Bias and fairness analysis

---

### Key Findings

#### Model Performance
| Model | RMSE | MAE | R² Score | CV R² |
|-------|------|-----|----------|-------|
| **Linear Regression** | **$53,248** | **$33,731** | **0.4098** | 0.4203 |
| Random Forest (Tuned) | $53,436 | $32,189 | 0.4056 | 0.4762 |
| XGBoost (Tuned) | $53,813 | $32,574 | 0.3972 | 0.4801 |

> **Note**: The relatively modest R² scores reflect the inherent difficulty of predicting salary from limited categorical features. The dataset lacks key predictors like education, years of experience, company prestige, and total compensation details. All three models were hyperparameter-tuned using RandomizedSearchCV with 5-fold cross-validation.

#### Top Salary Drivers (SHAP Analysis)
1. **Experience Level** - Most significant predictor (EN→EX increases salary ~$50K-$80K)
2. **Job Title** - Specialization commands premium (e.g., ML Engineers, Data Engineers)
3. **US Location** - US-based positions ~15-20% higher salary on average
4. **Remote Ratio** - Fully remote roles slightly lower than on-site
5. **Company Size** - Large companies offer 10-15% premium

#### Geographic Insights
- **North America** (US/Canada): Highest average ($130K-$160K)
- **Europe**: Strong market ($80K-$120K)
- **Asia**: Growing market with salary variation ($40K-$100K)
- **South America**: Emerging market ($30K-$80K)

#### Experience Level Impact
- **Entry Level**: $40K-$65K
- **Mid Level**: $70K-$110K
- **Senior**: $110K-$170K
- **Executive**: $140K-$250K+

---

### Model Details

#### Data Preprocessing
- ✅ Removed 42 duplicate rows (607 → 565 records)
- ✅ Handled missing values (none found)
- ✅ Dropped non-predictive columns (Unnamed: 0)
- ✅ Engineered 9+ new features

#### Feature Engineering
1. **Ordinal Encoding**: experience_level (EN=1, MI=2, SE=3, EX=4)
2. **One-Hot Encoding**: employment_type (4 categories)
3. **One-Hot Encoding**: company_size (3 categories)
4. **Label Encoding**: job_title (grouped rare titles <5 occurrences as "Other")
5. **Label Encoding**: continent (mapped from country codes)
6. **Binary Feature**: is_us (1 if US-based, 0 otherwise)
7. **Dimensionality Reduction**: PCA & UMAP visualization

#### Train-Test Split
- Training: 80% (452 samples)
- Testing: 20% (113 samples)
- Random state: 42 (reproducibility)
- Feature scaling: StandardScaler applied for Linear Regression (fit on train only)

#### Hyperparameter Tuning

All tree-based models were tuned using **RandomizedSearchCV** (50 iterations, 5-fold CV):

**Random Forest** (Tuned):
- n_estimators: 100
- max_depth: 20
- min_samples_split: 10
- min_samples_leaf: 1
- max_features: sqrt

**XGBoost** (Tuned):
- n_estimators: 300
- max_depth: 10
- learning_rate: 0.01
- subsample: 0.6
- colsample_bytree: 0.6
- min_child_weight: 7
- reg_alpha: 0.01
- reg_lambda: 1.0

---

### Explainability & Fairness

#### SHAP Analysis
- TreeExplainer for best tree-based model
- Feature importance ranking
- Individual prediction explanations
- Summary plots (bar + beeswarm)

#### Bias Analysis
- ✅ Evaluated across experience levels (EN, MI, SE, EX)
- ✅ Evaluated across continents (NA, EU, Asia, etc.)
- ✅ Analyzed prediction errors by demographic group
- ✅ Fairness metrics: Statistical Parity, Equalized Odds, Calibration

#### Key Fairness Findings
- Model shows consistent RMSE across experience levels
- Some geographic bias for underrepresented regions
- Overall prediction bias (mean error) < 2%
- Potential perpetuation of historical salary gaps

---

### Visualizations

The project generates 11 comprehensive visualizations:

1. **Salary Distribution** - Linear and log scale histograms
   ![Salary Distribution](reports/01_salary_distribution.png)
2. **Salary vs Experience** - Boxplot by career stage
   ![Salary vs Experience](reports/02_salary_vs_experience.png)
3. **Top Paying Jobs** - Bar chart of highest average salaries
   ![Top Paying Jobs](reports/03_top_paying_jobs.png)
4. **Remote Work Distribution** - Count by work arrangement
   ![Remote Work Distribution](reports/04_remote_ratio_distribution.png)
5. **Correlation Heatmap** - Feature relationships
   ![Correlation Heatmap](reports/05_correlation_heatmap.png)
6. **PCA Visualization** - 2D projection of features
   ![PCA Visualization](reports/06_pca_visualization.png)
7. **UMAP Visualization** - Non-linear dimensionality reduction
   ![UMAP Visualization](reports/07_umap_visualization.png)
8. **Actual vs Predicted** - 3-model comparison scatter plots
   ![Actual vs Predicted](reports/08_actual_vs_predicted.png)
9. **SHAP Feature Importance** - Mean impact ranking
   ![SHAP Feature Importance](reports/09_shap_feature_importance.png)
10. **SHAP Summary Plot** - Feature impact direction and magnitude
    ![SHAP Summary Plot](reports/10_shap_summary_plot.png)
11. **Error Analysis** - Residuals distribution and bias
    ![Error Analysis](reports/11_error_analysis.png)

---

### Dependencies

All required packages are listed in `requirements.txt`:

```
pandas>=1.5.0        # Data manipulation
numpy>=1.23.0        # Numerical computing
scikit-learn>=1.2.0  # ML algorithms & preprocessing
xgboost>=1.7.0       # Gradient boosting
shap>=0.41.0         # Model explainability
matplotlib>=3.6.0    # Plotting
seaborn>=0.12.0      # Statistical visualization
plotly>=5.0.0        # Interactive plots
umap-learn>=0.5.0    # Dimensionality reduction
streamlit>=1.20.0    # Web app framework
nbformat>=5.0.0      # Notebook format
joblib>=1.2.0        # Model serialization
```

---

### Limitations

1. **Dataset Bias**: Heavy US representation (~56%), primarily recent years (2020-2022)
2. **Missing Features**: No education, certifications, company reputation, equity/bonus data
3. **Model Variance**: ~60% of salary variation unexplained (external factors not captured in features)
4. **Small Dataset**: Only 565 records after deduplication limits model complexity
5. **Rare Titles**: Job titles with <5 occurrences grouped as "Other"
6. **Geographic Imbalance**: Some regions underrepresented (n<5 samples)

---

### Future Work

1. **Data Enhancement**:
   - Collect company industry/sector information
   - Add education level and certifications
   - Include historical salary trends
   - Track equity compensation and bonuses

2. **Model Improvements**:
   - Implement ensemble stacking
   - Neural network models for non-linear patterns
   - Time-series analysis for salary trends
   - Separate models per job title category

3. **Fairness & Ethics**:
   - Formal algorithmic audit
   - Implement fairness constraints
   - Bias mitigation strategies
   - Regular fairness testing

4. **Deployment**:
   - Web app for interactive salary estimates
   - A/B testing with real outcomes
   - Continuous learning pipeline
   - Model monitoring and drift detection

---

### License

**CC0 1.0 Universal** (Public Domain)

This project uses the Kaggle dataset under CC0 license (public domain). All code and analyses are provided as-is for educational purposes.

---

### Author

**Jerel Velarde**
Pillar 5 - Machine Learning Capstone Project
2026

---

### Contact & Questions

For questions about this project, methodology, or results, please refer to the comprehensive Jupyter notebook which contains detailed code, explanations, and visualizations.

**Notebook**: `notebooks/Jerel_Velarde_Pillar5_Capstone_AI_Job_Market.ipynb`

---

### Acknowledgments

- Dataset: [Kaggle - Data Science Job Salaries](https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries)
- Libraries: scikit-learn, XGBoost, SHAP, pandas, matplotlib, seaborn
- Inspiration: Real-world salary disparities in tech industry
