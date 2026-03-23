# AI Salary Predictor App

A Streamlit web application that predicts AI/Data Science job salaries based on various factors including job title, experience level, employment type, company size, remote work ratio, and company location.

## Features

- **Interactive Sidebar Inputs**:
  - Job Title dropdown (15+ common AI/Data Science roles)
  - Experience Level selector (Entry/Mid/Senior/Executive)
  - Employment Type selector (Full-time/Part-time/Contract/Freelance)
  - Company Size selector (Small/Medium/Large)
  - Remote Work Ratio slider (0-100%)
  - Company Location dropdown (20+ countries)

- **Dual Prediction Modes**:
  - **XGBoost Model**: Uses trained machine learning model when available
  - **Fallback Mode**: Statistical estimation based on training data when model unavailable

- **Rich Output**:
  - Annual, monthly, and hourly salary estimates
  - Salary percentile within dataset
  - Min/max salary comparison
  - Comparison to median salary
  - Summary of selected factors

- **User-Friendly Interface**:
  - Clean, professional UI with custom styling
  - Real-time predictions
  - Model status indicator
  - Informative tooltips
  - Comprehensive model information

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the following data structure exists:
```
ai-job-market-ml-capstone/
├── data/
│   └── raw/
│       └── ds_salaries.csv
├── models/
│   └── trained_models/
│       └── xgboost_model.joblib (optional)
└── app/
    └── salary_predictor.py
```

## Running the App

```bash
streamlit run salary_predictor.py
```

The app will open in your default browser at `http://localhost:8501`

## Dataset Columns

The application uses data with the following columns:
- `work_year`: Year of the salary record
- `experience_level`: EN (Entry), MI (Mid), SE (Senior), EX (Executive)
- `employment_type`: FT (Full-time), PT (Part-time), CT (Contract), FL (Freelance)
- `job_title`: Job position title
- `salary`: Salary in original currency
- `salary_currency`: Currency code
- `salary_in_usd`: Salary converted to USD
- `employee_residence`: Employee's country of residence
- `remote_ratio`: Percentage of remote work (0, 50, or 100)
- `company_location`: Company's country location
- `company_size`: S (Small), M (Medium), L (Large)

## Prediction Logic

### XGBoost Mode (When Model Available)
- Uses trained gradient boosting model for accurate predictions
- Requires feature engineering to match training data format
- Provides most accurate predictions based on historical patterns

### Fallback Mode (When Model Unavailable)
- Calculates median salary for experience level and job title
- Applies multipliers based on:
  - Experience level (EX: 1.3x, SE: 1.15x, MI: 1.0x, EN: 0.7x)
  - Remote work bonus (5% per 100% remote)
  - Company size bonus (S: 0.9x, M: 1.0x, L: 1.1x)
- Provides reasonable estimates without ML model

## Supported Job Titles

- Data Scientist
- Machine Learning Engineer
- Data Engineer
- Data Analyst
- AI Scientist
- Research Scientist
- ML Engineer
- Data Science Manager
- Analytics Engineer
- Applied Scientist
- AI Engineer
- Machine Learning Scientist
- Deep Learning Engineer
- Computer Vision Engineer
- NLP Engineer

## Supported Locations

US, GB, DE, CA, IN, FR, JP, AU, SG, NL, CH, SE, ES, IT, BR, MX, NZ, IE, IL, RU

## Notes

- Predictions are estimates based on historical data
- Actual salaries may vary based on:
  - Specific technical skills and certifications
  - Company reputation and prestige
  - Geographic cost of living adjustments
  - Individual negotiation outcomes
  - Additional company-specific factors

## Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities
- **xgboost**: Gradient boosting model
- **joblib**: Model serialization
- **shap**: Model explainability (optional)
- **matplotlib**: Visualization
- **plotly**: Interactive visualization

## Author

Created as part of the AI/ML Capstone Project

## License

This project is provided as-is for educational purposes.
