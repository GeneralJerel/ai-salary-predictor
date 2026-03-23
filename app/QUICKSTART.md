# Quick Start Guide - AI Salary Predictor

## 1. Install Dependencies

```bash
cd /path/to/ai-job-market-ml-capstone/app
pip install -r requirements.txt
```

## 2. Run the App

```bash
streamlit run salary_predictor.py
```

## 3. Use the Predictor

1. **Configure your profile** using the sidebar inputs:
   - Select your job title
   - Choose your experience level
   - Select employment type
   - Choose company size
   - Set remote work percentage
   - Select company location

2. **Click "Predict Salary"** to get the prediction

3. **View results** showing:
   - Annual salary in USD
   - Monthly salary
   - Hourly rate
   - Salary percentile in dataset
   - Min/max salary comparison
   - Comparison to median

## Available Job Titles

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

## Available Company Locations

US, GB, DE, CA, IN, FR, JP, AU, SG, NL, CH, SE, ES, IT, BR, MX, NZ, IE, IL, RU

## Experience Level Codes

- **Entry (EN)**: 0-2 years of experience
- **Mid (MI)**: 2-5 years of experience
- **Senior (SE)**: 5+ years of experience
- **Executive (EX)**: Leadership and management roles

## Employment Types

- **Full-time**: Regular permanent employment
- **Part-time**: Less than 40 hours per week
- **Contract**: Temporary or project-based work
- **Freelance**: Independent contractor

## Model Operation

### XGBoost Mode ✅ (When Trained Model Exists)
- Accurate predictions using trained machine learning model
- Located at: `models/trained_models/xgboost_model.joblib`
- Model status shown in top-right corner

### Fallback Mode ⚠️ (When No Model Available)
- Statistical estimation based on training data
- Uses experience level and job title averages
- Applies reasonable adjustments for other factors
- Still provides useful predictions

## Tips for Better Predictions

1. **Be Accurate**: Select the job title that best matches your role
2. **Remote Work**: Adjust slider to reflect actual remote percentage
3. **Company Size**: Consider the actual size of the organization
4. **Location**: Select where the company is headquartered
5. **Experience**: Choose the level that reflects your actual experience

## Troubleshooting

### App Won't Start
```bash
# Check Streamlit installation
pip list | grep streamlit

# Update if needed
pip install --upgrade streamlit
```

### Missing Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt

# Specific packages
pip install xgboost joblib pandas numpy scikit-learn
```

### Data Not Loading
- Ensure `data/raw/ds_salaries.csv` exists in the correct location
- Check file permissions
- Verify CSV format is correct

### Model Not Loading
- Check if `models/trained_models/xgboost_model.joblib` exists
- App will work in fallback mode if model is missing
- Status indicator in top-right shows which mode is active

## Example Scenarios

### Scenario 1: Mid-Level Data Scientist in US
- Job Title: Data Scientist
- Experience: Mid
- Employment: Full-time
- Company Size: Large
- Remote: 50%
- Location: US

### Scenario 2: Senior ML Engineer in Europe (Remote)
- Job Title: Machine Learning Engineer
- Experience: Senior
- Employment: Full-time
- Company Size: Large
- Remote: 100%
- Location: DE

### Scenario 3: Entry-Level Analyst
- Job Title: Data Analyst
- Experience: Entry
- Employment: Full-time
- Company Size: Medium
- Remote: 0%
- Location: CA

## Understanding the Output

```
Annual Salary: $120,000 (main prediction)
Monthly Salary: $10,000 (annual / 12)
Hourly Rate: $57.69 (annual / 2080 work hours)
Salary Percentile: 75.3% (your salary vs. others in dataset)
Min Salary: $20,000 (lowest in dataset)
Max Salary: $380,000 (highest in dataset)
vs Median: +45.2% (how much above median)
```

## Performance Notes

- Predictions load instantly with caching enabled
- First run may take a few seconds
- Subsequent predictions are cached for speed
- App uses browser-based computation for privacy

## Support

For issues or questions:
1. Check the Model Information expander in the app
2. Review README.md for detailed documentation
3. Verify all data files are in correct locations
4. Ensure dependencies are properly installed
