# Installation & Deployment Guide

## System Requirements

- Python 3.7 or higher
- pip (Python package manager)
- 100 MB free disk space
- Internet connection (for initial package installation)

## Installation Steps

### 1. Navigate to the App Directory

```bash
cd /path/to/ai-job-market-ml-capstone/app
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- streamlit (web framework)
- pandas (data manipulation)
- numpy (numerical computing)
- scikit-learn (ML utilities)
- xgboost (model framework)
- joblib (model serialization)
- shap (model explainability)
- matplotlib (plotting)
- plotly (interactive viz)

### 4. Verify Installation

```bash
# Check Streamlit installation
streamlit --version

# Check other key packages
python -c "import pandas; import xgboost; import sklearn; print('All packages installed successfully!')"
```

## Running the Application

### Local Development

```bash
streamlit run salary_predictor.py
```

The app will automatically open in your default browser at `http://localhost:8501`

### With Custom Port

```bash
streamlit run salary_predictor.py --server.port 8502
```

### Remote/Headless Server

```bash
streamlit run salary_predictor.py --server.headless true --logger.level=info
```

## Data Setup

The application requires the following data file:

```
data/raw/ds_salaries.csv
```

**File structure required:**
- Columns: work_year, experience_level, employment_type, job_title, salary, 
  salary_currency, salary_in_usd, employee_residence, remote_ratio, 
  company_location, company_size
- Format: CSV (comma-separated values)
- Encoding: UTF-8

**Example:**
```csv
,work_year,experience_level,employment_type,job_title,salary,salary_currency,salary_in_usd,employee_residence,remote_ratio,company_location,company_size
0,2020,MI,FT,Data Scientist,70000,EUR,79833,DE,0,DE,L
1,2020,SE,FT,Machine Learning Scientist,260000,USD,260000,JP,0,JP,S
```

If data file is missing, the app will show a warning but still work in fallback mode.

## Model Setup (Optional)

To use the XGBoost model predictions, place the trained model at:

```
models/trained_models/xgboost_model.joblib
```

**Creating a Model File:**
```python
import joblib
import xgboost as xgb

# Train your model
model = xgb.XGBRegressor(...)
model.fit(X_train, y_train)

# Save it
joblib.dump(model, 'models/trained_models/xgboost_model.joblib')
```

If no model file exists, the app automatically uses statistical fallback mode.

## Troubleshooting

### Issue: "No module named 'streamlit'"

**Solution:**
```bash
pip install streamlit
```

### Issue: "ModuleNotFoundError: No module named 'xgboost'"

**Solution:**
```bash
pip install xgboost
```

### Issue: App runs but shows warning about missing data

**Solution:**
Ensure `data/raw/ds_salaries.csv` exists with correct structure. The app will work with fallback values.

### Issue: Streamlit doesn't open browser automatically

**Solution:**
Manually open `http://localhost:8501` in your browser

### Issue: "Address already in use" error

**Solution:**
Use a different port:
```bash
streamlit run salary_predictor.py --server.port 8502
```

### Issue: Slow startup time

**Solution:**
First run caches dependencies. Subsequent runs will be faster. Ensure adequate disk space.

## Performance Optimization

### For Large Datasets

If using a large dataset (10,000+ rows):

```bash
# Increase memory allocation
streamlit run salary_predictor.py --client.maxMessageSize 200
```

### For Multiple Users (Production)

```bash
# Use Streamlit Cloud or Docker deployment
# See deployment section below
```

## Production Deployment

### Option 1: Streamlit Cloud (Recommended for beginners)

1. Push code to GitHub repository
2. Visit https://share.streamlit.io/
3. Connect GitHub account
4. Deploy directly from repository

### Option 2: Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "salary_predictor.py", "--server.port=8501", "--server.headless=true"]
```

Build and run:
```bash
docker build -t salary-predictor .
docker run -p 8501:8501 salary-predictor
```

### Option 3: Traditional Server (AWS, Heroku, etc.)

1. Upload files to server
2. Install Python 3.7+
3. Run installation steps above
4. Use process manager (e.g., supervisor, systemd)
5. Use reverse proxy (nginx) for public access

## Security Considerations

### For Production

1. **Never commit credentials or API keys**
   - Use environment variables
   - Use .gitignore for sensitive files

2. **Use HTTPS**
   - Deploy behind nginx/Apache reverse proxy
   - Use SSL certificates

3. **Restrict access if needed**
   - Use authentication layer (Streamlit Cloud auto-provides this)
   - Use firewall rules

4. **Data privacy**
   - Ensure salary data complies with privacy regulations
   - Use secure data storage
   - Encrypt sensitive files

### Environment Variables

Create `.env` file (not committed to git):
```
MODEL_PATH=/path/to/model.joblib
DATA_PATH=/path/to/data.csv
```

Load in app:
```python
import os
from dotenv import load_dotenv

load_dotenv()
MODEL_PATH = os.getenv('MODEL_PATH', 'models/trained_models/xgboost_model.joblib')
```

## Updating Dependencies

To update all packages to latest versions:

```bash
# Update pip first
pip install --upgrade pip

# Update all packages
pip install --upgrade -r requirements.txt

# Generate updated requirements file
pip freeze > requirements.txt
```

## Uninstalling

To remove the virtual environment and all packages:

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv  # macOS/Linux
rmdir /s venv  # Windows
```

## Testing the Installation

Run this test script to verify everything works:

```python
import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib

print("Testing installation...")

# Test imports
try:
    import streamlit
    print("✅ Streamlit OK")
except:
    print("❌ Streamlit failed")

try:
    import pandas
    print("✅ Pandas OK")
except:
    print("❌ Pandas failed")

try:
    import xgboost
    print("✅ XGBoost OK")
except:
    print("❌ XGBoost failed")

try:
    import joblib
    print("✅ Joblib OK")
except:
    print("❌ Joblib failed")

print("\nAll tests passed! Ready to run the app.")
```

## Support & Additional Resources

- **Streamlit Docs**: https://docs.streamlit.io/
- **XGBoost Guide**: https://xgboost.readthedocs.io/
- **Pandas Documentation**: https://pandas.pydata.org/docs/
- **Python Virtual Environments**: https://docs.python.org/3/tutorial/venv.html

## Version Information

Recommended versions:
- Python: 3.8 or 3.9 (most stable)
- Streamlit: 1.28.0+
- XGBoost: 2.0.0+
- Pandas: 2.0.0+

## Next Steps

After installation:
1. Read QUICKSTART.md for usage instructions
2. Check README.md for feature documentation
3. Review APP_SUMMARY.txt for technical details
4. Run `streamlit run salary_predictor.py`
