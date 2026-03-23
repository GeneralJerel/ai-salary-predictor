import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="AI Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 20px;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .salary-value {
        font-size: 36px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-title'>💰 AI Salary Predictor</h1>", unsafe_allow_html=True)

# App paths
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "trained_models" / "xgboost_model.joblib"
DATA_PATH = BASE_DIR / "data" / "raw" / "ds_salaries.csv"

# Load training data for fallback and feature engineering
@st.cache_data
def load_training_data():
    """Load the training dataset for fallback statistics"""
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except Exception as e:
        st.warning(f"Could not load training data: {e}")
        return None

# Load model
@st.cache_resource
def load_model():
    """Load the trained XGBoost model"""
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            return model, True
        else:
            return None, False
    except Exception as e:
        st.warning(f"Could not load model: {e}")
        return None, False

# Load data
training_data = load_training_data()
model, model_loaded = load_model()

# Feature engineering function
def engineer_features(job_title, experience_level, employment_type, company_size,
                      remote_ratio, company_location):
    """Engineer features for prediction"""
    features = {}

    # Map categorical variables to training codes
    experience_mapping = {
        'Entry': 'EN',
        'Mid': 'MI',
        'Senior': 'SE',
        'Executive': 'EX'
    }

    employment_mapping = {
        'Full-time': 'FT',
        'Part-time': 'PT',
        'Contract': 'CT',
        'Freelance': 'FL'
    }

    company_size_mapping = {
        'Small': 'S',
        'Medium': 'M',
        'Large': 'L'
    }

    # Create features dict
    features['experience_level'] = experience_mapping.get(experience_level, 'MI')
    features['employment_type'] = employment_mapping.get(employment_type, 'FT')
    features['company_size'] = company_size_mapping.get(company_size, 'M')
    features['remote_ratio'] = remote_ratio
    features['job_title'] = job_title
    features['company_location'] = company_location

    return features

# Fallback salary estimation
def estimate_salary_fallback(features, data):
    """Estimate salary using training data statistics"""
    if data is None:
        return 70000  # Default fallback

    exp_level = features['experience_level']
    job_title = features['job_title']

    # Filter by experience level
    exp_data = data[data['experience_level'] == exp_level]

    if len(exp_data) > 0:
        # Further filter by job title if available
        job_data = exp_data[exp_data['job_title'] == job_title]
        if len(job_data) > 0:
            base_salary = job_data['salary_in_usd'].median()
        else:
            base_salary = exp_data['salary_in_usd'].median()
    else:
        base_salary = data['salary_in_usd'].median()

    # Adjustments based on features
    adjustments = {
        'EX': 1.3,
        'SE': 1.15,
        'MI': 1.0,
        'EN': 0.7
    }

    multiplier = adjustments.get(exp_level, 1.0)
    remote_bonus = 1 + (features['remote_ratio'] / 100) * 0.05  # 5% boost per 100% remote
    company_size_bonus = {
        'S': 0.9,
        'M': 1.0,
        'L': 1.1
    }.get(features['company_size'], 1.0)

    predicted_salary = base_salary * multiplier * remote_bonus * company_size_bonus

    return int(predicted_salary)

# Sidebar inputs
st.sidebar.markdown("## Prediction Inputs")

# Job Title
job_titles = [
    'Data Scientist',
    'Machine Learning Engineer',
    'Data Engineer',
    'Data Analyst',
    'AI Scientist',
    'Research Scientist',
    'ML Engineer',
    'Data Science Manager',
    'Analytics Engineer',
    'Applied Scientist',
    'AI Engineer',
    'Machine Learning Scientist',
    'Deep Learning Engineer',
    'Computer Vision Engineer',
    'NLP Engineer'
]

selected_job_title = st.sidebar.selectbox(
    "Job Title",
    job_titles,
    index=0,
    help="Select your AI/Data job title"
)

# Experience Level
experience_level = st.sidebar.selectbox(
    "Experience Level",
    ["Entry", "Mid", "Senior", "Executive"],
    index=1,
    help="Your professional experience level"
)

# Employment Type
employment_type = st.sidebar.selectbox(
    "Employment Type",
    ["Full-time", "Part-time", "Contract", "Freelance"],
    index=0,
    help="Type of employment"
)

# Company Size
company_size = st.sidebar.selectbox(
    "Company Size",
    ["Small", "Medium", "Large"],
    index=1,
    help="Size of the company"
)

# Remote Ratio
remote_ratio = st.sidebar.slider(
    "Remote Work Ratio (%)",
    min_value=0,
    max_value=100,
    value=50,
    step=10,
    help="0% = Fully On-site, 100% = Fully Remote"
)

# Company Location
locations = ['US', 'GB', 'DE', 'CA', 'IN', 'FR', 'JP', 'AU', 'SG', 'NL',
             'CH', 'SE', 'ES', 'IT', 'BR', 'MX', 'NZ', 'IE', 'IL', 'RU']

company_location = st.sidebar.selectbox(
    "Company Location",
    locations,
    index=0,
    help="Country where the company is based"
)

# Engineer features
features = engineer_features(
    selected_job_title,
    experience_level,
    employment_type,
    company_size,
    remote_ratio,
    company_location
)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Selected Profile")
    profile_info = f"""
    - **Job Title:** {selected_job_title}
    - **Experience Level:** {experience_level}
    - **Employment Type:** {employment_type}
    - **Company Size:** {company_size}
    - **Remote Ratio:** {remote_ratio}%
    - **Company Location:** {company_location}
    """
    st.info(profile_info)

with col2:
    st.markdown("### Model Status")
    if model_loaded:
        st.success("✅ Model Loaded")
    else:
        st.warning("⚠️ Using Fallback Mode")

# Make prediction
st.markdown("---")
st.markdown("### Salary Prediction")

if st.button("🔮 Predict Salary", use_container_width=True):
    try:
        if model_loaded and model is not None:
            # Use the trained model
            st.info("Using trained XGBoost model for prediction...")

            # Prepare features for model
            # Note: In production, you'd need to match the exact feature engineering
            # that was used during model training
            feature_dict = {
                'experience_level_code': {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}.get(features['experience_level'], 1),
                'employment_type_code': {'FT': 0, 'PT': 1, 'CT': 2, 'FL': 3}.get(features['employment_type'], 0),
                'company_size_code': {'S': 0, 'M': 1, 'L': 2}.get(features['company_size'], 1),
                'remote_ratio': features['remote_ratio'],
            }

            # Try to predict with model
            try:
                # Create DataFrame with expected features
                pred_data = pd.DataFrame([feature_dict])
                predicted_salary = model.predict(pred_data)[0]
                prediction_source = "XGBoost Model"
            except Exception as e:
                # Fallback if model prediction fails
                st.warning(f"Model prediction failed: {e}. Using fallback method...")
                predicted_salary = estimate_salary_fallback(features, training_data)
                prediction_source = "Fallback Estimation"
        else:
            # Use fallback estimation
            st.info("Using fallback estimation based on training data statistics...")
            predicted_salary = estimate_salary_fallback(features, training_data)
            prediction_source = "Fallback Estimation"

        # Display prediction
        st.markdown("---")
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Annual Salary (USD)",
                f"${predicted_salary:,.0f}",
                delta=f"Prediction Source: {prediction_source}"
            )

        with col2:
            monthly = predicted_salary / 12
            st.metric(
                "Monthly Salary",
                f"${monthly:,.0f}"
            )

        with col3:
            hourly = predicted_salary / 2080  # Standard 2080 work hours per year
            st.metric(
                "Hourly Rate",
                f"${hourly:,.2f}"
            )

        st.markdown('</div>', unsafe_allow_html=True)

        # Additional insights
        st.markdown("### 📊 Insights")

        if training_data is not None:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                avg_salary = training_data['salary_in_usd'].mean()
                percentile = (training_data['salary_in_usd'] <= predicted_salary).sum() / len(training_data) * 100
                st.metric("Salary Percentile", f"{percentile:.1f}%")

            with col2:
                min_salary = training_data['salary_in_usd'].min()
                st.metric("Min Salary (Dataset)", f"${min_salary:,.0f}")

            with col3:
                max_salary = training_data['salary_in_usd'].max()
                st.metric("Max Salary (Dataset)", f"${max_salary:,.0f}")

            with col4:
                median_salary = training_data['salary_in_usd'].median()
                diff = ((predicted_salary - median_salary) / median_salary) * 100
                st.metric(
                    "vs Median Dataset",
                    f"{diff:+.1f}%"
                )

        # Salary factors breakdown
        st.markdown("### 🎯 Salary Factors")

        factors = {
            "Experience Level": f"{experience_level} (Codes: EN=Entry, MI=Mid, SE=Senior, EX=Executive)",
            "Employment Type": employment_type,
            "Company Size": company_size,
            "Remote Work": f"{remote_ratio}% remote",
            "Location": company_location
        }

        factors_df = pd.DataFrame(list(factors.items()), columns=["Factor", "Value"])
        st.table(factors_df)

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.info("Please check your inputs and try again.")

# Model information
st.markdown("---")
with st.expander("📋 Model Information"):
    st.markdown("""
    ### About This Predictor

    This AI Salary Predictor uses machine learning to estimate salaries in the AI/Data Science job market.

    **Features Considered:**
    - Job Title
    - Experience Level
    - Employment Type
    - Company Size
    - Remote Work Ratio
    - Company Location

    **Model Modes:**
    - **XGBoost Model**: When a trained model is available, predictions are made using this high-performance model
    - **Fallback Mode**: When no model is available, salaries are estimated using statistical analysis of historical data

    **Data Source:**
    - Training data from AI/Data Science salary datasets
    - Multiple years of industry data

    **Disclaimer:**
    This tool provides estimates based on historical data and machine learning. Actual salaries may vary
    based on additional factors not captured in this model such as:
    - Specific skills and certifications
    - Company reputation and prestige
    - Geographic cost of living adjustments
    - Negotiation outcomes
    - Company-specific compensation policies
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px; margin-top: 20px;'>
    <p>AI Salary Predictor | Built with Streamlit | ML-Powered Salary Estimation</p>
</div>
""", unsafe_allow_html=True)
