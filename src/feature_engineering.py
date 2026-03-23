"""
Feature Engineering Module
Transforms raw features into ML-ready features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Mapping of country codes to continents
COUNTRY_TO_CONTINENT = {
    'US': 'North America', 'CA': 'North America', 'MX': 'North America',
    'GB': 'Europe', 'DE': 'Europe', 'FR': 'Europe', 'ES': 'Europe', 'IT': 'Europe',
    'NL': 'Europe', 'SE': 'Europe', 'CH': 'Europe', 'AT': 'Europe', 'BE': 'Europe',
    'DK': 'Europe', 'PL': 'Europe', 'RO': 'Europe', 'CZ': 'Europe', 'GR': 'Europe',
    'PT': 'Europe', 'IE': 'Europe', 'UA': 'Europe', 'HR': 'Europe', 'LV': 'Europe',
    'JP': 'Asia', 'IN': 'Asia', 'CN': 'Asia', 'SG': 'Asia', 'AU': 'Oceania',
    'NZ': 'Oceania', 'BR': 'South America', 'AR': 'South America', 'CL': 'South America',
    'ZA': 'Africa', 'NG': 'Africa', 'EG': 'Africa', 'KE': 'Africa',
    'HN': 'Central America', 'KR': 'Asia', 'RU': 'Europe',
    'TH': 'Asia', 'ID': 'Asia', 'VN': 'Asia', 'MY': 'Asia', 'PH': 'Asia',
    'PK': 'Asia', 'IR': 'Asia', 'IL': 'Asia', 'AE': 'Asia', 'SA': 'Asia',
    'JO': 'Asia', 'TR': 'Europe', 'HK': 'Asia', 'TW': 'Asia', 'GH': 'Africa',
}

def create_continent_feature(df):
    """
    Create continent feature from company_location country codes.
    """
    df['continent'] = df['company_location'].map(COUNTRY_TO_CONTINENT)
    # Handle unmapped countries
    df['continent'] = df['continent'].fillna('Other')
    return df

def create_is_us_feature(df):
    """Create binary feature for US-based companies."""
    df['is_us'] = (df['company_location'] == 'US').astype(int)
    return df

def group_rare_job_titles(df, min_count=5):
    """
    Group job titles appearing less than min_count times into 'Other'.
    """
    title_counts = df['job_title'].value_counts()
    rare_titles = title_counts[title_counts < min_count].index
    df['job_title'] = df['job_title'].apply(
        lambda x: 'Other' if x in rare_titles else x
    )
    return df

def encode_experience_level(df):
    """
    Ordinal encode experience level:
    EN (Entry) = 1
    MI (Mid) = 2
    SE (Senior) = 3
    EX (Executive) = 4
    """
    experience_mapping = {
        'EN': 1,
        'MI': 2,
        'SE': 3,
        'EX': 4
    }
    df['experience_level_encoded'] = df['experience_level'].map(experience_mapping)
    return df

def encode_categorical_features(df):
    """
    One-hot encode employment_type and company_size.
    Label encode job_title and continent (already grouped rare titles).
    """
    # One-hot encode employment_type
    df = pd.concat([
        df,
        pd.get_dummies(df['employment_type'], prefix='employment_type')
    ], axis=1)

    # One-hot encode company_size
    df = pd.concat([
        df,
        pd.get_dummies(df['company_size'], prefix='company_size')
    ], axis=1)

    # Label encode job_title
    le_title = LabelEncoder()
    df['job_title_encoded'] = le_title.fit_transform(df['job_title'])

    # Label encode continent
    le_continent = LabelEncoder()
    df['continent_encoded'] = le_continent.fit_transform(df['continent'])

    return df

def engineer_features(df):
    """
    Apply all feature engineering transformations.
    """
    print("Starting feature engineering...")

    # 1. Group rare job titles
    df = group_rare_job_titles(df, min_count=5)
    print(f"Job titles after grouping rare: {df['job_title'].nunique()} unique titles")

    # 2. Create continent feature
    df = create_continent_feature(df)
    print(f"Continents: {df['continent'].unique()}")

    # 3. Create is_us feature
    df = create_is_us_feature(df)

    # 4. Encode experience level (ordinal)
    df = encode_experience_level(df)

    # 5. Encode other categorical features
    df = encode_categorical_features(df)

    print("Feature engineering complete!")
    return df

def prepare_features(df):
    """
    Prepare final feature matrix X and target y.
    Select relevant engineered features.
    """
    # Feature engineering
    df = engineer_features(df)

    # Select features for the model
    feature_columns = [
        'work_year',
        'experience_level_encoded',
        'is_us',
        'continent_encoded',
        'job_title_encoded',
        'remote_ratio',
    ]

    # Add one-hot encoded columns
    for col in df.columns:
        if col.startswith('employment_type_') or col.startswith('company_size_'):
            feature_columns.append(col)

    # Create feature matrix and target
    X = df[feature_columns].copy()
    y = df['salary_in_usd'].copy()

    print(f"\nFinal feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeatures used: {feature_columns}")

    return X, y, df

def main(input_path='data/processed/ds_salaries_clean.csv'):
    """Main feature engineering pipeline."""
    df = pd.read_csv(input_path)
    X, y, df_engineered = prepare_features(df)
    return X, y, df_engineered

if __name__ == '__main__':
    main()
