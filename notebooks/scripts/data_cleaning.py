# data_cleaning.py
import pandas as pd
import numpy as np

def clean_titanic_data(df):
    """
    Perform data cleaning on Titanic dataset
    """
    df = df.copy()
    
    # Drop Cabin (77% missing)
    df.drop('Cabin', axis=1, inplace=True)
    
    # Impute Age with median
    df['Age'].fillna(df['Age'].median(), inplace=True)
    
    # Impute Embarked with mode
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Create missing indicator
    df['Age_Missing'] = 0
    
    # Cap Fare outliers at 95th percentile
    fare_95 = df['Fare'].quantile(0.95)
    df['Fare'] = np.where(df['Fare'] > fare_95, fare_95, df['Fare'])
    
    # Cap Age outliers at 95th percentile
    age_95 = df['Age'].quantile(0.95)
    df['Age'] = np.where(df['Age'] > age_95, age_95, df['Age'])
    
    # Fix Sex values
    df['Sex'] = df['Sex'].map({'male': 'Male', 'female': 'Female'})
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    return df

if __name__ == "__main__":
    train = pd.read_csv('data/train.csv')
    cleaned = clean_titanic_data(train)
    cleaned.to_csv('data/train_cleaned.csv', index=False)
    print(f"Cleaned {len(train)} rows -> {len(cleaned)} rows")
    print("Missing values after cleaning:\n", cleaned.isnull().sum())
