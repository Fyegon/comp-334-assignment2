# feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Create derived features from Titanic dataset
    """
    df = df.copy()
    
    # Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Is Alone
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Title extraction
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
        'Capt': 'Rare', 'Sir': 'Rare'
    }
    df['Title'] = df['Title'].map(title_mapping)
    
    # Age groups
    def age_group(age):
        if age < 12: return 'Child'
        elif age < 18: return 'Teen'
        elif age < 60: return 'Adult'
        else: return 'Senior'
    df['AgeGroup'] = df['Age'].apply(age_group)
    
    # Fare per person
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    
    # One-hot encoding
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title', 'AgeGroup'], drop_first=True)
    
    # Drop original text columns
    df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
    
    # Interaction features
    df['Pclass_Fare'] = df['Pclass'] * df['Fare']
    
    # Log transforms
    df['Fare_log'] = np.log1p(df['Fare'])
    df['Age_log'] = np.log1p(df['Age'])
    
    return df

if __name__ == "__main__":
    train = pd.read_csv('data/train_cleaned.csv')
    engineered = engineer_features(train)
    engineered.to_csv('data/train_engineered.csv', index=False)
    print(f"Features created: {list(engineered.columns)}")
