# feature_selection.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def select_features(df, target_col='Survived', n_features=10):
    """
    Select top n features using Random Forest importance
    """
    df = df.copy()
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Random Forest importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Remove highly correlated features
    corr_matrix = df.corr()
    to_drop = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.85:
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                if col1 not in to_drop and col2 not in to_drop:
                    to_drop.append(col2)
    
    selected = [f for f in importance_df['feature'].head(n_features).tolist() 
                if f not in to_drop]
    
    return selected, importance_df

if __name__ == "__main__":
    df = pd.read_csv('data/train_engineered.csv')
    selected_features, importance = select_features(df)
    
    print("Selected features:")
    for f in selected_features:
        print(f"  - {f}")
    
    # Save final dataset
    final_df = df[selected_features + ['Survived']]
    final_df.to_csv('data/train_selected_features.csv', index=False)
    
    # Save importance for reference
    importance.to_csv('data/feature_importance.csv', index=False)
