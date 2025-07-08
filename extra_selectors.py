
import pandas as pd
from sklearn.feature_selection import f_classif, RFE
from sklearn.linear_model import LogisticRegression

def calculate_anova_f_test(data: pd.DataFrame):
    """Calculates feature importance using ANOVA F-test."""
    print("Calculating ANOVA F-test scores...")
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    
    f_scores, p_values = f_classif(X, y)
    results = pd.DataFrame({
        'Feature_Name': X.columns,
        'ANOVA_F_Score': f_scores,
        'P_Value': p_values
    })
    
    return results.sort_values(by='ANOVA_F_Score', ascending=False)

def calculate_rfe(data: pd.DataFrame):
    """Calculates feature ranking using Recursive Feature Elimination (RFE)."""
    print("Calculating RFE ranks...")
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    
    # Use a simple, fast model for RFE
    estimator = LogisticRegression(solver='liblinear', random_state=42)
    # The rank is the inverse of the selection order. Rank 1 is the best feature.
    # Limit n_features_to_select for performance on free tiers.
    n_features_to_select = min(50, X.shape[1])
    rfe = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
    rfe.fit(X, y)
    
    results = pd.DataFrame({
        'Feature_Name': X.columns,
        'RFE_Rank': rfe.ranking_
    })
    
    return results.sort_values(by='RFE_Rank', ascending=True)
