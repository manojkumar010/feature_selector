import pandas as pd
from mrmr import mrmr_classif
from sklearn.feature_selection import mutual_info_classif

def calculate_mrmr_features(data: pd.DataFrame):
    """
    Calculates feature importance using the MRMR method for ranking and
    Mutual Information for scoring.
    """
    print("Calculating MRMR feature scores...")
    
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    
    # 1. Get the feature ranking from mrmr_classif
    ranked_features = mrmr_classif(X=X, y=y, K=len(X.columns))
    mrmr_ranks = pd.DataFrame({
        'MRMR_Ranked_Feature': ranked_features,
        'MRMR_Rank': range(1, len(ranked_features) + 1)
    })

    # 2. Calculate the Mutual Information score for each feature
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mrmr_scores = pd.DataFrame({
        'Feature_Name': X.columns,
        'MRMR_Score (Mutual Info)': mi_scores
    })

    # 3. Combine the rank and score
    final_mrmr_results = pd.merge(mrmr_scores, mrmr_ranks, left_on='Feature_Name', right_on='MRMR_Ranked_Feature')
    final_mrmr_results = final_mrmr_results.drop(columns=['MRMR_Ranked_Feature'])
    final_mrmr_results = final_mrmr_results.sort_values(by='MRMR_Rank', ascending=True)
    
    print("MRMR calculation finished.")
    return final_mrmr_results