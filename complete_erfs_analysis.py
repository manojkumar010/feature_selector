import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
import argparse
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

def proposed_approach_python(data: np.ndarray):
    """
    This is a direct Python translation of the `proposed_approach.m` MATLAB script.
    It calculates feature weights based on the overlap of class distributions.
    """
    n_samples, n_features_total = data.shape
    dim = n_features_total - 1  # Number of features, excluding the class column
    
    # Ensure the first column (class labels) is integer
    class_labels = data[:, 0].astype(int)
    features = data[:, 1:]
    
    unique_classes = np.unique(class_labels)
    n_class = len(unique_classes)
    
    if n_class <= 1:
        return np.zeros(dim)

    # Calculate class probabilities
    class_prob = np.array([np.sum(class_labels == c) for c in unique_classes])
    cl_fac = class_prob / np.sum(class_prob)
    
    area_coeff = np.zeros(dim)

    for i in range(dim):
        overlap_area = 0
        feature_vector = features[:, i]
        
        cl_range = np.zeros((n_class, 2))
        
        # Calculate statistical range for each class
        for j, cls in enumerate(unique_classes):
            idx = np.where(class_labels == cls)[0]
            if len(idx) > 1:
                mu = np.mean(feature_vector[idx])
                sigma = np.std(feature_vector[idx], ddof=1) # ddof=1 for sample std dev
            else:
                mu = feature_vector[idx][0] if len(idx) > 0 else 0
                sigma = 0
            
            # This formula matches the MATLAB code: mu +/- (1-cl_fac)*sqrt(3)*sigma
            # sqrt(3) is approx 1.732
            range_val = (1 - cl_fac[j]) * 1.732 * sigma
            cl_range[j, 0] = mu - range_val
            cl_range[j, 1] = mu + range_val

        # Sort ranges for consistent overlap calculation
        s_cl_range = cl_range[np.argsort(cl_range[:, 0])]
        
        # Calculate pairwise overlap area
        for j in range(n_class - 1):
            for k in range(j + 1, n_class):
                # Check for overlap
                if (s_cl_range[j, 1] - s_cl_range[k, 0]) > 0:
                    overlap_area += abs(s_cl_range[j, 1] - s_cl_range[k, 0])
        
        # Normalize overlap area by the total range
        total_range = np.max(s_cl_range[:, 1]) - np.min(s_cl_range[:, 0])
        if total_range > 0:
            area_coeff[i] = overlap_area / total_range
        else:
            area_coeff[i] = 0

    # Scale weights: weight = 1 - normalized_overlap
    area_coeff_scaled = area_coeff / np.max(area_coeff) if np.max(area_coeff) > 0 else area_coeff
    wt = 1 - area_coeff_scaled
    wt[np.isnan(wt)] = 0 # Handle potential NaNs
    
    # Final normalization
    final_weight = wt / np.max(wt) if np.max(wt) > 0 else wt
    return final_weight


def run_erfs_ensemble(data: np.ndarray, n_iterations=100, subset_fraction=0.8):
    """
    Implements the full "Ensemble of Random Feature Subsets" (ERFS) approach.
    """
    print(f"\nRunning ERFS Ensemble with {n_iterations} iterations...")
    n_samples, n_features_total = data.shape
    total_features = n_features_total - 1
    
    # Store weights for each feature across all iterations
    all_weights = np.zeros((n_iterations, total_features))
    
    feature_indices = np.arange(total_features)

    for i in range(n_iterations):
        # Create a random subset of features
        n_subset = int(total_features * subset_fraction)
        subset_indices = np.random.choice(feature_indices, n_subset, replace=False)
        
        # Prepare data subset (class column + selected feature columns)
        class_column = data[:, 0].reshape(-1, 1)
        feature_subset = data[:, subset_indices + 1]
        data_subset = np.hstack([class_column, feature_subset])
        
        # Calculate weights for the subset
        subset_weights = proposed_approach_python(data_subset)
        
        # Assign calculated weights back to their original feature positions
        all_weights[i, subset_indices] = subset_weights
        print(f"  Iteration {i+1}/{n_iterations} complete.", end='\r')

    print("\nEnsemble run finished.")
    
    # Aggregate weights by averaging over all iterations
    final_weights = np.mean(all_weights, axis=0)
    return final_weights


def tune_hyperparameters_and_evaluate(data: np.ndarray, weights: np.ndarray):
    """
    Performs hyperparameter tuning for `top_n_features` and evaluates the model.
    """
    print("\nStarting hyperparameter tuning for 'top_n_features'...")
    
    X = data[:, 1:]
    y = data[:, 0]
    
    # Sort features by their calculated ERFS weights in descending order
    sorted_feature_indices = np.argsort(weights)[::-1]
    
    results = []
    
    # Define the range of `top_n_features` to test
    n_features_to_test = range(5, min(51, X.shape[1] + 1))

    for n in n_features_to_test:
        # Select the top N features
        top_n_indices = sorted_feature_indices[:n]
        X_subset = X[:, top_n_indices]
        
        # Evaluate using 10-fold stratified cross-validation
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        clf = LinearDiscriminantAnalysis()
        
        scores = cross_val_score(clf, X_subset, y, cv=cv, scoring='accuracy')
        results.append({'n_features': n, 'mean_accuracy': np.mean(scores)})
        print(f"  Tested with {n} features: Mean Accuracy = {np.mean(scores):.4f}", end='\r')

    print("\nHyperparameter tuning finished.")
    
    # Find the best result
    best_result = max(results, key=lambda x: x['mean_accuracy'])
    return best_result, results

def main():
    parser = argparse.ArgumentParser(
        description="""
        A complete Python implementation of the ERFS algorithm with hyperparameter tuning.
        This script calculates robust feature weights using an ensemble method and
        finds the optimal number of features to maximize classifier accuracy.
        """
    )
    parser.add_argument(
        'excel_path', 
        type=str, 
        help="Path to the input Excel file (e.g., 'AFT_SCA1_SCA2.xlsx')."
    )
    parser.add_argument(
        'sheet_name', 
        type=str, 
        help="Name of the sheet to process (e.g., 'Sheet3' or 'Sheet4')."
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default="feature_weights_output.csv",
        help="Name of the CSV file to save the calculated feature weights."
    )
    args = parser.parse_args()

    print(f"--- Starting Analysis for {args.sheet_name} ---")
    
    # 1. Load Data
    try:
        data_df = pd.read_excel(args.excel_path, sheet_name=args.sheet_name, header=None)
        # Assuming the first column is class and the rest are features
        data_np = data_df.to_numpy()
        print(f"Successfully loaded data: {data_np.shape[0]} samples, {data_np.shape[1]-1} features.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Run the full ERFS ensemble to get robust feature weights
    erfs_weights = run_erfs_ensemble(data_np)
    
    # Save the calculated weights
    pd.DataFrame(erfs_weights, columns=['weight']).to_csv(args.output_file, index_label='feature_index')
    print(f"\nRobust feature weights saved to '{args.output_file}'")

    # 3. Tune hyperparameters and find the best model
    best_params, all_results = tune_hyperparameters_and_evaluate(data_np, erfs_weights)
    
    # 4. Print the final report
    print("\n--- Final Report ---")
    print(f"Dataset Source: '{args.excel_path}' (Sheet: {args.sheet_name})")
    print("\n**Optimal Hyperparameters Found:**")
    print(f"  - Best Number of Features (top_n_features): {best_params['n_features']}")
    print(f"  - Highest Cross-Validation Accuracy: {best_params['mean_accuracy']:.4f}")
    print("\nThis result was achieved by selecting the top features based on the ERFS weights and training an LDA classifier.")
    print("--- Analysis Complete ---")


if __name__ == '__main__':
    main()
