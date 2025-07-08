
import numpy as np
import pandas as pd

def proposed_approach_python(data: np.ndarray):
    """
    Python translation of the `proposed_approach.m` script.
    Calculates feature weights based on the overlap of class distributions.
    """
    n_samples, n_features_total = data.shape
    dim = n_features_total - 1
    
    class_labels = data[:, 0].astype(int)
    features = data[:, 1:]
    
    unique_classes = np.unique(class_labels)
    n_class = len(unique_classes)
    
    if n_class <= 1:
        return np.zeros(dim)

    class_prob = np.array([np.sum(class_labels == c) for c in unique_classes])
    cl_fac = class_prob / np.sum(class_prob)
    
    area_coeff = np.zeros(dim)

    for i in range(dim):
        overlap_area = 0
        feature_vector = features[:, i]
        cl_range = np.zeros((n_class, 2))
        
        for j, cls in enumerate(unique_classes):
            idx = np.where(class_labels == cls)[0]
            if len(idx) > 1:
                mu = np.mean(feature_vector[idx])
                sigma = np.std(feature_vector[idx], ddof=1)
            else:
                mu = feature_vector[idx][0] if len(idx) > 0 else 0
                sigma = 0
            
            range_val = (1 - cl_fac[j]) * 1.732 * sigma
            cl_range[j, 0] = mu - range_val
            cl_range[j, 1] = mu + range_val

        s_cl_range = cl_range[np.argsort(cl_range[:, 0])]
        
        for j in range(n_class - 1):
            for k in range(j + 1, n_class):
                if (s_cl_range[j, 1] - s_cl_range[k, 0]) > 0:
                    overlap_area += abs(s_cl_range[j, 1] - s_cl_range[k, 0])
        
        total_range = np.max(s_cl_range[:, 1]) - np.min(s_cl_range[:, 0])
        if total_range > 0:
            area_coeff[i] = overlap_area / total_range
        else:
            area_coeff[i] = 0

    area_coeff_scaled = area_coeff / np.max(area_coeff) if np.max(area_coeff) > 0 else area_coeff
    wt = 1 - area_coeff_scaled
    wt[np.isnan(wt)] = 0
    
    final_weight = wt / np.max(wt) if np.max(wt) > 0 else wt
    return final_weight

def run_erfs_ensemble(data: np.ndarray, n_iterations=50, subset_fraction=0.8):
    """
    Implements the full "Ensemble of Random Feature Subsets" (ERFS) approach.
    """
    n_samples, n_features_total = data.shape
    total_features = n_features_total - 1
    all_weights = np.zeros((n_iterations, total_features))
    feature_indices = np.arange(total_features)

    for i in range(n_iterations):
        n_subset = int(total_features * subset_fraction)
        subset_indices = np.random.choice(feature_indices, n_subset, replace=False)
        
        class_column = data[:, 0].reshape(-1, 1)
        feature_subset = data[:, subset_indices + 1]
        data_subset = np.hstack([class_column, feature_subset])
        
        subset_weights = proposed_approach_python(data_subset)
        all_weights[i, subset_indices] = subset_weights

    final_weights = np.mean(all_weights, axis=0)
    return final_weights
