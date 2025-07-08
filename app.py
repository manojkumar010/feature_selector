

import os
import pandas as pd
from flask import Flask, request, render_template, send_file, jsonify
import io

# Import our feature selection functions
from erfs import run_erfs_ensemble
from mrmr_logic import calculate_mrmr_features
from extra_selectors import calculate_anova_f_test, calculate_rfe

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'

def create_analysis_sheet(all_results):
    """Creates a DataFrame with a summary analysis of all four methods."""
    
    top_features = {}
    for method in ['ERFS_Weight', 'MRMR_Rank', 'ANOVA_F_Score', 'RFE_Rank']:
        sort_ascending = 'Rank' in method
        top_features[method] = all_results.sort_values(by=method, ascending=sort_ascending).head(10)['Feature_Name'].tolist()

    common_features = set(top_features['ERFS_Weight'])
    for method in top_features:
        common_features.intersection_update(set(top_features[method]))

    analysis_text = [
        ("ERFS Top 10", ", ".join(top_features['ERFS_Weight'])),
        ("MRMR Top 10", ", ".join(top_features['MRMR_Rank'])),
        ("ANOVA F-test Top 10", ", ".join(top_features['ANOVA_F_Score'])),
        ("RFE Top 10", ", ".join(top_features['RFE_Rank'])),
        ("Common Features in All Top 10s", ", ".join(common_features) if common_features else "None"),
        ("ERFS Summary", "Ensemble method, good at finding features that are powerful in combination."),
        ("MRMR Summary", "Selects features that are highly relevant to the class but not redundant with each other."),
        ("ANOVA F-test Summary", "A statistical test that scores features based on their individual correlation with the class. Very fast."),
        ("RFE Summary", "A wrapper method that recursively removes the weakest features to find the most powerful feature subset."),
        ("Interpretation Note", "Features that rank highly across multiple methods are excellent candidates for a final model. Common features are the most robust.")
    ]
    
    return pd.DataFrame(analysis_text, columns=['Analysis Item', 'Details'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_analysis', methods=['POST'])
def run_analysis():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    sheet_name = request.form.get('sheet_name')

    if file.filename == '' or not sheet_name:
        return jsonify({'error': 'Missing file or sheet name'}), 400

    try:
        data_df = pd.read_excel(file, sheet_name=sheet_name, header=0)
        
        # --- Run All Selectors ---
        erfs_results = pd.DataFrame({'Feature_Name': data_df.columns[1:], 'ERFS_Weight': run_erfs_ensemble(data_df.to_numpy())})
        mrmr_results = calculate_mrmr_features(data_df)
        anova_results = calculate_anova_f_test(data_df)
        rfe_results = calculate_rfe(data_df)

        # --- Combine All Results ---
        # Start with a list of all features
        all_features = pd.DataFrame({'Feature_Name': data_df.columns[1:]})
        # Merge all results together
        all_results = all_features.merge(erfs_results, on='Feature_Name', how='left')\
                                .merge(mrmr_results, on='Feature_Name', how='left')\
                                .merge(anova_results, on='Feature_Name', how='left')\
                                .merge(rfe_results, on='Feature_Name', how='left')

        # --- Create Analysis Sheet ---
        analysis_df = create_analysis_sheet(all_results)

        # --- Prepare for Download ---
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            all_results.to_excel(writer, sheet_name='All_Feature_Scores', index=False)
            analysis_df.to_excel(writer, sheet_name='Analysis_Summary', index=False)
        output.seek(0)
        
        return send_file(
            output, 
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True, 
            download_name='feature_selection_analysis.xlsx'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

