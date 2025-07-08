
# Comprehensive Feature Selection Analysis Tool

This web application provides a user-friendly interface to analyze feature importance in a dataset using four different powerful selection algorithms. It is designed for researchers and data scientists who need to compare various feature selection techniques to build robust machine learning models.

## Features

- **Simple Web Interface:** Upload your dataset and get results without writing any code.
- **Four Selection Algorithms:** Compares the results from:
  1.  **ERFS (Ensemble of Random Feature Subsets):** A robust method for finding features that are powerful in combination.
  2.  **mRMR (Minimum Redundancy Maximum Relevance):** Selects features that are highly relevant to the class but not redundant with each other.
  3.  **ANOVA F-test:** A fast statistical test to score features based on their individual correlation with the class.
  4.  **RFE (Recursive Feature Elimination):** A wrapper method that recursively removes the weakest features to find the most powerful feature subset.
- **Comprehensive Output:** Downloads an Excel file with:
  - A sheet containing all feature scores and ranks from all four methods.
  - An analysis summary sheet that explains the results and highlights the top features.

## How to Use

### Prerequisites

- Python 3.7+
- Pip (Python package installer)

### Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-folder>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Start the Flask server:**
    ```bash
    python app.py
    ```

2.  **Open your web browser** and navigate to:
    [http://127.0.0.1:5000](http://127.0.0.1:5000)

3.  **Upload your data:**
    - Your data must be in an Excel (.xlsx) file.
    - The first row must be the header (column names).
    - The first column must be the target/class variable (e.g., "Groups").

4.  **Enter the sheet name** you want to analyze.

5.  **Click "Run Analysis"** and wait for the results to be downloaded.

## Project Structure

```
.gitignore
README.md
app.py
erfs.py
extra_selectors.py
mrmr_logic.py
requirements.txt
templates/
    index.html
```
