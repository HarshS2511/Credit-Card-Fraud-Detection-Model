# Credit Card Fraud Detection & Risk Mitigation Pipeline

## Project Overview
This repository contains a Python-based data analytics and risk classification pipeline designed to identify fraudulent financial transactions. Built using real-world transaction logging schemas, the project implements rigorous Exploratory Data Analysis (EDA), resolves critical class imbalances, and trains an ensemble machine learning classifier to isolate fraud vectors while preserving transaction performance metrics.

## Tech Stack & Dependencies
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Data Visualization:** Matplotlib, Seaborn

## Data Engineering & Pipeline Architecture

### 1. Data Auditing & Cleaning
* Checked and validated integrity checks across the transaction dataframe schema.
* Checked for systemic null values (`.isnull()`) and dropped missing observations to ensure robust model ingestion.
* Conducted duplicate tracking loops to eliminate data redundancies.

### 2. Exploratory Data Analysis (EDA)
* **Feature Correlation:** Generated numeric correlation heatmaps to assess collinearity between transactional balances (`oldbalanceOrg`, `newbalanceOrig`, etc.).
* **Behavioral Distribution Analysis:** Leveraged Seaborn boxplots to isolate transaction amount trends across fraudulent vs. legitimate transaction types.

### 3. Class Imbalance Resolution (Downsampling)
* Because fraud represents a microscopic percentage of global financial transactions, the baseline data suffers from heavy class imbalance.
* Implemented a **Random Under-Sampling (Downsampling)** workflow: Isolated fraudulent entries and drawn an exactly equivalent stratified sample size from the legitimate transactions class to create a perfectly balanced `df_balanced` dataset.

### 4. Model Training & Feature Importance
* Segmented data into a **70/30 Stratified Train-Test Split** to maintain perfect distribution parity.
* Deployed an ensemble **Random Forest Classifier** (100 estimators) to extract non-linear fraud indicators.
* Evaluated feature importance metrics using bar plots to isolate which balance-shifts (`amount`, `oldbalanceOrg`) served as the strongest fraud triggers.

## How to Run the Project
1. Clone this repository to your local machine.
2. Ensure you have the dependencies installed: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Place your source file (`Fraud.csv`) in the root directory.
4. Run the notebook or python script file.
