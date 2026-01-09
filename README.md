# Patient Readmission Prediction

Predicting which patients are at risk of hospital readmission is critical for improving quality of care and reducing avoidable costs. This project builds machine learning models to estimate the probability of readmission for diabetic patients using structured hospital data.

## Project Overview

- **Goal:** Predict whether a discharged patient will be readmitted to hospital (yes/no).
- **Domain:** Healthcare, hospital readmissions, diabetes management.
- **Tech stack:** Python, pandas, scikit-learn, matplotlib, seaborn.
- **ML task:** Binary classification (`readmitted`: yes / no).

The project walks through the full workflow: exploratory data analysis (EDA), feature engineering, model training and evaluation, and interpretation of key risk factors for readmission.[web:98][web:133]

## Dataset

- **Size:** 25,000 hospital encounters.
- **Features:** 17 variables capturing:
  - Patient **age bands** (e.g. [50-60), [60-70), [70-80)).
  - **Utilization history:** number of prior inpatient, outpatient, and emergency visits.
  - **Hospital stay characteristics:** time in hospital, number of lab procedures, procedures, and medications.
  - **Clinical information:** primary/secondary/tertiary diagnosis groups (e.g. Circulatory, Respiratory, Diabetes), medical specialty.
  - **Diabetes management:** glucose tests, A1C tests, treatment changes, and diabetes medications.
- **Target:** `readmitted`  
  - Values: `yes` (readmitted) and `no` (not readmitted).  
  - Class balance: approximately 53% no vs 47% yes.

Source: Public hospital readmission dataset for diabetic patients (Kaggle: “Predicting Hospital Readmissions”).[web:82][web:93]

> To reproduce results, download the dataset from Kaggle and place the CSV in the `data/` folder (for example: `data/hospital_readmissions.csv`).

## Methodology

1. **Exploratory Data Analysis**
   - Inspected shape, data types, and class balance.
   - Explored distributions of key numeric features (time in hospital, number of lab procedures, number of medications, prior visits).
   - Examined readmission rates across age groups, diagnoses, and utilization history.

2. **Preprocessing and Feature Engineering**
   - Separated features and target (`readmitted`).
   - Treated numeric features as continuous:
     - `time_in_hospital`, `n_lab_procedures`, `n_procedures`, `n_medications`.
     - Prior utilization: `n_outpatient`, `n_inpatient`, `n_emergency`.
   - One-hot encoded categorical variables:
     - `age`, `medical_specialty`, `diag_1`, `diag_2`, `diag_3`,
       `glucose_test`, `A1Ctest`, `change`, `diabetes_med`.
   - Used a `ColumnTransformer` and `Pipeline` to keep preprocessing and models reproducible.

3. **Models**
   - **Baseline model:** Logistic Regression.
   - **Tree-based model:** Random Forest Classifier.
   - **Train-test split:** 80% train, 20% test with stratification on target.
   - **Evaluation metrics:** Accuracy, precision, recall, F1-score, ROC-AUC.[web:93][web:119]

All analysis is contained in the Jupyter notebook:

- `notebooks/readmission_analysis.ipynb`

## Results

On the held-out test set (5,000 encounters):

- **Logistic Regression**
  - Accuracy: ~0.61
  - ROC-AUC: ~0.64
  - Readmitted class (1):
    - Precision: ~0.63
    - Recall: ~0.41
    - F1-score: ~0.50

- **Random Forest**
  - Accuracy: ~0.61
  - ROC-AUC: ~0.65
  - Readmitted class (1):
    - Precision: ~0.61
    - Recall: ~0.51
    - F1-score: ~0.55

Random Forest slightly improves recall for readmitted patients compared with the baseline logistic model, while maintaining similar overall ROC-AUC. Logistic Regression remains useful for interpretability, and Random Forest provides a non-linear benchmark.[web:93][web:119]

## Key Insights

Using feature importance from the Random Forest model, the following patterns emerge:

- **Hospital intensity and complexity strongly drive readmission risk.**
  - Higher numbers of **lab procedures**, **medications**, and **hospital procedures** are among the most important predictors, suggesting that more complex or unstable patients are more likely to be readmitted.[web:98][web:133]
  - Longer **time in hospital** is also associated with increased readmission risk.

- **Prior healthcare utilization is a strong risk signal.**
  - More previous **inpatient**, **outpatient**, and **emergency** visits are key predictors of readmission, consistent with literature that heavy utilizers are at higher risk.[web:98][web:116]

- **Clinical profile and age matter.**
  - Diagnoses related to **circulatory conditions** across primary, secondary, and tertiary diagnoses appear frequently among the most important features, highlighting cardiovascular comorbidities as a major driver of readmission.[web:107][web:118]
  - Older age groups, particularly **[70-80)**, are more likely to be readmitted, aligning with known patterns of higher readmission rates in older patients.[web:118]

These insights could help hospitals prioritise follow-up care, medication reconciliation, and discharge planning for high-risk patients.

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/patient-readmission-prediction.git
cd patient-readmission-prediction
