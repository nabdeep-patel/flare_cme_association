# Interpretable ML-Based Forecasting of CMEs Associated with Flares (Raju et al. 2023)

This repository contains the datasets, analysis code, and machine learning model implementations for predicting solar flare and coronal mass ejection (CME) associations. The work applies various machine learning algorithms to analyze SHARP parameters and their relationship with CME occurrence.

---

## Repository Organization

### 2018 Analysis/
* **Data/**
  * Contains temporal and non-temporal feature datasets for 2018
  * `Temporal/`: Time-series features organized by time windows before flare onset
  * `Non_temporal/`: Static features extracted from active regions

* **ml_method/**
  * Machine learning model implementations and evaluation results based on correct and incorrect normalisation
  * Python scripts for various algorithms:
    * `adaboost.py` - AdaBoost classifier
    * `dt.py` - Decision Tree classifier
    * `gradientboost.py` - Gradient Boosting classifier
    * `lda.py` - Linear Discriminant Analysis
    * `lr.py` - Logistic Regression
    * `rf.py` - Random Forest classifier
    * `svm.py` - Support Vector Machine
    * `xb.py` - XGBoost classifier
  * CSV files storing correct and incorrect predictions for each model:
    * `{model}_correct.csv` - Correctly classified instances
    * `{model}_incorrect.csv` - Misclassified instances

* **analysis.ipynb**
  * Comprehensive analysis and evaluation of all machine learning models
  * Includes model comparisons, performance metrics, and feature importance analysis

### 2024 Analysis/
* **Data/**
  * Contains temporal and non-temporal feature datasets for 2024
  * `Temporal/`: Time-series features organized by time windows before flare onset
  * `Non_temporal/`: Static features extracted from active regions

* **ml_methods/**
  * Machine learning model implementations and evaluation results for 2024 dataset
  * Similar structure to 2018 Analysis/ml_method/ with updated models and predictions

* **analysis.ipynb**
  * Analysis and evaluation of machine learning models on 2024 flare dataset
  * Includes model comparisons, performance metrics, and feature importance analysis


---

## Methodology and Tools

* The analysis is conducted in **Python**, primarily using the following libraries:
  `pandas`, `numpy`, `matplotlib`, `scikit-learn`, and `sunpy`.
* Certain workflows are configured for execution in **Google Colab**, however all the files access had been made available in this github repository.
* The datasets are systematically organized by **year** and by **prediction time window**, in alignment with the classification experiments.

---

## Reference

Raju, H., et al. (2023). *Interpretable ML-Based Forecasting of CMEs Associated with Flares*.  
[DOI: https://doi.org/10.1007/s11207-023-02187-6](https://doi.org/10.1007/s11207-023-02187-6)]
