Using Machine Learning Methods to Forecast if Solar Flares Will Be Associated with CMEs and SEPs (Inceoglu et al. 2018)

This repository contains the datasets, analysis code, and machine learning model implementations for predicting solar flare and coronal mass ejection (CME) associations. The work applies various machine learning algorithms to analyze SHARP parameters and their relationship with CME occurrence.

---

## Repository Organization

### 2018 Analysis/
* **Data 2018/**
  * Contains the datasets used for the 2018 analysis
  * Includes feature data for training and evaluating machine learning models

* **models/**
  * Trained machine learning models and model artifacts
  * Saved model files for deployment and evaluation

* **analysis.ipynb**
  * Comprehensive analysis and evaluation of all machine learning models
  * Includes model comparisons, performance metrics, and feature importance analysis

* **data_variability.ipynb**
  * Analysis of data variability and feature distributions
  * Exploratory data analysis and statistical characterizations

### 2024 Analysis/
* **Data 2024/**
  * Contains the datasets used for the 2024 analysis
  * Includes feature data for training and evaluating machine learning models

* **models/**
  * Trained machine learning models and model artifacts
  * Saved model files for deployment and evaluation

* **analysis.ipynb**
  * Analysis and evaluation of machine learning models on 2024 flare dataset
  * Includes model comparisons, performance metrics, and feature importance analysis

* **data_variability.ipynb**
  * Analysis of data variability and feature distributions for 2024 dataset
  * Exploratory data analysis and statistical characterizations


---

## Methodology and Tools

* The analysis is conducted in **Python**, primarily using the following libraries:
  `pandas`, `numpy`, `matplotlib`, `scikit-learn`, and `sunpy`.
* Certain workflows are configured for execution in **Google Colab**, however all the files access had been made available in this github repository.
* The datasets are systematically organized by **year** and by **prediction time window**, in alignment with the classification experiments.

---

## Reference

Inceoglu, F., et al. (2018). *Using Machine Learning Methods to Forecast if Solar Flares Will Be Associated with CMEs and SEPs*.  
[DOI: https://doi.org/10.3847/1538-4357/aac81e](https://doi.org/10.3847/1538-4357/aac81e)]
