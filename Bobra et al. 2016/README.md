# Predicting Coronal Mass Ejections using Machine Learning Methods (Based on Bobra et al., 2016)

This repository contains the datasets, analysis code, and results related to the work of solar flare and coronal mass ejection (CME) associations. The methodology and overall framework are taken from the work of **Bobra et al. (2016)** on CME and Flare Association using machine learning.

---

## Repository Organization

### 2016 Analysis/
* **2016 Analysis/Data 2016/**
  * Folder containing the data for 2016 over different time windows before the flare
  * `data_collection_2016.ipynb`: Jupyter notebook related to data collection.
  * `{class}{time}.csv`: Data files for different time windows, 
    * _class_ : positive (Eruptive flares), negative (Confined Flares)
    * _time_ : time window (in hours) for data collection before the onset of flare.

* **analysis2016.ipynb**
  * Uses the 2016 data for the time 24hrs before the flare.
  * Describes the ranking of the different SHARP parameters in predicting the target variable
  * Includes scatter plots for each SHARP Parameter along with its Probability Distribution of the two classes showing the spread of the data

* **CME vs Time.ipynb**
  * Provides an analysis of CME occurrence as a function of time, including visualizations and classification metrics.
  * Contains the analysis using both the correct and incorrect normalisation methods.

* **result_2016.ipynb**
  * Provides the analysis for 24hrs before flare dataset and uses SVM (Support Vector Machines) to determine whether a flaring event will result in a CME or not.
  * Describes the change in results using both the correct and incorrect normalisation techniques.

### 2024 Analysis/
* **2024 Analysis/Data 2024/**
  * Folder containing the data for 2024 over different time windows before the flare
  * `data_collection_2024.ipynb`: Jupyter notebook related to data collection.
  * `{class}{time}.csv`: Data files for different time windows, 
    * _class_ : positive (Eruptive flares), negative (Confined Flares)
    * _time_ : time window (in hours) for data collection before the onset of flare.

* **analysis2024.ipynb**
  * Contains the analysis for the 2024 flare dataset, including data fetching, preprocessing and model training using Support Vector Machines.
  * Analyzes the variation of predictive power over time before the onset of flare.

* **CME vs Time.ipynb**
  * Provides an analysis of CME occurrence as a function of time, including visualizations and classification metrics.
  * Contains the analysis using both the correct and incorrect normalisation methods.

* **result_2024.ipynb**
  * Provides the analysis for 24hrs before flare dataset and uses SVM (Support Vector Machines) to determine whether a flaring event will result in a CME or not.
  * Describes the change in results using both the correct and incorrect normalisation techniques.

### Analysis_with_Bobra's_data/
* **result_bobra_data.ipynb**
  * Analysis using the original data from Bobra et al. (2016)
  * Reproduces results and validates methodology against the original study using the exact flares data from the published results and studying the changes using correct normalisation.


---

## Methodology and Tools

* The analysis is conducted in **Python**, primarily using the following libraries:
  `pandas`, `numpy`, `matplotlib`, `scikit-learn`, and `sunpy`.
* Certain workflows are configured for execution in **Google Colab**, however all the files access had been made available in this github repository.
* The datasets are systematically organized by **year** and by **prediction time window**, in alignment with the classification experiments.

---

## Reference

Bobra, M. G., et al. (2016). *Predicting Coronal Mass Ejections using Machine Learning Methods*.  
[DOI: http://dx.doi.org/10.3847/0004-637X/821/2/127](http://dx.doi.org/10.3847/0004-637X/821/2/127)]
