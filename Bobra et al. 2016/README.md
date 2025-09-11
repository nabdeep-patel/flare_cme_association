# Predicting Coronal Mass Ejections using Machine Learning Methods (Based on Bobra et al., 2016)

This repository contains the datasets, analysis code, and results related to the work of solar flare and coronal mass ejection (CME) associations. The methodology and overall framework are taken from the work of **Bobra et al. (2016)** on CME and Flare Association using machine learning.

---

## Repository Organization

* **Data 2016/**
  * Folder containing the data for 2016 over different time before the flare
  * `data_collection_2016.ipynb`: Jupyter notebook related to data collection.
  * `{class}{time}.csv`: Data files for different timewindow, 
    * _class_ : positive (Eruptive flares), negative (Confined Flares)
    * _time_ : time window for data collection before the onset on flare.
 
* **Data2024/**
  * Folder containing the data for 2024 over different time before the flare
  * `data_collection_2024.ipynb`: Jupyter notebook related to data collection.
  * `{class}{time}.csv`: Data files for different timewindow, 
    * _class_ : positive (Eruptive flares), negative (Confined Flares)
    * _time_ : time window for data collection before the onset on flare.

* **2024.ipynb**
  Contains the analysis for the 2010 - 2024 flare dataset, it includes data fetching, preprocessing and model training using Support Vector Machines. Also the variation of predictive power over time before the onset of flare

* **analysis2016.ipynb**
  * Uses the 2016 data for the time 24hrs before the flare.
  * Describes the ranking of the different SHARP parameters in predicting the target variable
  * Has the Scatter plots for each SHARP Parameter along with its Probability Distribution of the two classes showing the spread of the data

* **CME vs Time.ipynb**
  * Provides an analysis of CME occurrence as a function of time, including visualizations and classification metrics.
  * Contains the analysis using the correct as well as the incorrect normalisation methods.

* **result\_2016.ipynb**
  * It provides the analysis for 24hrs before flare dataset and uses SVM (Support Vector Machines) to determine whether a flaring event will result in a CME or not.
  * It describes the change in results using the correct as well as the incorrect normalisation technique.



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
