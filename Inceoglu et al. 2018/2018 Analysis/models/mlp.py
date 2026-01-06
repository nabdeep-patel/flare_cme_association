# Importing the modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# Defining the functions
def standardize(flare_data):
  median = []
  std = []
  flare_data = np.array(flare_data)
  n_elements = flare_data.shape[0]
  for j in range(flare_data.shape[1]):
    standard_deviation_of_this_feature = np.std(flare_data[:, j])
    median_of_this_feature = np.median(flare_data[:, j])
    std.append(standard_deviation_of_this_feature)
    median.append(median_of_this_feature)
    for i in range(n_elements):
        flare_data[i, j] = (flare_data[i, j] - median_of_this_feature) / (standard_deviation_of_this_feature)
  return flare_data, median, std

def fit_data(flare_data, median, std):
    flare_data = np.array(flare_data)
    for j in range(flare_data.shape[1]):
        flare_data[:, j] = (flare_data[:, j] - median[j]) / std[j]
    return flare_data

# Training with MLP for incorrect normalisation
tss_mean = []
tss_std = []
hss_mean = []
hss_std = []
param_layer = []
param_alpha = []
param_tol = []
method = []
time_list = []

# Time Series Loop
for i in [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]:
    print(f"\nProcessing data for {i} hrs before flare .........")

    # Loading data
    flare_linked = pd.read_csv(f"https://raw.githubusercontent.com/nabdeep-patel/flare_cme_association/refs/heads/main/Inceoglu/jsoc/flare_linked_{i}.csv").iloc[:, :-4]
    only_cme = pd.read_csv(f"https://raw.githubusercontent.com/nabdeep-patel/flare_cme_association/refs/heads/main/Inceoglu/jsoc/only_cme_{i}.csv").iloc[:, :-4]
    only_flares = pd.read_csv(f"https://raw.githubusercontent.com/nabdeep-patel/flare_cme_association/refs/heads/main/Inceoglu/jsoc/only_flares_{i}.csv").iloc[:, :-4]

    # Standardizing data
    only_flares_st = standardize(only_flares)[0]
    flare_linked_st = standardize(flare_linked)[0]
    only_cme_st = standardize(only_cme)[0]

    for name in ["only_flares", "flare_linked", "only_cme"]:

        # Creating features and labels
        xdata = np.vstack((only_flares_st, flare_linked_st, only_cme_st))
        if name == "only_flares":
            ydata = np.hstack((np.ones(len(only_flares_st)), np.zeros(len(flare_linked_st) + len(only_cme_st))))
        elif name == "flare_linked":
            ydata = np.hstack((np.zeros(len(only_flares_st)), np.ones(len(flare_linked_st)), np.zeros(len(only_cme_st))))
        else:
            ydata = np.hstack((np.zeros(len(only_flares_st) + len(flare_linked_st)), np.ones(len(only_cme_st))))

        for layer in [18, 36, 54, 72, 90, 108]:
            for alpha in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
                for tol in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
                  skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
                  tss_scores = []
                  hss_scores = []

                  for train_index, test_index in skf.split(xdata, ydata):
                      xtrain, xtest = xdata[train_index], xdata[test_index]
                      ytrain, ytest = ydata[train_index], ydata[test_index]

                      clf = MLPClassifier(hidden_layer_sizes=(layer,), alpha = alpha, tol = tol, activation='relu', solver='adam', max_iter=1000, random_state=42)
                      clf.fit(xtrain, ytrain)
                      ypred = clf.predict(xtest)

                      TN, FP, FN, TP = confusion_matrix(ytest, ypred).ravel()

                      if (TP + FN == 0 or FP + TN == 0):
                          tss_scores.append(np.nan)
                          continue

                      # TSS
                      tss = TP / (TP + FN) - FP / (FP + TN)
                      tss_scores.append(tss)

                      # HSS
                      numerator = 2 * (TP * TN - FP * FN)
                      denominator = ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))
                      hss = numerator / denominator if denominator != 0 else np.nan
                      hss_scores.append(hss)

                  # Store average scores
                  method.append(name)
                  tss_mean.append(round(np.nanmean(tss_scores), 4))
                  tss_std.append(round(np.nanstd(tss_scores), 4))
                  hss_mean.append(round(np.nanmean(hss_scores), 4))
                  hss_std.append(round(np.nanstd(hss_scores), 4))
                  param_layer.append(layer)
                  param_alpha.append(alpha)
                  param_tol.append(tol)
                  time_list.append(i)
                  print(f"Completed for {name},{i} with layers={layer}, alpha={alpha}, tol={tol}")

# Create the final DataFrame
df = pd.DataFrame({"Method": method,"Time": time_list,"Layers": param_layer,"Alpha": param_alpha,"Tolerance": param_tol,"TSS Mean": tss_mean,"TSS Std": tss_std,"HSS Mean": hss_mean,"HSS Std": hss_std})

# Save the file
df.to_csv("results_incorrect.csv", index=False)

# Training with MLP for correct normalisation
tss_mean = []
tss_std = []
hss_mean = []
hss_std = []
param_layer = []
param_alpha = []
param_tol = []
method = []
time_list = []

# Time Series Loop
for i in [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]:
    print(f"\nProcessing data for {i} hrs before flare .........")

    # Loading data
    flare_linked = pd.read_csv(f"https://raw.githubusercontent.com/nabdeep-patel/flare_cme_association/refs/heads/main/Inceoglu/jsoc/flare_linked_{i}.csv").iloc[:, :-4]
    only_cme = pd.read_csv(f"https://raw.githubusercontent.com/nabdeep-patel/flare_cme_association/refs/heads/main/Inceoglu/jsoc/only_cme_{i}.csv").iloc[:, :-4]
    only_flares = pd.read_csv(f"https://raw.githubusercontent.com/nabdeep-patel/flare_cme_association/refs/heads/main/Inceoglu/jsoc/only_flares_{i}.csv").iloc[:, :-4]

    for name in ["only_flares", "flare_linked", "only_cme"]:

        # Creating features and labels
        xdata = np.vstack((only_flares, flare_linked, only_cme))
        if name == "only_flares":
            ydata = np.hstack((np.ones(len(only_flares)), np.zeros(len(flare_linked) + len(only_cme))))
        elif name == "flare_linked":
            ydata = np.hstack((np.zeros(len(only_flares)), np.ones(len(flare_linked)), np.zeros(len(only_cme))))
        else:
            ydata = np.hstack((np.zeros(len(only_flares) + len(flare_linked)), np.ones(len(only_cme))))

        for layer in [18, 36, 54, 72, 90, 108]:
            for alpha in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
                for tol in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
                  skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
                  tss_scores = []
                  hss_scores = []

                  for train_index, test_index in skf.split(xdata, ydata):
                      xtrain, xtest = xdata[train_index], xdata[test_index]
                      ytrain, ytest = ydata[train_index], ydata[test_index]

                      xtrain_norm, median, std = standardize(xtrain)
                      xtest_norm = fit_data(xtest, median, std)

                      clf = MLPClassifier(hidden_layer_sizes=(layer,), alpha = alpha, tol = tol, activation='relu', solver='adam', max_iter=1000, random_state=42)
                      clf.fit(xtrain_norm, ytrain)
                      ypred = clf.predict(xtest_norm)

                      TN, FP, FN, TP = confusion_matrix(ytest, ypred).ravel()

                      if (TP + FN == 0 or FP + TN == 0):
                          tss_scores.append(np.nan)
                          continue

                      # TSS
                      tss = TP / (TP + FN) - FP / (FP + TN)
                      tss_scores.append(tss)

                      # HSS
                      numerator = 2 * (TP * TN - FP * FN)
                      denominator = ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))
                      hss = numerator / denominator if denominator != 0 else np.nan
                      hss_scores.append(hss)

                  # Store average scores
                  method.append(name)
                  tss_mean.append(round(np.nanmean(tss_scores), 4))
                  tss_std.append(round(np.nanstd(tss_scores), 4))
                  hss_mean.append(round(np.nanmean(hss_scores), 4))
                  hss_std.append(round(np.nanstd(hss_scores), 4))
                  param_layer.append(layer)
                  param_alpha.append(alpha)
                  param_tol.append(tol)
                  time_list.append(i)
                  print(f"Completed for {name},{i} with layers={layer}, alpha={alpha}, tol={tol}")

# Create the final DataFrame
df = pd.DataFrame({"Method": method,"Time": time_list,"Layers": param_layer,"Alpha": param_alpha,"Tolerance": param_tol,"TSS Mean": tss_mean,"TSS Std": tss_std,"HSS Mean": hss_mean,"HSS Std": hss_std})

# Save the file
df.to_csv("results_correct.csv", index=False)