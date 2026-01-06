# Importing the modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC 
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


# Training with SVM for correct normalisation
method = []
tss_mean = []
tss_std = []
hss_mean = []
hss_std = []
param_C = []
param_gamma = []
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

        for C in [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]:
            for gamma in [1, 0.1, 0.01, 0.07, 0.001, 0.005, 0.0001]:
                skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
                tss_scores = []
                hss_scores = []

                for train_index, test_index in skf.split(xdata, ydata):
                    xtrain, xtest = xdata[train_index], xdata[test_index]
                    ytrain, ytest = ydata[train_index], ydata[test_index]

                    xtrain_norm, median, std = standardize(xtrain)
                    xtest_norm = fit_data(xtest, median, std)

                    clf = SVC(class_weight="balanced", C=C, gamma=gamma, kernel='rbf', probability=True)
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
                param_C.append(C)
                param_gamma.append(gamma)
                time_list.append(i)

                print(f"Completed for {name},{i} with C={C} and gamma={gamma}")

# Create the final DataFrame
df = pd.DataFrame({"Method": method,"Time": time_list,"C": param_C,"Gamma": param_gamma,"TSS Mean": tss_mean,"TSS Std": tss_std,"HSS Mean": hss_mean,"HSS Std": hss_std})

# Save the file
df.to_csv("results_correct.csv", index=False)