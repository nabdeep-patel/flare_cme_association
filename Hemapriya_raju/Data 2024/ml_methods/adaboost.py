import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier as DT

# Defining the functions
def standardize(flare_data):
    median, std = [], []
    flare_data = np.array(flare_data)
    n_elements = flare_data.shape[0]
    for j in range(flare_data.shape[1]):
        standard_deviation_of_this_feature = np.std(flare_data[:, j])
        median_of_this_feature = np.median(flare_data[:, j])
        median.append(median_of_this_feature)
        std.append(standard_deviation_of_this_feature)

        for i in range(n_elements):
            flare_data[i, j] = (
                flare_data[i, j] - median_of_this_feature) / (standard_deviation_of_this_feature)
    return flare_data, median, std

def fit_data(flare_data, median, std):
    flare_data = np.array(flare_data)
    n_elements = flare_data.shape[0]
    for i in range(n_elements):
        flare_data[i] = (flare_data[i] - median) / std
    return flare_data

# Incorrect Normalisation
final_results = []
for datatype in ['Non_temporal','Temporal']:
    for sampling ,sampling_name in zip([SMOTE, RandomOverSampler, RandomUnderSampler, 'class_weight'], ['SMOTE', 'RandomOverSampler', 'RandomUnderSampler', 'class_weight']):
        
        # Specifying the time
        time_steps = [8, 12, 24, 36, 48]

        for time_step in time_steps:
            # Loading the data
            positive = pd.read_csv(f'G:/My Drive/Latest Research/08-12-2025/Data/{datatype}/positive_df_{time_step}.csv').iloc[:,:-4]
            negative = pd.read_csv(f'G:/My Drive/Latest Research/08-12-2025/Data/{datatype}/negative_df_{time_step}.csv').iloc[:,:-4]

            positive.dropna(inplace=True)
            negative.dropna(inplace=True)

            # Standardisation of data
            positive = standardize(positive)[0]
            negative = standardize(negative)[0]

            xdata = np.vstack([positive, negative])
            ydata = np.hstack([np.ones(positive.shape[0]), np.zeros(negative.shape[0])])

            for n_estimators in [5, 10, 20, 50]:
                for learning_rate in [0.001, 0.01, 0.05, 0.1]:
                    skf = StratifiedKFold(n_splits=10, shuffle=True,random_state=42)
                    tss_scores = []
                    far_scores = []
                    fdr_scores = []
                    pod_scores = []
                    for train_index, test_index in skf.split(xdata, ydata):

                        # Test Train Split
                        xtrain, ytrain = xdata[train_index], ydata[train_index]
                        xtest, ytest = xdata[test_index], ydata[test_index]

                        if sampling_name == 'class_weight':

                            clf = AdaBoostClassifier(estimator=DT(class_weight='balanced'),n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
                            clf.fit(xtrain, ytrain)
                        
                        else:
                            sm = sampling(random_state=42)
                            xtrain, ytrain = sm.fit_resample(xtrain, ytrain)

                            clf = AdaBoostClassifier(estimator=DT(),n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
                            clf.fit(xtrain, ytrain)
                        
                        # Scores
                        TN, FP, FN, TP = confusion_matrix(ytest, clf.predict(xtest)).ravel()
                        if ((TP+FN)==0 or (FP+TN)==0 or (TP+FP)==0):
                            tss_scores.append(np.nan)
                            far_scores.append(np.nan)
                            fdr_scores.append(np.nan)
                            pod_scores.append(np.nan)
                        else:
                            tss_scores.append(TP/(TP+FN) - FP/(FP+TN))
                            far_scores.append(FP/(FP+TN))
                            fdr_scores.append(FP/(TP+FP))
                            pod_scores.append(TP/(TP+FN))
                        print(f"Completed for 'datatype': {datatype}, 'sampling': {sampling_name}, 'time_step': {time_step}, 'n_estimators': {n_estimators}, 'learning_rate': {learning_rate},")
                        
                    # Saving the results
                    final_results.append({'datatype': datatype,'sampling': sampling_name,'time_step': time_step,'n_estimators': n_estimators,'learning_rate': learning_rate,
                        'tss_mean': np.nanmean(tss_scores),'tss_std': np.nanstd(tss_scores),
                        'far_mean': np.nanmean(far_scores),'far_std': np.nanstd(far_scores),
                        'fdr_mean': np.nanmean(fdr_scores),'fdr_std': np.nanstd(fdr_scores),
                        'pod_mean': np.nanmean(pod_scores),'pod_std': np.nanstd(pod_scores)})

# Dataframe
results_incorrect = pd.DataFrame(final_results)
results_incorrect.to_csv("adaboost_incorrect.csv", index=False)

# Correct Normalisation
final_results = []
for datatype in ['Non_temporal','Temporal']:
    for sampling ,sampling_name in zip([SMOTE, RandomOverSampler, RandomUnderSampler, 'class_weight'], ['SMOTE', 'RandomOverSampler', 'RandomUnderSampler', 'class_weight']):
        
        # Specifying the time
        time_steps = [8, 12, 24, 36, 48]

        for time_step in time_steps:
            positive = pd.read_csv(f'G:/My Drive/Latest Research/08-12-2025/Data/{datatype}/positive_df_{time_step}.csv').iloc[:,:-4]
            negative = pd.read_csv(f'G:/My Drive/Latest Research/08-12-2025/Data/{datatype}/negative_df_{time_step}.csv').iloc[:,:-4]

            positive.dropna(inplace=True)
            negative.dropna(inplace=True)

            xdata = np.vstack([positive, negative])
            ydata = np.hstack([np.ones(positive.shape[0]), np.zeros(negative.shape[0])])

            for n_estimators in [5, 10, 20, 50]:
                for learning_rate in [0.001, 0.01, 0.05, 0.1]:
                    skf = StratifiedKFold(n_splits=10, shuffle=True,random_state=42)
                    tss_scores = []
                    far_scores = []
                    fdr_scores = []
                    pod_scores = []
                    for train_index, test_index in skf.split(xdata, ydata):

                        # Test Train Split
                        xtrain, ytrain = xdata[train_index], ydata[train_index]
                        xtest, ytest = xdata[test_index], ydata[test_index]

                        # Standardisation of Data
                        xtrain, median, std = standardize(xtrain)
                        xtest = fit_data(xtest, median, std)

                        if sampling_name == 'class_weight':

                            clf = AdaBoostClassifier(estimator=DT(class_weight='balanced'),n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
                            clf.fit(xtrain, ytrain)
                        
                        else:
                            sm = sampling(random_state=42)
                            xtrain, ytrain = sm.fit_resample(xtrain, ytrain)

                            clf = AdaBoostClassifier(estimator=DT(),n_estimators=n_estimators, learning_rate=learning_rate,  random_state=42)
                            clf.fit(xtrain, ytrain)
                        
                        # Scores
                        TN, FP, FN, TP = confusion_matrix(ytest, clf.predict(xtest)).ravel()
                        if ((TP+FN)==0 or (FP+TN)==0 or (TP+FP)==0):
                            tss_scores.append(np.nan)
                            far_scores.append(np.nan)
                            fdr_scores.append(np.nan)
                            pod_scores.append(np.nan)
                        else:
                            tss_scores.append(TP/(TP+FN) - FP/(FP+TN))
                            far_scores.append(FP/(FP+TN))
                            fdr_scores.append(FP/(TP+FP))
                            pod_scores.append(TP/(TP+FN))
                        print(f"Completed for 'datatype': {datatype}, 'sampling': {sampling_name}, 'time_step': {time_step}, 'n_estimators': {n_estimators}, 'learning_rate': {learning_rate}")

                    # Saving the results
                    final_results.append({'datatype': datatype,'sampling': sampling_name,'time_step': time_step,'n_estimators': n_estimators,'learning_rate': learning_rate,
                        'tss_mean': np.nanmean(tss_scores),'tss_std': np.nanstd(tss_scores),
                        'far_mean': np.nanmean(far_scores),'far_std': np.nanstd(far_scores),
                        'fdr_mean': np.nanmean(fdr_scores),'fdr_std': np.nanstd(fdr_scores),
                        'pod_mean': np.nanmean(pod_scores),'pod_std': np.nanstd(pod_scores)})

# Dataframe
results_correct = pd.DataFrame(final_results)
results_correct.to_csv("adaboost_correct.csv", index=False)