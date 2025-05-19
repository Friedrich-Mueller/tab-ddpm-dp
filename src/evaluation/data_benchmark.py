import os
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy as sp

import tensorflow as tf
from tensorflow.keras import Model, Input, losses, layers, optimizers

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
import seaborn as sns

import itertools

from src.datasets import adults

def binning_borders(series, nbin, drop_duplicates=True):
    """Returns a list of bin borders
    e.g. generates a list of 10 values for 9 bins (for [0,1])
    """
    
    bin_borders = [0]
    
    vals = np.sort(series.to_numpy())
    if drop_duplicates==True:
        vals = list(dict.fromkeys(vals)) # drop duplicates

    bin_depth = len(vals)/nbin
    for i in range(1, nbin):
        border_value = vals[int(i*bin_depth)]
        bin_borders.append(border_value)
    bin_borders.append(1.0)
    
    return bin_borders

def value_to_bin_index(x, bin_borders):
    """Returns the index of the bin a value belongs to
    e.g. a list with 10 binning borders consists of 9 bins/indices
    """
    for i in range(len(bin_borders)-1):
        if (x>=bin_borders[i] and x<bin_borders[i+1]):
            return i+1
    if x == 1.0:
        return 19
    if x < 0:
        return 0
    if x > 1.0:
        return 20
    raise Exception("value cannot belong to any bin", x, i, bin_borders, len(bin_borders))
    
def cramers_v_corrected(ftable):
    """Computes Cramer's V with bias correction
    to measure the correlation between two attributes.
    Input is a frequency table between 2 attributes.
    Returns Cramers V value € [0, 1]
    """
    n = np.sum(ftable)
    k, r = ftable.shape
    X2 = sp.stats.chi2_contingency(ftable)[0] # Pearson's chi-squared test statistic
    phi2 = X2 / n
    
    phitilde2 = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    ktilde = k - (k - 1)**2 / (n - 1)
    rtilde = r - (r - 1)**2 / (n - 1)

    V = np.sqrt(phitilde2 / min(ktilde - 1, rtilde - 1))
       
    return V

def TVD_ind(data_orig, data_synth, cont_cols, cat_cols, nbin=19, drop_duplicates=True):
    """ 
    Generates equal-depth bins based on the original data.
    Calculates the TVD between two one-dimensional distributions.
    Returns the 1 minus the average TVD over all marginals.
    """
    
    tvds = []
    
    # calculate TVDs for cont columns
    for col in cont_cols:
        
        # discretize/bin based on original data
        bin_borders = binning_borders(data_orig[col], nbin, drop_duplicates)

        vals_orig = data_orig[col].to_numpy()
        vals_synth = data_synth[col].to_numpy()
        bin_counts_orig = []
        bin_counts_synth = []
        for i in range(nbin):
            
            col_bin_orig = [x for x in vals_orig if (x >= bin_borders[i] and x < bin_borders[i+1])]
            if i == (nbin-1):
                col_bin_orig = [x for x in vals_orig if (x >= bin_borders[i] and x <= bin_borders[i+1])]
            
            col_bin_synth = [x for x in vals_synth if (x >= bin_borders[i] and x < bin_borders[i+1])]
            if i == (nbin-1):
                col_bin_synth = [x for x in vals_synth if (x >= bin_borders[i] and x <= bin_borders[i+1])]            
            
            bin_counts_orig.append(len(col_bin_orig))
            bin_counts_synth.append(len(col_bin_synth))
            
        # add additional bins to each end of the range for synth data outside of range
        bin_counts_orig = [0] + bin_counts_orig + [0]
        bin_counts_synth = [len([x for x in vals_synth if x < bin_borders[0]])] + bin_counts_synth + [len([x for x in vals_synth if x > bin_borders[nbin]])]
            
        # calculate TVD for columns
        a = np.array(bin_counts_orig/np.sum(bin_counts_orig)).reshape(nbin+2,1)
        b = np.array(bin_counts_synth/np.sum(bin_counts_synth)).reshape(nbin+2,1)
        tvds.append(0.5 * sum(abs(a - b))[0])
        
    # calculate TVDs for cat columns
    for col in cat_cols:

        counts_orig_lst = []
        counts_synth_lst = []
        
        counts_orig = data_orig[col].value_counts(normalize=True)
        counts_synth = data_synth[col].value_counts(normalize=True)
        
        # align counts and treat non-occuring cat values in synth data
        for index, value in counts_orig.items():
            counts_orig_lst.append(value)
            if index in counts_synth:
                counts_synth_lst.append(counts_synth[index])
            else:
                counts_synth_lst.append(0)
                
        # calculate TVD for column
        a = np.array(counts_orig_lst).reshape(-1,1)
        b = np.array(counts_synth_lst).reshape(-1,1)
        tvds.append(0.5 * sum(abs(a - b))[0]) 
                
    print('\n1-TVD (Ind)')
    print(1-np.average(tvds))
    return 1-np.average(tvds)

def TVD_pair(data_orig, data_synth, cont_cols, cat_cols, nbin=19, drop_duplicates=True):
    """
    Generates equal-depth bins based on the original data.
    Calculates the TVD for each two-way marginal.
    Returns the 1 minus the average TVD over all marginals.
    """
    
    tvds = []
        
    data_o = data_orig.copy()
    data_s = data_synth.copy()
    
    # discretize cont cols
    for col in cont_cols:
               
        # bin based on original data
        bin_borders = binning_borders(data_o[col], nbin, drop_duplicates)
        data_o[col] = data_o[col].apply(value_to_bin_index, bin_borders=bin_borders)
        data_s[col] = data_s[col].apply(value_to_bin_index, bin_borders=bin_borders)
        
        # bin counts
        counts_orig = data_o[col].value_counts(normalize=False)
        counts_synth = data_s[col].value_counts(normalize=False)
        
    cols = cont_cols + cat_cols
    
    for i in range(len(cols)-1):
        for j in range(i+1, len(cols)):
            
            marginals1 = pd.crosstab(data_o[cols[i]], data_o[cols[j]], margins=False, normalize=True)
            marginals2 = pd.crosstab(data_s[cols[i]], data_s[cols[j]], margins=False, normalize=True)
            
            marginals_subtracted = marginals1.subtract(marginals2)
            
            # drop NaNs which occur when subtraction includes value counts of 0 (e.g.  Holand-Netherlands 1 )
            marginals_subtracted = marginals_subtracted.dropna(axis=1, how='all')
            marginals_subtracted = marginals_subtracted.dropna(axis=0, how='all')

            
            tvds.append(0.5*marginals_subtracted.abs().to_numpy().sum())
    
    print('\n1-TVD (Pair)')
    print(1-np.average(tvds))
    return 1-np.average(tvds)

def cramer_v_levels(data, cat_cols, cont_cols=None):
    """
    Generates a list of correlation values for each pair of columns of a given dataset.
    Sticks to the convention of 4 degrees of correlation, depending on the magnitude of V.
    [0, .1) is low
    [.1, .3) is weak
    [.3, .5) is middle
    [.5, 1) is strong
    """
    
    V_levels = []
    
#     data = df.copy()
    
#     # discretize cont cols
#     for col in cont_cols:
               
#         # bin based on original data
#         bin_borders = binning_borders(data[col], nbin, drop_duplicates)
#         data[col] = data[col].apply(value_to_bin_index, bin_borders=bin_borders)
        
# #         counts_orig = data_o[col].value_counts(normalize=True)
#         counts = data[col].value_counts(normalize=False)

        
        
#         print(' ')
#         print(col)
#         print(bin_borders) 
#         print(counts)
        
    cols = cat_cols
    
    for i in range(len(cols)):
        row = []
        for j in range(len(cols)):
            
            frequency_table = pd.crosstab(data[cols[i]], data[cols[j]])
            
            V = cramers_v_corrected(frequency_table.to_numpy())
            
            # assign levels to values
            if V > 0.5:
                row.append(1)
            elif V > 0.3:
                row.append(0.5)
            elif V > 0.1:
                row.append(0.3)
            else:
                row.append(0.1)
        V_levels.append(row)

    return np.array(V_levels)

def cramer_v_coracc(Vs_orig, Vs_synth):
    """Returns CorAcc metric reporting fraction of pairs where
    synth and orig data assign same correlation level.
    """
    h = len(Vs_orig)
    n = Vs_orig.size
    equals = ((Vs_orig == Vs_synth).sum() - h ) / 2
    total = ((h**2)-h)/2
    
    return equals/total

def cramer_corr_heatmap_cat(data_o, data_s, cat_cols):
    
    cramer_Vs_o = cramer_v_levels(data_o, cat_cols)
    cramer_Vs_s = cramer_v_levels(data_s, cat_cols)
    
    df_o = pd.DataFrame(cramer_Vs_o, index=cat_cols, columns=cat_cols)
    df_s = pd.DataFrame(cramer_Vs_s, index=cat_cols, columns=cat_cols)

    corr_acc = "{:.2f}".format(cramer_v_coracc(cramer_Vs_o, cramer_Vs_s))
    norm = BoundaryNorm([0, 0.1000001, 0.3000001, 0.5000001, 1.0],5)
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, gridspec_kw=dict(width_ratios=[2,2.5]), figsize=(13, 5))
    
    ax1.set_title("Cramer V Correlation between Variables\n Original Data CorAcc = " + str(1));
    sns.heatmap(df_o, cmap=sns.cubehelix_palette(5, as_cmap=False), linewidths=0.2, cbar=False, ax=ax1)
    
    ax2.set_title("Cramer V Correlation between Variables\n Synthetic Data CorAcc = " + str(corr_acc)); 
    sns.heatmap(df_s, cmap=sns.cubehelix_palette(5, as_cmap=False), linewidths=0.2, cbar=True, ax=ax2,
                norm=norm, cbar_kws=dict(spacing="proportional") )   
    
    fig.subplots_adjust(wspace=0.4)

    plt.show()

def XGBoost_classifier_F1(data_orig, data_synth):
    """
    Generic XGBoost classifier.
    Expects normalized continuous and numeric categorical data in numpy array format.

    Runs the classification task:
    - once trained on and classified with the original data
    - once trained on synthetic data and classified with original data
    """
    
    ### train on and classify original data
    print("\nOriginal data F1 macro score:")
    X_train = data_orig.iloc[:, 0:-1]
    y_train = data_orig.iloc[:, -1]
    
    X_test = X_train
    y_test = y_train
    
    # run with numeric categorical features
        
    # fit model to training data
    xgb_model = XGBClassifier().fit(X_train, y_train)

    # make predictions for test data
    y_pred = xgb_model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    f1 = f1_score(y_test, predictions, average='macro') *100
    print("F1: %.2f%%" % (f1))       
        
    #### train on synthetic data and classify original data
    print("\nSynthetic data F1 macro score:")
    X_train = data_synth.iloc[:, 0:-1]
    y_train = data_synth.iloc[:, -1]
    
    X_test = data_orig.iloc[:, 0:-1]
    y_test = data_orig.iloc[:, -1]
    
    # run with numeric categorical features
    
    # fit model to training data
    xgb_model = XGBClassifier().fit(X_train, y_train)

    # make predictions for test data
    y_pred = xgb_model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    f1 = f1_score(y_test, predictions, average='macro') *100
    print("F1: %.2f%%" % (f1))

def evaluate_synthetic_data(data_orig, data_synth, cont_cols, cat_cols, metrics):
    """
    Takes an original dataset and a synthetic dataset generated from
    an algorithm that is trained on the former.
    Continuous values must be normalized.
    Categorical values must be mapped to numerical features.
    
    Returns 3 statistical metrics including:
    - average 1-TVD over all one-way marginals
    - average 1-TVD oder all two-way marginals
    - correlation accuracy (fraction of pairs of same correlation level)
    
    Returns the classification accuracy (F1 score using macro average) 
    of an XGBoost classifier trained on the synthetic data, used to
    make predictions on the original data.
    """

    # Statistical metrics
    if metrics[0]:
        TVD_ind(data_synth, data_orig, cont_cols, cat_cols)
    if metrics[1]:
        TVD_pair(data_synth, data_orig, cont_cols, cat_cols)
    if metrics[2]:
        cramer_corr_heatmap_cat(data_orig, data_synth, cat_cols)
    
    # Classification accuracy
    if metrics[3]:
        XGBoost_classifier_F1(data_orig, data_synth)