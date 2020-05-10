#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All the results from section 3.3 are from that program.

The three sections of that program are: 
    1) Create the three feature sets for all the ML models
    2) Define the function to run the models and compute the performance metrics
    3) Organize and print the outputs for section 3.3
    
author: Victor Chabot
"""
import os
os.chdir('/home/victor/gdrive/thesis_victor/codes/')
from metric_modules import gen_metrics
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from scipy import sparse

import time
import pickle
import re

# Import the df
dfg = pd.read_pickle('/home/victor/gdrive/thesis_victor/codes/4_prediction/final_df_scaled.pkl')

# Select the appropriate df
os.chdir('/home/victor/gdrive/thesis_victor/codes/4_prediction')

##############################################################################
##########################  SECTION 1) GEN THE FEATURE SETS
##########################
##############################################################################

# Open the bow representation of the Blurbs
df_words = pd.read_pickle('/home/victor/gdrive/thesis_victor/codes/4_prediction/bow.pkl')

# Keep the words
bow_cols = list(df_words.columns)

# Extract NLP information
df_words_count = pd.read_pickle('/home/victor/gdrive/thesis_victor/codes/3_features/temp_df_word_count.pkl')
# Define appropriate col names
word_types_col = ['noun_count', 'verb_count', 'adj_count', 'adv_count', 'pron_count']

# Keep only appropriate variables
df_word_types = df_words_count[word_types_col]

# Select columns NLP related
topic_pattern = re.compile(r".+[_]{1}[topicB]{6}[_]{1}.+")
topic_cols = [col for col in dfg.columns if re.search(topic_pattern, col)]

nlp_cols_0 = ['word_count', 'char_count', 'avg_word', 'upper', 'frac_six_letters', 'sentiment', 'ira', 'fog', 'flesh_score']

nlp_cols = nlp_cols_0 + word_types_col
nlp_lda_cols = bow_cols + topic_cols

# Select target varialbe
y = dfg.success

# Keep only features
X = dfg.drop(['success'], axis=1)

# Add variables to the df
X_nlp_lda_0 = X.merge(df_words, how='left', right_index=True, left_index=True)

# Delete useless objects
del df_words, df_words_count, X, dfg

# 3rd feature set
X_nlp_lda = X_nlp_lda_0.merge(df_word_types, how='left', right_index=True, left_index=True)

# 2nd feature set
X_nlp = X_nlp_lda.drop(nlp_lda_cols, axis=1)

# 1st feature set
X_base = X_nlp.drop(nlp_cols, axis=1)

##############################################################################
##########################  SECTION 2) DEF FUNCTION TO RUN MODELS
##########################
##############################################################################

os.chdir('/home/victor/gdrive/thesis_victor/codes/4_prediction')

#Define models name for the models to run
models_name = ["logistic reg l1", "logistic reg l2", "tree crit gini", "tree crit entropy",
         "random forest gini",  "extra trees entropy",
         'LGBM']

# IMPORT APPROPRIATE MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# DEFINE THE PARAMETERS TO USE FOR XGBOOST MODEL
# THOSE PARAMETERS ARE FROM THA BAYESIAN OPTIMISATION 
best_param = {'boosting_type': 'gbdt', 'colsample_bytree': 0.648298266686153, 'is_unbalance': False, 'learning_rate': 0.0195145030407894, 'min_child_samples': 55, 'num_leaves': 115, 'reg_alpha': 0.6837057054471795, 'reg_lambda': 0.2764383951893788, 'subsample_for_bin': 220000, 'subsample': 0.8710428722692829, 'n_estimators': 1595}
# DEFINE XGBOOST MODEL
tuned_LGBMC = lgb.sklearn.LGBMClassifier(**best_param)

# DEFINE LIST OF MODELS
models = [
    LogisticRegression( penalty = 'l1', solver = 'saga'),
    LogisticRegression( penalty = 'l2', solver = 'sag'),
    DecisionTreeClassifier( criterion = 'gini'),
    DecisionTreeClassifier(criterion = 'entropy'),
    RandomForestClassifier(n_estimators= 100, criterion = 'gini'),
    ExtraTreesClassifier(n_estimators = 100, criterion = 'entropy'),
    tuned_LGBMC
    ]

 
"""
This function use a feature set and target as input and a iteration name 
1) it splits the sets
2) train the models
3) compute performance metric
4) store each model and performance metric
"""
def run_models(X, Y, iteration):

    # Extract the np.array from df
    X = sparse.csr_matrix(X.values)
    Y = np.ndarray.flatten(Y.values)

    # Split dataset for train and test sets
    train_X, valid_X, train_Y, valid_Y = sklearn.model_selection.train_test_split(X, Y,test_size=0.2,random_state=1)
    
    #Create dataframe to store values of results
    results = pd.DataFrame()
    
    # Create a columns with the models names
    results['model_name'] = models_name
    for i, element in enumerate(models_name):
        #specify the iteration for each modela
        results.at[i, 'model_name'] = element + '_' + str(iteration)

    # Create null columns in which to store the performance metrics
    results['accuracy'] = 0.0
    results['AUC'] = 0.0
    results['time'] = 0.0
    results['PredT_ActualT'] = 0.0
    results['PredT_ActualF'] = 0.0
    results['PredF_ActualT'] = 0.0
    results['PredF_ActualF'] = 0.0

    # Loop for each models    
    for i, model in enumerate(models):
        
        # time to know the computing time to train of each model
        start = time.time()
        # train a model
        fitted_model = model.fit(train_X, train_Y)
        end = time.time()
        
        # Create the predicted value of each observation of test set
        pred_Y = fitted_model.predict(valid_X)
        pred_prob_Y = fitted_model.predict_proba(valid_X)
        
        # Compute confusion matrix
        conf_m = confusion_matrix(valid_Y , pred_Y)
        # Assign each value of conf_matrix
        results.at[i, 'PredT_ActualT'] = conf_m[0,0]/len(valid_Y)
        results.at[i, 'PredT_ActualF'] = conf_m[0,1]/len(valid_Y)
        results.at[i, 'PredF_ActualT'] = conf_m[1,0]/len(valid_Y)
        results.at[i, 'PredF_ActualF'] = conf_m[1,1]/len(valid_Y)
        
        # Compute other perf metrics
        results.at[i, 'accuracy'] = fitted_model.score(valid_X, valid_Y)
        results.at[i, 'AUC'] = roc_auc_score(valid_Y, pred_prob_Y[:,1])
        # Compute total computation time
        results.at[i, 'time'] = end - start
        # Save model in a file
        pickle.dump(fitted_model, open( iteration + '_' + models_name[i] + '.joblib', 'wb'))
        print(results)
        del fitted_model
            
    del i, model, start, end
    # Save results
    results.to_csv('results_' + iteration + '.csv')
    return results



###################################  RUN THE MODELS FOR EACH FEATURE SET
results_nlp_lda = run_models(X = X_nlp_lda, Y = y, iteration = 'nlp_lda')
results_nlp = run_models(X = X_nlp, Y = y, iteration = 'nlp')
results_base = run_models(X = X_base, Y = y, iteration = 'base')

# Select the appropriate information
res = results_base.loc[:,['model_name',  'accuracy', 'AUC']]
res.columns = ['model_name', 'base_accuracy', 'AUC']
res['model_name'] = [name[:-5] for name in res.model_name]
res['nlp_accuracy'] = results_nlp['accuracy']
res['nlp_AUC'] = results_nlp['AUC']
res['nlp_lda_accuracy'] = results_nlp_lda['accuracy']
res['nlp_lda_AUC'] = results_nlp_lda['AUC']

res.to_csv('summary_results.csv')
# Agregate the information in one table
RES = pd.concat([results_base, results_nlp, results_nlp_lda])
# Save the information
RES.to_csv('total_results.csv')

##############################################################################
##########################  SECTION 3: OUTPUT MACHINE LEARNING MODELS
##########################  section 3.3 in thesis
##############################################################################

# Load info
total = pd.read_csv('total_results.csv', index_col=1)

# Compute RMSE
total['RMSE'] = np.sqrt(1 - total['accuracy'])

total = total.drop(['Unnamed: 0', 'time'], axis=1)

# Rename columns
total.columns = ['Accuracy', 'AUC', 
                 'PT_AT', 'PT_AF', 'PF_AT', 'PF_AF', 'RMSE']

# Compute other metrics
total = gen_metrics(total)

# Keep only necessary metrics
total = total[['Accuracy', 'AUC', 'RMSE']]

# Change column names for presentation
indexes = [ind.split('_')[0] for ind in total.index]
feature_set = [ind.split('_')[1:] for ind in total.index]
total.index=indexes
total_index_1 = ['Base ' + idx for idx in total.index.unique()]
total_index_2 = ['NLP ' + idx for idx in total.index.unique()]
total_index_3 = ['NLP LDA ' + idx for idx in total.index.unique()]

new_idx_total = total_index_1 + total_index_2 + total_index_3

##################################### SELECT SUBSET OF RESULTS FOR EACH SECTION

# Section 3.3.1 in thesis
set_1 = total.iloc[:7,:]

# Section 3.3.2 in thesis
set_2 = total.iloc[7:14,:]

# Section 3.3.3 in thesis
set_3 = total.iloc[14:21,:]

total.index = new_idx_total

# Print for latex
print(set_1.to_latex(caption="Performance Metrics: ML Models Without NLP Variables"))
print(set_2.to_latex(caption="Performance Metrics: ML Models with Simple NLP Variables"))
print(set_3.to_latex(caption="Performance Metrics: ML Models with Every NLP Variables"))

################################################### ENOCOMETRIC FULL PERF
# Change display option for pandas
pd.options.display.float_format = '{:,.3f}'.format

# Load econometrics and causality performance metrics to compare with ML performance
_2OLS = pd.read_pickle('3_perfo_causality.pkl')
summary_results_econ = pd.read_pickle('econometric_full_results.pkl')

# Select only important info
t_logit_base = summary_results_econ.iloc[0,:]
t_logit_base.index = _2OLS.index
t_logit_nlp = summary_results_econ.iloc[1,:]
t_logit_nlp.index = _2OLS.index

list_row = [_2OLS, t_logit_base, t_logit_nlp]

# Create a df with pertinent info
full_econ_final_df = pd.DataFrame(data=list_row, index=['2SOLS', 'Logit Base', 'Logit NLP'])

# Compute RMSE
full_econ_final_df['RMSE'] = np.sqrt(1 - full_econ_final_df['Accuracy'])
# Keep only wanted metrics
full_econ_final_df = full_econ_final_df[['Accuracy', 'AUC', 'RMSE']]
full_econ_final_df.name = 'Performance Metrics of Econometrics Models'

# Print to latex
print(full_econ_final_df.to_latex(caption=full_econ_final_df.name))

################################################## FINAL ACCURACY AND AUC TABLE
# Load info from ML models
summary = pd.read_csv('summary_results.csv', index_col=1)
summary = summary.drop('Unnamed: 0', axis=1)
# Change name columns
summary.columns = ['Base Accuracy', 'Base AUC', 'NLP Accuracy', 'NLP AUC', 'NLP LDA Accuracy', 'NLP LDA AUC']

# Load models performance
perfo_causality = pd.read_pickle('3_perfo_causality.pkl').iloc[:2]
perfo_causality.index = ['Base Accuracy', 'Base AUC']

perfo_logit_base = summary_results_econ.iloc[0,:2]
perfo_logit_base.index = ['Base Accuracy', 'Base AUC']

perfo_logit_nlp = summary_results_econ.iloc[1,:2]
perfo_logit_nlp.index = ['NLP Accuracy', 'NLP AUC']

# Concat in one table
perfo_logit_fin = pd.concat([perfo_logit_base, perfo_logit_nlp])

list_1 = [perfo_causality, perfo_logit_fin]
summary_fin = pd.concat(list_1, axis=1).transpose()
summary_fin.index=['2SOLS', 'Logit']
summary_fin = pd.concat([summary_fin, summary])
summary_fin.name = "Performance Comparison of Models and Datasets"

print(summary_fin.to_latex(caption = summary_fin.name))


############################################# % increase in performance
############################################# section 3.3.4 in thesis
"""
This function takes a df of performance metrics and compute 
the increase in % of a given model compare to all the other models
"""
def change_pct(df, col):
    
    # Create a list of models name
    temp_index = list(df.index)
    # Remove the model to compare to other models from the list
    temp_index.remove(col)
    
    #Create a new df
    new_df = df.copy()

    # For each model (except the one we compare to the other ones)
    for row in temp_index:
        
        # Compute the difference of each metric to col
        temp_row = (df.loc[col,:] - df.loc[row,:])
        
        # Divided by the metric (increase in %)
        new_df.loc[row,:] = temp_row.div(df.loc[row,:]).values
        
    return new_df
    

# Change display format
pd.options.display.float_format = '{:.1f}%'.format

# Create one df with ML and econometric models
total_fin = pd.concat([full_econ_final_df, total])

# Compute the % increase for LGBM model
LGBM = change_pct(total_fin, 'NLP LDA LGBM').iloc[:3,:]*100
LGBM.name = "Percentage increase of the GBM compared to the econometric models"

print(LGBM.to_latex(caption=LGBM.name))
