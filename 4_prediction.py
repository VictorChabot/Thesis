#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program is for the first regressions and model
"""
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from scipy import sparse
#import xgboost

import scipy
import time
import sys
import pickle # to save files
import collections # to sort dictionary
import re
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels
##
#For tfidf
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
dfg = pd.read_pickle('/home/victor/gdrive/thesis_victor/codes/4_prediction/final_df_scaled.pkl')
os.chdir('/home/victor/gdrive/thesis_victor/codes/4_prediction')


df_words = pd.read_pickle('/home/victor/gdrive/thesis_victor/codes/4_prediction/bow.pkl')

bow_cols = list(df_words.columns)

df_words_count = pd.read_pickle('/home/victor/gdrive/thesis_victor/codes/3_features/temp_df_word_count.pkl')

word_types_col = ['noun_count', 'verb_count', 'adj_count', 'adv_count', 'pron_count']

df_word_types = df_words_count[word_types_col]


topic_pattern = re.compile(r".+[_]{1}[topicB]{6}[_]{1}.+")

topic_cols = [col for col in dfg.columns if re.search(topic_pattern, col)]

nlp_cols_0 = ['word_count', 'char_count', 'avg_word', 'upper', 'frac_six_letters', 'sentiment', 'ira', 'fog', 'flesh_score']

nlp_cols = nlp_cols_0 + word_types_col

nlp_lda_cols = bow_cols + topic_cols

dfg.info(memory_usage='deep')

y = dfg.success

X = dfg.drop(['success'], axis=1)


X_nlp_lda_0 = X.merge(df_words, how='left', right_index=True, left_index=True)

del df_words, df_words_count, X, dfg

X_nlp_lda = X_nlp_lda_0.merge(df_word_types, how='left', right_index=True, left_index=True)

X_nlp = X_nlp_lda.drop(nlp_lda_cols, axis=1)

X_base = X_nlp.drop(nlp_cols, axis=1)

#
#X_nlp_lda = X_nlp_lda.iloc[:10000,:]
#
#X_nlp = X_nlp_lda.iloc[:10000,:]
#X_base = X_nlp_lda.iloc[:10000,:]
#y = y[:10000]

os.chdir('/home/victor/gdrive/thesis_victor/codes/4_prediction')
#first model with everything,
#models without lda
#model without any words

#Define models name for the models to run
models_name = [ "tree crit gini", "tree crit entropy",
         "random forest gini", "random forest entropy", "extra trees gini", "extra trees entropy",
         "Naive Bayes", 'LGBM']

models_name = ["logistic reg l1", "logistic reg l2", "tree crit gini", "tree crit entropy",
         "random forest gini",  "extra trees entropy",
         'LGBM']

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

#Define the parameters of each model to try
best_param = {'boosting_type': 'gbdt', 'colsample_bytree': 0.648298266686153, 'is_unbalance': False, 'learning_rate': 0.0195145030407894, 'min_child_samples': 55, 'num_leaves': 115, 'reg_alpha': 0.6837057054471795, 'reg_lambda': 0.2764383951893788, 'subsample_for_bin': 220000, 'subsample': 0.8710428722692829, 'n_estimators': 1595}

tuned_LGBMC = lgb.sklearn.LGBMClassifier(**best_param)

models = [
    #Logistic regressions
    LogisticRegression( penalty = 'l1', solver = 'saga'),
    LogisticRegression( penalty = 'l2', solver = 'sag'),
    #Trees
    DecisionTreeClassifier( criterion = 'gini'),
    DecisionTreeClassifier(criterion = 'entropy'),
    #Random forests
    RandomForestClassifier(n_estimators= 100, criterion = 'gini'),
#    RandomForestClassifier(n_estimators= 100, criterion = 'entropy'),
    #Extra Trees
    ExtraTreesClassifier(n_estimators = 100, criterion = 'entropy'),
#    ExtraTreesClassifier(n_estimators = 100, criterion = 'gini'),
    tuned_LGBMC
    ]

#Run all the models with a specific dataset, returns a dataframe with the results

#  X = x_nlp_lda
#  Y = y
 # iteration = 'nlp_lda'
 
 

def run_models(X, Y, iteration):

    X = sparse.csr_matrix(X.values)
    Y = np.ndarray.flatten(Y.values)
#    Y = sparse.csr_matrix(Y)

    train_X, valid_X, train_Y, valid_Y = sklearn.model_selection.train_test_split(X, Y,test_size=0.2,random_state=1)

    #train_X = train_X.toarray()
    # valid_X = valid_X.toarray()
    # train_Y = train_Y.toarray()
    # valid_Y = valid_Y.toarray()
    # #
    # train_Y = np.ndarray.flatten(train_Y)
    # valid_Y = np.ndarray.flatten(valid_Y)
    #
    # # label encode the target variable
    # encoder = sklearn.preprocessing.LabelEncoder()
    # train_Y = encoder.fit_transform(train_Y)
    # valid_Y = encoder.fit_transform(valid_Y)
    
    #Create dataframe to store values of results
    results = pd.DataFrame()
    
    results['model_name'] = models_name
    for i, element in enumerate(models_name):
        results.at[i, 'model_name'] = element + '_' + str(iteration)

    results['accuracy'] = 0.0
    results['AUC'] = 0.0
    results['time'] = 0.0
    results['PredT_ActualT'] = 0.0
    results['PredT_ActualF'] = 0.0
    results['PredF_ActualT'] = 0.0
    results['PredF_ActualF'] = 0.0

    #Create a vector of fitted model, that will be fitted in a loop
    fitted_models = models
    
    for i, model in enumerate(models):
        start = time.time()
        fitted_model = model.fit(train_X, train_Y)
        end = time.time()
        pred_Y = fitted_model.predict(valid_X)
        pred_prob_Y = fitted_model.predict_proba(valid_X)
        conf_m = confusion_matrix(valid_Y , pred_Y)
        results.at[i, 'PredT_ActualT'] = conf_m[0,0]/len(valid_Y)
        results.at[i, 'PredT_ActualF'] = conf_m[0,1]/len(valid_Y)
        results.at[i, 'PredF_ActualT'] = conf_m[1,0]/len(valid_Y)
        results.at[i, 'PredF_ActualF'] = conf_m[1,1]/len(valid_Y)
        results.at[i, 'accuracy'] = fitted_model.score(valid_X, valid_Y)
        results.at[i, 'AUC'] = roc_auc_score(valid_Y, pred_prob_Y[:,1])
        results.at[i, 'time'] = end - start
        pickle.dump(fitted_model, open( iteration + '_' + models_name[i] + '.joblib', 'wb'))
        print(results)
        del fitted_model
            
    del i, model, start, end
    results.to_csv('results_' + iteration + '.csv')
    return results


#Create Dataset for the model with every variables, run models


results_nlp_lda = run_models(X = X_nlp_lda, Y = y, iteration = 'nlp_lda')

results_nlp = run_models(X = X_nlp, Y = y, iteration = 'nlp')

results_base = run_models(X = X_base, Y = y, iteration = 'base')


res = results_base.loc[:,['model_name',  'accuracy', 'AUC']]
res.columns = ['model_name', 'base_accuracy', 'AUC']
res['model_name'] = [name[:-5] for name in res.model_name]
res['nlp_accuracy'] = results_nlp['accuracy']
res['nlp_AUC'] = results_nlp['AUC']
res['nlp_lda_accuracy'] = results_nlp_lda['accuracy']
res['nlp_lda_AUC'] = results_nlp_lda['AUC']

res.to_csv('summary_results.csv')

RES = pd.concat([results_base, results_nlp, results_nlp_lda])
RES.to_csv('total_results.csv')
