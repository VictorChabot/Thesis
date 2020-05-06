#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This programs contains few function I use for the outputs and the performance metrics
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

'''
Calculate the AUC, accuracy amd 
'''

def score_metrics(Y, X, fitted_model, name):

    Y_hat_prob = pd.DataFrame(fitted_model.predict(X), columns=['Y_hat_prob'])
    Y = pd.DataFrame(data=Y)
    Y.columns = ['Y']
    
    score = pd.concat([Y, Y_hat_prob], axis=1)
    
    score['Y_hat'] = (score['Y_hat_prob'] >= 0.5).astype('int').values
    
    pred_Y = score['Y_hat'].values.reshape(len(Y))
    real_Y = Y.values.reshape(len(Y))

    MSE  = np.sum(np.square(real_Y - pred_Y ))/len(Y)
    RMSE = np.sqrt(MSE)   
    
    score = score.drop('Y_hat_prob', axis=1)
    
    score['Accuracy'] = score['Y'] == score['Y_hat']
    score['AUC'] = 0
    score['PT_AT'] = (score['Y_hat'] == 1) & (score['Y'] == 1)
    score['PT_AF'] = (score['Y_hat'] == 1) & (score['Y'] == 0)
    score['PF_AT'] = (score['Y_hat'] == 0) & (score['Y'] == 1)
    score['PF_AF'] = (score['Y_hat'] == 0) & (score['Y'] == 0)
    
    score = score.drop(['Y', 'Y_hat'], axis=1)
    
    performance = score.mean()
    performance['AUC'] = roc_auc_score(Y, Y_hat_prob)
    performance['RMSE'] = RMSE
    
    performance = gen_metrics(performance)
    
    performance.index.name = name
    
    return performance

'''
Calculate true positive rate, true negative rate, positive predictive value 
and F score
input: df with good column names
output: df 
'''
def gen_metrics(df):
    
    df['TPR'] =  df['PT_AT']/(df['PT_AT'] + df['PF_AT'])
    
    df['TNR'] = df['PF_AF']/(df['PF_AF'] + df['PT_AF'])
    
    df['PPV'] = df['PT_AT']/(df['PT_AT'] + df['PT_AF'])
    
    
    df['F-Score'] = 2*(df['PPV']*df['TPR'])/(df['PPV'] + df['TPR'])
    
    df = df[['Accuracy', 'AUC', 'RMSE']]
    
    df.name = "Performance Metrics"
    
    return df