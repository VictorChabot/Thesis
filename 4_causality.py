#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 10:14:56 2020

@author: victor
"""


import os
os.chdir('/home/victor/gdrive/thesis_victor/codes/')

#from metric_modules import score_metrics

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
import sklearn.model_selection
import matplotlib.pyplot as plt

pd.options.display.float_format = '{:,.3f}'.format

import sklearn

df = pd.read_pickle('/home/victor/gdrive/thesis_victor/codes/4_prediction/final_df.pkl')

os.chdir('/home/victor/gdrive/thesis_victor/codes/4_prediction')

# Select columns
#time_vars = [col for col in df.columns if (col[:14]=='month_creation') | (col[:13]=='year_creation')]

time_vars = [col for col in df.columns if (col[:14]=='month_creation') ]

sub_cat_vars =  [col for col in df.columns if (col[:13]=='category_slug') ]


location_vars = [col for col in df.columns if (col[:8]=='location')]

cat_vars = [col for col in df.columns if (col[:8]=='category')]

main_cat_vars = [col for col in df.columns if (col[:8]=='main_cat')]

#df[time_vars + location_vars + cat_vars + main_cat_vars] = df[time_vars + location_vars + cat_vars + main_cat_vars].astype('bool').values

#df[['period', 'period_2', 'duration', 'duration_2', 'goal', 'goal_2', 'nth_project', 'intercept']] = df[['period', 'period_2', 'duration', 'duration_2', 'goal', 'goal_2', 'nth_project', 'intercept']].astype('int').values

# Select the list of variables for the first model of logistic regression

numeric_var_stage_1  = ['intercept', 'ln_goal', 'choc', 'nth_project', 'period', 'period_2', 'ratio_nth_city_country', 'lfd_new_projects']

numeric_var_stage_2  = ['intercept', 'ln_goal', 'duration_pred', 'nth_project', 'period', 'period_2', 'ratio_nth_city_country', 'lfd_new_projects']
numeric_var_stage_3  = ['intercept', 'ln_goal', 'duration_pred', 'duration_pred_2', 'nth_project', 'period', 'period_2', 'ratio_nth_city_country', 'lfd_new_projects']

numeric_vars_pol_test_1 = ['intercept', 'ln_goal', 'duration', 'nth_project', 'period', 'period_2', 'ratio_nth_city_country', 'lfd_new_projects']
numeric_vars_pol_test_2 = ['intercept', 'ln_goal', 'choc', 'duration', 'nth_project', 'period', 'period_2', 'ratio_nth_city_country', 'lfd_new_projects']

numeric_data_description = ['intercept', 'ln_goal', 'duration', 'nth_project', 'period', 'period_2', 'ratio_nth_city_country', 'lfd_new_projects']


vars_stage_1 = numeric_var_stage_1 + time_vars + main_cat_vars + location_vars

vars_stage_2 = numeric_var_stage_2 + time_vars + main_cat_vars + location_vars

vars_stage_3 = numeric_var_stage_3 + main_cat_vars + time_vars + location_vars

vars_pol_test_1 = numeric_vars_pol_test_1 + main_cat_vars + time_vars + location_vars
vars_pol_test_11 = numeric_vars_pol_test_1 + main_cat_vars +  location_vars
vars_pol_test_2 = numeric_vars_pol_test_2 + main_cat_vars + time_vars + location_vars

def compute_RMSE(Y, Y_hat):    
    MSE  = np.sum(np.square(Y - Y_hat ))/len(Y)
    RMSE = np.sqrt(MSE)
    
    return RMSE

X_1 = df[vars_stage_1]
Z_1 = df[['duration']]

y = df['success']


# Two stage OLS

# First stage with OLD
first_stage = sm.OLS(Z_1, X_1)

results_1 = first_stage.fit()

Z_pred = results_1.predict(X_1)


######################## Compute RMSE

Z_1_arr = np.array(Z_1).reshape(len(Z_1))
Z_pred_arr = Z_pred.values.reshape(len(Z_pred))
    
def compute_RMSE(Y, Y_hat):    
    MSE  = np.sum(np.square(Y - Y_hat ))/len(Y)
    RMSE = np.sqrt(MSE)
    
    return RMSE

RMSE_first_stage = compute_RMSE(Z_1_arr, Z_pred_arr)




df['duration_pred'] = Z_pred
df['duration_pred_2'] = Z_pred**2
df['duration_pred_3'] = Z_pred**3

df['duration_resid'] = results_1.resid

X_2 = df[vars_stage_2]
################################################# Duration on success before and after policy change

#%%
def run_OLS(Y, X):
    
    OLS = sm.OLS(Y, X)
    
    OLS_results = OLS.fit()
    
    return OLS_results


def compute_t_stat(res_bp, res_ap, variable):
    
    beta_bp = res_bp.params[variable]
    se_bp = res_bp.bse[variable]
    
    beta_ap = res_ap.params[variable]
    se_ap = res_ap.bse[variable]
    
    t_stat = (beta_bp - beta_ap)/(np.sqrt(se_bp**2 + se_ap**2))
    
    print(variable + " tstat: " + str(t_stat))
    
    
    return t_stat


X_bp = df.loc[df['choc']==0, vars_pol_test_1]
Y_bp = df.loc[df['choc']==0, 'success']

X_ap = df.loc[df['choc']==1, vars_pol_test_1]
Y_ap = df.loc[df['choc']==1, 'success']

res_bp = run_OLS(Y_bp, X_bp)

res_ap = run_OLS(Y_ap, X_ap)

t_stat_1 = compute_t_stat(res_bp, res_ap, 'duration')

prob_bp = res_ap.predict(X_bp)
Y_pred_bp  = np.where(prob_bp>=0.5, 1,0)

Y_pred_bp = Y_pred_bp.reshape(len(Y_pred_bp))
Y_real_bp = Y_bp.values.reshape(len(Y_pred_bp))

prob_ap = res_ap.predict(X_ap)
Y_pred_ap  = np.where(prob_ap>=0.5, 1,0)

Y_pred_ap = Y_pred_ap.reshape(len(Y_pred_ap))
Y_real_ap = Y_ap.values.reshape(len(Y_pred_ap))

RMSE_bp  = compute_RMSE(Y_real_bp, Y_pred_bp)

RMSE_ap  = compute_RMSE(Y_real_ap, Y_pred_ap)

#%%
################################################# Duration on success with polici change

X_test_2 = df[vars_pol_test_2]

res_test_2 = run_OLS(Y=y, X=X_test_2)

res_test_2.summary()

##################################### Second stage with linear duration

OLS_second_stage = sm.OLS(y, X_2)

OLS_results_2 = OLS_second_stage.fit()

duration_pred_coeff = OLS_results_2.params['duration_pred']

choc_coeff = results_1.params['choc']

IV_estimator = duration_pred_coeff / choc_coeff


# F-test

A = np.identity(len(results_1.params))
A = A[1,:]

test_F_IV = results_1.f_test(A)

# Save results

def save_csv_summary(results, filename):
    f = open( filename + '.csv','w')
    f.write(results.summary().as_csv()) #Give your csv text here.
    ## Python will convert \n to os.linesep
    f.close()
    
    
save_csv_summary(results_1, '3_first_stage')

save_csv_summary(OLS_results_2, '3_OLS_results_2')


##################################### with quadratic duration

## With only one duration variable 
X_3 = df[vars_stage_3]

OLS_second_stage_3 = sm.OLS(y, X_3)

OLS_results_3 = OLS_second_stage_3.fit()

duration_pred_coeff = OLS_results_3.params['duration_pred']

duration_pred_coeff_2 = OLS_results_3.params['duration_pred_2']

choc_coeff = results_1.params['choc']

IV_estimator = duration_pred_coeff / choc_coeff

# F-test

A = np.identity(len(results_1.params))
A = A[1,:]

test_F_IV = results_1.f_test(A)

# Save results

perf_second_stage = score_metrics(Y=y, X=X_3, fitted_model=OLS_results_3, name='second_stage_perf')



perf_second_stage.to_pickle('3_perfo_causality.pkl')

save_csv_summary(OLS_results_3, '3_OLS_results_2')

results_1.summary2()

duration_pred_coeff = OLS_results_3.params['duration_pred']

duration_pred_coeff_2 = OLS_results_3.params['duration_pred_2']

OLS_results_2.summary2()

df_duration = pd.DataFrame(data=np.arange(1,91,1), columns=['duration'])

df_duration['duration_2'] = df_duration['duration']**2

df_duration['eff_duration'] = df_duration['duration']*duration_pred_coeff + df_duration['duration_2']*duration_pred_coeff_2

plt.plot(df_duration['duration'], df_duration['eff_duration'])
plt.xlabel('duration in days')
plt.ylabel('dy/dx')
plt.axvline(x=33)

plt.savefig('success_over_duration.png')


##################################### Negative binomial for duration

poisson = sm.Poisson(Z_1, X_1)

poisson = poisson.fit()

poisson.summary()

poisson_1_dydx = poisson.get_margeff(method='dydx', at='median')

poisson_1_dydx.summary()

negative_binomial = sm.NegativeBinomial(Z_1, X_1)

negative_binomial = negative_binomial.fit(method="newton", max_iter=100)

nbinomial_1_dydx = negative_binomial.get_margeff(method='dydx', at='median')

nbinomial_1_dydx.summary()
################################## LATEX OUTPUT

def select_n_coeffs(results, nb_first):

    summary = results.summary2()
    
#    meta = results.summary2().tables[0]
    
    coeffs = results.summary2().tables[1].iloc[:nb_first]
    
    summary.tables[1] = coeffs
    
    print(summary.as_latex())
    
    return summary


################### summary first stage

first_stage = select_n_coeffs(results_1, nb_first=8)

################### summary second stage

second_stage = select_n_coeffs(OLS_results_2, nb_first=8)

################### summary second stage with quadratic

second_stage = select_n_coeffs(OLS_results_3, nb_first=9)


OLS_results_2.summary().as_latex()

print(perf_second_stage.to_latex())


##########################  ANNEXES

print(results_1.summary2().as_latex())

print(OLS_results_2.summary2().as_latex())

print(OLS_results_3.summary2().as_latex())

print(perf_second_stage.to_latex(caption='Performance of the Second Stage Regression'))

############ 

res_bp_s= select_n_coeffs(res_bp,8)
res_ap_s= select_n_coeffs(res_ap,8)

res_bp_s= select_n_coeffs(res_X_bp2,8)
res_ap_s= select_n_coeffs(res_X_ap2,8)


print(res_bp_s.summary2().as_latex())
print(res_ap_s.summary2().as_latex())


#################### RMSE

Z_1_arr = np.array(Z_1).reshape(len(Z_1))
Z_pred_arr = Z_pred.values.reshape(len(Z_pred))


RMSE_first_stage = compute_RMSE(Z_1_arr, Z_pred_arr)

pred_binom = negative_binomial.predict(X_1)

pred_binom = np.array(pred_binom).reshape(len(X_1))

RMSE_neg_binom = compute_RMSE(Z_1_arr, pred_binom)

##
#X_3 = df[vars_stage_3]
#
#FCT_CTRL = sm.Logit(y, X_3)
#
#FCT_CTRL_results = FCT_CTRL.fit()


