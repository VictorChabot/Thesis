#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program contains all the regressions and table from section 
3.1 of the thesis. 

There are four sections to this program:
    1) OLS regression before and after policy change
    2) Two stage least square model
    3) Poisson and negative binomial regression
    4) Latex outputs

author: Victor Chabot
"""
import os
os.chdir('/home/victor/gdrive/thesis_victor/codes/')

from metric_modules import score_metrics

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
import sklearn.model_selection
import matplotlib.pyplot as plt
import linearmodels as lm

from linearmodels.iv import IV2SLS

pd.options.display.float_format = '{:,.3f}'.format
df = pd.read_pickle('/home/victor/gdrive/thesis_victor/codes/4_prediction/final_df.pkl')
os.chdir('/home/victor/gdrive/thesis_victor/codes/4_prediction')

####################################### VARIABLE LISTS FOR EVERY REGRESSION

location_vars = [col for col in df.columns if (col[:8]=='location')]
cat_vars = [col for col in df.columns if (col[:8]=='category')]
main_cat_vars = [col for col in df.columns if (col[:8]=='main_cat')]
time_vars = [col for col in df.columns if (col[:14]=='month_creation') ]


numeric_var_stage_3  = ['intercept', 'ln_goal', 'duration_pred', 'duration_pred_2', 'nth_project', 'period', 'period_2', 'ratio_nth_city_country', 'lfd_new_projects']

numeric_vars_pol_test_2 = ['intercept', 'ln_goal', 'choc', 'duration', 'nth_project', 'period', 'period_2', 'ratio_nth_city_country', 'lfd_new_projects']

numeric_data_description = ['intercept', 'ln_goal', 'duration', 'nth_project', 'period', 'period_2', 'ratio_nth_city_country', 'lfd_new_projects']

IV_numeric_val = ['intercept', 'ln_goal', 'nth_project', 'period', 'period_2', 'ratio_nth_city_country', 'lfd_new_projects']


####################################### FUNCTIONS
"""
This function compute the RMSE.
input: np.array, Y is the actual target variable, Y_hat are the predicted values
"""
def compute_RMSE(Y, Y_hat):    
    MSE  = np.sum(np.square(Y - Y_hat ))/len(Y)
    RMSE = np.sqrt(MSE)
    
    return RMSE

"""
Run an OLS regression
"""
def run_OLS(Y, X):
    
    OLS = sm.OLS(Y, X)
    OLS_results = OLS.fit()
    
    return OLS_results

"""
Compute a t-stat.
input: two results object from OLS.fit() and the variable name
"""
def compute_t_stat(res_bp, res_ap, variable):
    
    beta_bp = res_bp.params[variable]
    se_bp = res_bp.bse[variable]
    
    beta_ap = res_ap.params[variable]
    se_ap = res_ap.bse[variable]
    
    t_stat = (beta_bp - beta_ap)/(np.sqrt(se_bp**2 + se_ap**2))
    
    print(variable + " tstat: " + str(t_stat))
    
    
    return t_stat

##############################################################################
##########################  SECTION 1: OLS REGRESSIONS before and after POL CHANGE
##########################  SECTION 3.1.1 OF THE THESIS
##############################################################################

# Define independant variables
numeric_vars_pol_test_1 = ['intercept', 'ln_goal', 'duration', 'nth_project', 'period', 'period_2', 'ratio_nth_city_country', 'lfd_new_projects']
vars_pol_test_1 = numeric_vars_pol_test_1 + main_cat_vars + time_vars + location_vars

# Define the sample before policy (bp)
X_bp = df.loc[df['choc']==0, vars_pol_test_1]
Y_bp = df.loc[df['choc']==0, 'success']

# Define the sample after policy (ap)

X_ap = df.loc[df['choc']==1, vars_pol_test_1]
Y_ap = df.loc[df['choc']==1, 'success']

# Run regressions
res_bp = run_OLS(Y_bp, X_bp)
res_ap = run_OLS(Y_ap, X_ap)

# Compute t-stat to know if coefficients are differents
t_stat_1 = compute_t_stat(res_bp, res_ap, 'duration')

# Score the observations with the OLS
prob_bp = res_ap.predict(X_bp)
Y_pred_bp  = np.where(prob_bp>=0.5, 1,0)

Y_pred_bp = Y_pred_bp.reshape(len(Y_pred_bp))
Y_real_bp = Y_bp.values.reshape(len(Y_pred_bp))

prob_ap = res_ap.predict(X_ap)
Y_pred_ap  = np.where(prob_ap>=0.5, 1,0)

Y_pred_ap = Y_pred_ap.reshape(len(Y_pred_ap))
Y_real_ap = Y_ap.values.reshape(len(Y_pred_ap))

# Compute RMSE for both regressions
RMSE_bp  = compute_RMSE(Y_real_bp, Y_pred_bp)
RMSE_ap  = compute_RMSE(Y_real_ap, Y_pred_ap)


##############################################################################
##########################  SECTION 2: FIRST STAGE and SECOND STAGE
##########################  SECTION: 3.1.3 to 3.3.6 IN THE THESIS
##############################################################################

#################################  with packages linear models (for corrected standard error on coeffs)
# Define VARIABLES FOR 2OLS
dep = ['success']
endog = ['duration']
instr = ['choc']
exog= IV_numeric_val  + main_cat_vars + time_vars + location_vars
data = df[dep + endog + instr + exog]

# First stage
res = IV2SLS(data[endog], data[instr+exog], None, None).fit()
print(res)

# SECOND STAGE
res_2sls = IV2SLS(data[dep], data[exog], data[endog], data[instr]).fit()
print(res_2sls)


#################################  with simple statsmodels
# First stage
numeric_var_stage_1  = ['intercept', 'ln_goal', 'choc', 'nth_project', 'period', 'period_2', 'ratio_nth_city_country', 'lfd_new_projects']
# Second stage with linear duration
numeric_var_stage_2  = ['intercept', 'ln_goal', 'duration_pred', 'nth_project', 'period', 'period_2', 'ratio_nth_city_country', 'lfd_new_projects']
# Second stage with quadratic duration
numeric_var_stage_3  = ['intercept', 'ln_goal', 'duration_pred', 'duration_pred_2', 'nth_project', 'period', 'period_2', 'ratio_nth_city_country', 'lfd_new_projects']

#Define total var list
vars_stage_1 = numeric_var_stage_1 + time_vars + main_cat_vars + location_vars
vars_stage_2 = numeric_var_stage_2 + time_vars + main_cat_vars + location_vars
vars_stage_3 = numeric_var_stage_3 + main_cat_vars + time_vars + location_vars

# Define VARIABLES FOR 2OLS
IV = df['choc']

#Define input and target for first stage
X_1 = df[vars_stage_1]
Z_1 = df[['duration']]

y = df['success']

# FIRST STAGE
first_stage = sm.OLS(Z_1, X_1)
results_1 = first_stage.fit()
results_1.summary2()

# Store predicted value
Z_pred = results_1.predict(X_1)

df['duration_pred'] = Z_pred
df['duration_pred_2'] = Z_pred**2

X_2 = df[vars_stage_2]

######################## Compute RMSE

Z_1_arr = np.array(Z_1).reshape(len(Z_1))
Z_pred_arr = Z_pred.values.reshape(len(Z_pred))
RMSE_first_stage = compute_RMSE(Z_1_arr, Z_pred_arr)

##################################### SECOND STAGE WITH LINEAR DURATION
##################################### section 3.1.5 in thesis

OLS_second_stage = sm.OLS(y, X_2)
OLS_results_2 = OLS_second_stage.fit()
OLS_results_2.summary2()

duration_pred_coeff = OLS_results_2.params['duration_pred']

choc_coeff = results_1.params['choc']
IV_estimator = duration_pred_coeff / choc_coeff

# F-test
A = np.identity(len(results_1.params))
A = A[1,:]
test_F_IV = results_1.f_test(A)

##################################### SECOND STAGE WITH QUADRATIC DURATION
##################################### section 3.1.6 in thesis
## With only one duration variable 
X_3 = df[vars_stage_3]

OLS_second_stage_3 = sm.OLS(y, X_3)
OLS_results_3 = OLS_second_stage_3.fit()

choc_coeff = results_1.params['choc']
IV_estimator = duration_pred_coeff / choc_coeff

# F-test
A = np.identity(len(results_1.params))
A = A[1,:]
test_F_IV = results_1.f_test(A)

# Compute performance second stage
perf_second_stage = score_metrics(Y=y, X=X_3, fitted_model=OLS_results_3, name='second_stage_perf')

OLS_results_2.summary2()

############################### FIND OPTIMAL DURATION OF CAMPAIGNS

df_duration = pd.DataFrame(data=np.arange(1,91,1), columns=['duration'])
# Extract coefficents to plot optimal duration
duration_pred_coeff = OLS_results_3.params['duration_pred']
duration_pred_coeff_2 = OLS_results_3.params['duration_pred_2']
df_duration['duration_2'] = df_duration['duration']**2

df_duration['eff_duration'] = df_duration['duration']*duration_pred_coeff + df_duration['duration_2']*duration_pred_coeff_2

plt.plot(df_duration['duration'], df_duration['eff_duration'])
plt.xlabel('duration in days')
plt.ylabel('dy/dx')
plt.axvline(x=33)

plt.savefig('success_over_duration.png')

##############################################################################
##########################  SECTION 3: POISSON AND NEGATIVE BINOMIAL FOR DURATION
##########################  SECTION: 3.1.4 IN THE THESIS
##############################################################################

# Compute poisson regression
poisson = sm.Poisson(Z_1, X_1)
poisson = poisson.fit()
poisson.summary()

# Compute marginal effects
poisson_1_dydx = poisson.get_margeff(method='dydx', at='median')
poisson_1_dydx.summary()

# Compute negative binomial
negative_binomial = sm.NegativeBinomial(Z_1, X_1)
negative_binomial = negative_binomial.fit(method="newton", max_iter=100)
nbinomial_1_dydx = negative_binomial.get_margeff(method='dydx', at='median')

nbinomial_1_dydx.summary()

# Compute RMSE for negative binomial
pred_binom = negative_binomial.predict(X_1)
pred_binom = np.array(pred_binom).reshape(len(X_1))
RMSE_neg_binom = compute_RMSE(Z_1_arr, pred_binom)

##############################################################################
##########################  SECTION 4: LATEX OUTPUT
##############################################################################

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


print(perf_second_stage.to_latex())


##########################  ANNEXES

print(results_1.summary2().as_latex())
print(OLS_results_2.summary2().as_latex())
print(OLS_results_3.summary2().as_latex())
print(perf_second_stage.to_latex(caption='Performance of the Second Stage Regression'))

############ 

res_bp_s= select_n_coeffs(res_bp,8)
res_ap_s= select_n_coeffs(res_ap,8)

print(res_bp_s.as_latex())
print(res_ap_s.as_latex())

