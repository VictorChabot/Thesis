#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program computes and print the logistic models and marginal effects 
for section 3.2 in this thesis.

The two sections of this program: 
    1) Compute models
    2) Generate latex ouput
    
author: Victor Chabot

compare coefficients from two regressions: 
    https://stats.stackexchange.com/questions/93540/testing-equality-of-coefficients-from-two-different-regressions
    
    https://psycnet.apa.org/record/1995-27766-001
"""
 
import os
os.chdir('/home/victor/gdrive/thesis_victor/codes/')
from metric_modules import score_metrics
from metric_modules import gen_metrics
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
import sklearn.model_selection
import matplotlib.pyplot as plt


import sklearn

def compute_RMSE(Y, Y_hat):    
    MSE  = np.sum(np.square(Y - Y_hat ))/len(Y)
    RMSE = np.sqrt(MSE)
    
    return RMSE

pd.options.display.float_format = '{:,.3f}'.format
df = pd.read_pickle('/home/victor/gdrive/thesis_victor/codes/4_prediction/final_df_creator_id.pkl')

###########################################################################
######################## MAKE DF WITHOUT GOAL OUTLIER
###########################################################################
limit_goal = df['goal'].describe(percentiles=[0.05, 0.95])

per_5 = limit_goal['5%']
per_95 = limit_goal['95%']
    
df_windzor = df.loc[(df['goal']>per_5) & (df['goal'] < per_95), :]


###########################################################################
######################## MAKE DF WITHOUT nth_project VARIABLE
###########################################################################
df_fp = df.loc[(df['nth_project']==1), :]

creator_id_df = df['creator_id'].values
creator_id_df_windzor = df_windzor['creator_id'].values
creator_id_df_fp = df_fp['creator_id'].values

#%%

os.chdir('/home/victor/gdrive/thesis_victor/codes/4_prediction')

"""
function to separate training and test set and train the logistic regression

returns the result, fitted model, train and test set
"""
def gen_logistic_model(y, df, train_set=False):
    
    train_X = df.astype('float')
    train_Y = y
    
    if train_set==True:
        train_X, valid_X, train_Y, valid_Y = sklearn.model_selection.train_test_split(train_X.values, y.values,test_size=0.2,random_state=1)
        
        train_X = pd.DataFrame(train_X, columns=df.columns)
        valid_X = pd.DataFrame(valid_X, columns=df.columns)
        train_Y = pd.DataFrame(train_Y)
        valid_Y = pd.DataFrame(valid_Y)
    
    model = sm.Logit(train_Y, train_X)

    results = model.fit()
    
    if train_set==True:
        return model, results, train_X, valid_X, train_Y, valid_Y 

    else:
        return model, results



###################################################  VARIABLE FOR EVERY MODELS
time_vars = [col for col in df.columns if (col[:14]=='month_creation') ]
location_vars = [col for col in df.columns if (col[:8]=='location')]
cat_vars = [col for col in df.columns if (col[:8]=='category')]
main_cat_vars = [col for col in df.columns if (col[:8]=='main_cat')]

###################################################  VARIABLE LOGISTIC MODELS

##############################################################################
##########################  SECTION 1: LOGISTIC REGRESSIONS
##########################   section 3.2 in the thesis
##############################################################################

# Basic Logistic regression
# Thesis section 3.2.1
continous_var_list_model_1 = ['intercept', 'ln_goal', 'duration', 'nth_project', 'period', 'ratio_nth_city_country', 'lfd_new_projects']
var_list_model_1 =  continous_var_list_model_1  + main_cat_vars + location_vars + time_vars

# Basic Logistic regression without the nth-project variable

continous_var_list_model_1_fp = ['intercept', 'ln_goal', 'duration', 'period', 'ratio_nth_city_country', 'lfd_new_projects']
var_list_model_1_fp =  continous_var_list_model_1_fp  + main_cat_vars + location_vars + time_vars


# Logistic regression with interaction of policy change and duration variable
# Thesis section 3.2.2
continous_var_list_model_3 = ['intercept', 'ln_goal', 'duration', 'choc_duration', 'nth_project', 'period', 'ratio_nth_city_country', 'lfd_new_projects']
var_list_model_3 =  continous_var_list_model_3 + main_cat_vars + location_vars + time_vars

# Logistic regression with NLP variable
# Thesis section 3.2.3
NLP_variables = ['word_count', 'frac_six_letters', 'ira', 'fog', 'flesh_score', 'sentiment']
var_list_model_2 = NLP_variables + var_list_model_1

# Define the df with features for each model
df_mod_1 = df[var_list_model_1]
df_mod_1_fp = df_fp[var_list_model_1_fp]
df_mod_1_windzor = df_windzor[var_list_model_1]

df_mod_2 = df[var_list_model_2]
df_mod_3 = df[var_list_model_3]

# Define the vector of target variable
y = df.success.astype('int')
y_w = df_windzor.success.astype('int')
y_fp = df_fp.success.astype('int')

# Train the regressions
model_1, results_1, train_X_1, valid_X_1, train_Y_1, valid_Y_1 = gen_logistic_model(y=y, df=df_mod_1, train_set=True)
model_1_w, results_1_w, train_X_1_w, valid_X_1_w, train_Y_1_w, valid_Y_1_w = gen_logistic_model(y=y_w, df=df_mod_1_windzor, train_set=True)
model_1_fp, results_1_fp, train_X_1_fp, valid_X_1_fp, train_Y_1_fp, valid_Y_1_fp = gen_logistic_model(y=y_fp, df=df_mod_1_fp, train_set=True)


#%%

model_2, results_2, train_X_2, valid_X_2, train_Y_2, valid_Y_2 = gen_logistic_model(y=y, df=df_mod_2, train_set=True)
model_3, results_3, train_X_3, valid_X_3, train_Y_3, valid_Y_3 = gen_logistic_model(y=y, df=df_mod_3, train_set=True)


# Compute the performance of each model using the score metric modules from file metric_modules.py
performance_model_1 = score_metrics(valid_Y_1, valid_X_1, results_1, name='Logistic non-NLP')
performance_model_2 = score_metrics(valid_Y_2, valid_X_2, results_2, name='Logistic NLP')
performance_model_3 = score_metrics(valid_Y_3, valid_X_3, results_3, name='Logistic Policy Change')


# Merge all those metrics in one DF
summary_results_econ = pd.DataFrame([performance_model_1, performance_model_3, performance_model_2])
summary_results_econ.index =['Logistic non-NLP', 'Logistic Policy Change', 'Logistic NLP']

summary_results_econ.to_pickle('econometric_full_results.pkl')

################# COMPUTE MARGINAL EFFECTS OF LOGISTIC REGRESSIONS

margeff_1_dydx = results_1.get_margeff(method='dydx', at='median')
margeff_2_dydx = results_2.get_margeff(method='dydx', at='median')

pol_change_duration_variables = ['duration', 'choc_duration']
margeff_3_dydx = results_3.get_margeff(method='dydx', at='median')



# Select only variables of interest
df_margeff_1 = margeff_1_dydx.summary_frame()
marg_loc = df_margeff_1.loc[location_vars,:]        

df_margeff_1_no_loc = df_margeff_1.drop(location_vars, axis=0)

margeff_3_dydx = margeff_3_dydx.summary_frame()
margeff_3_dydx  = margeff_3_dydx.loc[pol_change_duration_variables]

# Replace non-significant variables by 0 coeff
marg_loc.loc[marg_loc['Pr(>|z|)'] >= 0.05, ['dy/dx']] = 0
marg_loc.describe()['dy/dx'].to_csv('1_location_marginal_effect.csv', header=True)

##### MARGINAL EFFECT OF NLP VARIABLES
margeff_2_dydx.summary_frame()
eff_NLP = margeff_2_dydx.summary_frame().loc[NLP_variables]

# Describe NLP_variables top see if margeff have economic effects

"""
this function computes the average marginal effect of one standard deviation
of the independant variables selected
"""
def calc_marg_eff(df, df_marg, var_list, filename):
    
    NLP_described = df[var_list].describe().transpose()
    
    size_eff = NLP_described[['50%', 'std']].join(other=df_marg, how='left')
    
    size_eff['std_margeff'] = size_eff['std']*size_eff['dy/dx']
    
    size_eff = size_eff.drop(['Std. Err.', 'z', 'Conf. Int. Low', 'Cont. Int. Hi.'], axis=1)
    
    size_eff.to_csv(filename + '.csv')
    
    return size_eff


# Compute the one standard deviation shock of variables
eff_mod_1 = calc_marg_eff(df=df_mod_1, df_marg=df_margeff_1 , var_list=continous_var_list_model_1, filename='1_var_model_1')
eff_mod_1 = eff_mod_1.iloc[1:,:]
eff_mod_2 = calc_marg_eff(df=df_mod_2, df_marg=eff_NLP , var_list=NLP_variables, filename='2_nlp_var_model_2')

eff_mod_3 = calc_marg_eff(df=df_mod_3, df_marg=margeff_3_dydx , var_list=pol_change_duration_variables, filename='3_pol_change_logit')

##############################################################################
##########################  SECTION 2: LATEX OUTPUT OF LOGISTIC REGRESSIONS
##########################   AND MACHINE LEARNING MODELS
##############################################################################

"""
This function selects n-coeffs from results object of fitted regression
The objective is to hide the variables that are not necessary for the analysis
"""
def select_n_coeffs(results, nb_first):

    # Gen summary object
    summary = results.summary2()
    
    # Extract the coefficients to keep
    coeffs = results.summary2().tables[1].iloc[:nb_first]
    
    # Assign the new object to the attribute
    summary.tables[1] = coeffs
    
    # Print
    print(summary.as_latex())
    
    return summary
   
############################### LOGISTIC MODEL 1 COEFFS AND MARGEFF
# Print regression
results_1_print = select_n_coeffs(results = results_1, nb_first=21)

print(results_1_w.summary().as_latex())

print(results_1_fp.summary().as_latex())
 
# Print  MARGEFF
print(eff_mod_1.to_latex(caption="Average Maginal Effect of on the Median Regressor"))


############################### LOGISTIC MODEL 2 COEFFS AND MARGEFF
results_2_print = select_n_coeffs(results = results_2, nb_first=13)
print(eff_mod_2.to_latex(caption="Average Maginal Effect of NLP variables on the Median Regressor"))

############################### LOGISTIC MODEL 2 COEFFS AND MARGEFF

results_3_print = select_n_coeffs(results = results_3, nb_first=8)

############################### PERFORMANCES MACHINE LEANING MODELS
print(summary_results_econ.to_latex(caption="Performance Metrics of the Logit Regressions"))

##############################################################################
##########################  SECTION 3: LATEX OUTPUT OF MACHINE LEARNING MODELS
##########################
##############################################################################

# 
total = pd.read_csv('total_results.csv', index_col=1)

total['RMSE'] = np.sqrt(1 - total['accuracy'])

total = total.drop(['Unnamed: 0', 'time'], axis=1)

total.columns = ['Accuracy', 'AUC', 
                 'PT_AT', 'PT_AF', 'PF_AT', 'PF_AF', 'RMSE']


total = gen_metrics(total)

total = total[['Accuracy', 'AUC', 'RMSE']]

indexes = [ind.split('_')[0] for ind in total.index]
feature_set = [ind.split('_')[1:] for ind in total.index]

total.index=indexes

total_index_1 = ['Base ' + idx for idx in total.index.unique()]
total_index_2 = ['NLP ' + idx for idx in total.index.unique()]
total_index_3 = ['NLP LDA ' + idx for idx in total.index.unique()]

new_idx_total = total_index_1 + total_index_2 + total_index_3

def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]


set_1 = total.iloc[:7,:]

set_2 = total.iloc[7:14,:]

set_3 = total.iloc[14:21,:]

total.index = new_idx_total

print(set_1.to_latex(caption="Performance Metrics: ML Models Without NLP Variables"))

print(set_2.to_latex(caption="Performance Metrics: ML Models with Simple NLP Variables"))

print(set_3.to_latex(caption="Performance Metrics: ML Models with Every NLP Variables"))

################################################### ENOCOMETRIC FULL PERF
pd.options.display.float_format = '{:,.3f}'.format
_2OLS = pd.read_pickle('3_perfo_causality.pkl')

t_logit_base = summary_results_econ.iloc[0,:]
t_logit_base.index = _2OLS.index

t_logit_nlp = summary_results_econ.iloc[1,:]
t_logit_nlp.index = _2OLS.index

list_row = [_2OLS, t_logit_base, t_logit_nlp]

full_econ_final_df = pd.DataFrame(data=list_row, index=['2SOLS', 'Logit Base', 'Logit NLP'])

full_econ_final_df['RMSE'] = np.sqrt(1 - full_econ_final_df['Accuracy'])

full_econ_final_df = full_econ_final_df[['Accuracy', 'AUC', 'RMSE']]

full_econ_final_df.name = 'Performance Metrics of Econometrics Models'

print(full_econ_final_df.to_latex(caption=full_econ_final_df.name))

################################################### FINAL ACCURACY AND AUC TABLE



summary = pd.read_csv('summary_results.csv', index_col=1)
summary = summary.drop('Unnamed: 0', axis=1)

summary.columns = ['Base Accuracy', 'Base AUC', 'NLP Accuracy', 'NLP AUC', 'NLP LDA Accuracy', 'NLP LDA AUC']

perfo_causality = pd.read_pickle('3_perfo_causality.pkl').iloc[:2]

perfo_causality.index = ['Base Accuracy', 'Base AUC']

perfo_logit_base = summary_results_econ.iloc[0,:2]

perfo_logit_base.index = ['Base Accuracy', 'Base AUC']

perfo_logit_nlp = summary_results_econ.iloc[1,:2]

perfo_logit_nlp.index = ['NLP Accuracy', 'NLP AUC']

perfo_logit_fin = pd.concat([perfo_logit_base, perfo_logit_nlp])

list_1= [perfo_causality, perfo_logit_fin]

summary_fin = pd.concat(list_1, axis=1).transpose()

summary_fin.index=['2SOLS', 'Logit']

summary_fin = pd.concat([summary_fin, summary])

summary_fin.name = "Performance Comparison of Models and Datasets"

print(summary_fin.to_latex(caption = summary_fin.name))


def change_pct(df, col):
    

    temp_index = list(df.index)
    temp_index.remove(col)

    new_df = df.copy()

    for row in temp_index:
        
        temp_row = (df.loc[col,:] - df.loc[row,:])
        
        new_df.loc[row,:] = temp_row.div(df.loc[row,:]).values
        
    return new_df
    


pd.options.display.float_format = '{:.1f}%'.format

total_fin = pd.concat([full_econ_final_df, total])

LGBM = change_pct(total_fin, 'NLP LDA LGBM').iloc[:3,:]*100

LGBM.name = "Percentage increase of the GBM compared to the econometric models"

extra_tree = change_pct(total_fin, 'NLP LDA extra trees entropy').iloc[:3,:]*100

extra_tree.name = "Percentage increase of the Extra-trees compared to the econometric models"



print(LGBM.to_latex(caption=LGBM.name))

print(extra_tree.to_latex(caption=extra_tree.name))
