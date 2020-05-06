#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program is for the first regressions and model
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

pd.options.display.float_format = '{:,.3f}'.format
df = pd.read_pickle('/home/victor/gdrive/thesis_victor/codes/4_prediction/final_df.pkl')

os.chdir('/home/victor/gdrive/thesis_victor/codes/4_prediction')

# Select columns
#time_vars = [col for col in df.columns if (col[:14]=='month_creation') | (col[:13]=='year_creation')]

time_vars = [col for col in df.columns if (col[:14]=='month_creation') ]


location_vars = [col for col in df.columns if (col[:8]=='location')]

cat_vars = [col for col in df.columns if (col[:8]=='category')]

main_cat_vars = [col for col in df.columns if (col[:8]=='main_cat')]

#df[time_vars + location_vars + cat_vars + main_cat_vars] = df[time_vars + location_vars + cat_vars + main_cat_vars].astype('bool').values

#df[['period', 'period_2', 'duration', 'duration_2', 'goal', 'goal_2', 'nth_project', 'intercept']] = df[['period', 'period_2', 'duration', 'duration_2', 'goal', 'goal_2', 'nth_project', 'intercept']].astype('int').values


# Select the list of variables for the first model of logistic regression

continous_var_list_model_1 = ['intercept', 'ln_goal', 'duration', 'choc', 'nth_project', 'period', 'ratio_nth_city_country', 'lfd_new_projects']
continous_var_list_model_3 = ['intercept', 'ln_goal', 'duration', 'choc_duration', 'nth_project', 'period', 'ratio_nth_city_country', 'lfd_new_projects']


var_list_model_1 =  continous_var_list_model_1  + main_cat_vars + location_vars + time_vars

var_list_model_3 =  continous_var_list_model_3 + main_cat_vars + location_vars + time_vars


# Select the variables to add to the second models
NLP_variables = ['word_count', 'frac_six_letters', 'ira', 'fog', 'flesh_score', 'sentiment']


var_list_model_2 = NLP_variables + var_list_model_1

df_mod_1 = df[var_list_model_1]

df_mod_2 = df[var_list_model_2]

df_mod_3 = df[var_list_model_3]

y = df.success.astype('int')

# Save to csv so R can read it
#y_df_mod_1 = pd.concat([y, df_mod_1], axis=1)
#y_df_mod_2 = pd.concat([y, df_mod_2], axis=1)
#
#y_df_mod_1.to_csv('df_mod_1_csv')
#y_df_mod_2.to_csv('df_mod_2_csv')

#head = df_mod_1.head()
##


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


#var_exog = df_mod_1.iloc[10000:12500]

model_1, results_1, train_X_1, valid_X_1, train_Y_1, valid_Y_1 = gen_logistic_model(y=y, df=df_mod_1, train_set=True)

model_2, results_2, train_X_2, valid_X_2, train_Y_2, valid_Y_2 = gen_logistic_model(y=y, df=df_mod_2, train_set=True)

model_3, results_3, train_X_3, valid_X_3, train_Y_3, valid_Y_3 = gen_logistic_model(y=y, df=df_mod_3, train_set=True)


Y=valid_Y_1
X=valid_X_1
fitted_model = results_1

df_y = pd.DataFrame(y)

performance_model_1 = score_metrics(valid_Y_1, valid_X_1, results_1, name='Logistic non-NLP')
performance_model_2 = score_metrics(valid_Y_2, valid_X_2, results_2, name='Logistic NLP')
performance_model_3 = score_metrics(valid_Y_3, valid_X_3, results_3, name='Logistic Policy Change')

summary_results_econ = pd.DataFrame([performance_model_1, performance_model_3, performance_model_2])

summary_results_econ.index =['Logistic non-NLP', 'Logistic Policy Change', 'Logistic NLP']

summary_results_econ.to_csv('1_perfo_econ_models.csv')

#%%

margeff_1_dydx = results_1.get_margeff(method='dydx', at='median')

margeff_1_eydx = results_1.get_margeff(method='eydx', at='median')

margeff_2_dydx = results_2.get_margeff(method='dydx', at='median')
margeff_2_eydx = results_2.get_margeff(method='eydx', at='median')

df_margeff_1 = margeff_1_dydx.summary_frame()

marg_loc = df_margeff_1.loc[location_vars,:]        

df_margeff_1_no_loc = df_margeff_1.drop(location_vars, axis=0)

df_margeff_1

marg_loc.loc[marg_loc['Pr(>|z|)'] >= 0.05, ['dy/dx']] = 0


df_mod_1[continous_var_list_model_1].describe().to_csv('var_1_described.csv')

df_margeff_1_no_loc.to_csv('1_margeff_1_dydx.csv')



margeff_1_eydx.summary_frame().to_csv('1_margeff_1_eydx.csv')

marg_loc.describe()['dy/dx'].to_csv('1_location_marginal_effect.csv', header=True)

################################# PART TWO, NLP IN ECONOMETRIC

margeff_2_dydx.summary_frame().to_csv('1_margeff_2_dydx.csv') 
margeff_2_eydx.summary_frame().to_csv('1_margeff_2_eydx.csv') 

eff_NLP = margeff_2_dydx.summary_frame().loc[NLP_variables]

# Describe NLP_variables top see if margeff have economic effects

def calc_marg_eff(df, df_marg, var_list, filename):
    
    NLP_described = df[var_list].describe().transpose()
    
    size_eff = NLP_described[['mean', '50%', 'std']].join(other=df_marg, how='left')
    
    size_eff['std_margeff'] = size_eff['std']*size_eff['dy/dx']
    
    size_eff = size_eff.drop(['Std. Err.', 'z', 'Conf. Int. Low', 'Cont. Int. Hi.'], axis=1)
    
    size_eff.to_csv(filename + '.csv')
    
    return size_eff

eff_mod_1 = calc_marg_eff(df=df_mod_1, df_marg=df_margeff_1 , var_list=continous_var_list_model_1, filename='1_var_model_1')

eff_mod_1 = eff_mod_1.iloc[1:,:]

eff_mod_2 = calc_marg_eff(df=df_mod_2, df_marg=eff_NLP , var_list=NLP_variables, filename='2_nlp_var_model_2')


################################  FOR LATEX OUTPUT


def select_n_coeffs(results, nb_first):

    summary = results.summary2()
    
#    meta = results.summary2().tables[0]
    
    coeffs = results.summary2().tables[1].iloc[:nb_first]
    
    summary.tables[1] = coeffs
    
    print(summary.as_latex())
    
    return summary


results_2.summary()


############################### MODEL 1 COEFFS AND MARGEFF
results_1_print = select_n_coeffs(results = results_1, nb_first=21)

print(eff_mod_1.to_latex(caption="Average Maginal Effect of on the Median Observation"))


############################### MODEL 2 COEFFS AND MARGEFF
results_2_print = select_n_coeffs(results = results_2, nb_first=13)

print(eff_mod_2.to_latex(caption="Average Maginal Effect of NLP variables on the Median Observation"))


results_3_print = select_n_coeffs(results = results_3, nb_first=8)

############################### PERFORMANCES MACHINE LEANING MODELS

print(summary_results_econ.to_latex(caption="Performance Metrics of the Logit Regressions"))

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




#
#all_margeff = margeff_1_dydx_all.margeff
#
#margeff_df = pd.DataFrame(data=all_margeff, columns=df_mod_1.columns[1:], index=df_mod_1.index)
#
#margeff_df = margeff_df.add_prefix('m_')
#
#var= 'goal'
#m_var = 'm_' + var
#
#temp_df = pd.concat([df_mod_1[var], margeff_df[m_var]], axis=1)


###

#df = df
#model= results_1
#var = 'goal'
#var_2 = var + '_2'
#
#marge_frame = margeff.summary_frame()
#
#marg_eff_1 = marge_frame.loc[var, ['dy/dx']].values
#
#marg_eff_2 = marge_frame.loc[var_2, ['dy/dx']].values
#
#df_described = df[var].describe()
#
#start_lin_space = df_described['min']
#stop_lin_space = df_described['max']
#
#lin_space_1 = np.linspace(start_lin_space, stop_lin_space, 100)
#lin_space_2 = lin_space_1**2
#results = lin_space_2.copy()
#
#for i in range(len(lin_space)):
#    results[i] = marg_eff_1*lin_space[i] + marg_eff_2*lin_space_2[i]
#    
#plt.plot(lin_space_1,results)
    
    
    



