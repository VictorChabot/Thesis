"""
This program contains all the tables and graphs in chapter 2 of the thesis

The tree sections of this program
    1) Distribution of variables (section 1.1.1 of the thesis)
    2) Distribution of project durations (section 1.1.2 of the thesis)
    3) Serial creators (section 1.1.3 of the thesis)

author: Victor Chabot
"""
import pandas as pd
import numpy as np
import os
from scipy import stats
# Select display for pandas
pd.options.display.float_format = '{:,.3f}'.format

df = pd.read_pickle('/home/victor/gdrive/thesis_victor/codes/4_EDA/final_sample.pkl')
os.chdir('/home/victor/gdrive/thesis_victor/codes/4_EDA')

df_final = pd.read_pickle('/home/victor/gdrive/thesis_victor/codes/4_prediction/final_df.pkl')

df = df.loc[df_final.index, :]
df['success'] = df_final.success

df_0 = df.loc[df['choc']==0,:]

df_0['nb_over_60'] = df_0['duration']>=60

nb_over_60_days_before_pol_change = df_0['nb_over_60'].value_counts()

frac_over_60 = nb_over_60_days_before_pol_change[1]/(df_0['nb_over_60'].shape[0])

df_1 = df.loc[df['choc']==1,:]


def compute_tfidf(tf, df, N=10):
    
    w = tf*np.log(N/df)
    
    print(str(w))

compute_tfidf(tf=2, df=3, N=2)
compute_tfidf(tf=1, df=1, N=2)
compute_tfidf(tf=1, df=2, N=2)
compute_tfidf(tf=1, df=1, N=2)


"""
This function calculates the HHI, an index used for market concentration
"""
def calculate_hhi(serie):

    temp_array = np.array(serie)

    temp_array = np.square(temp_array)
    hhi = np.sum(temp_array)

    return hhi


"""
Saves a DF to latex and csv
"""
def save_df(tdf, name):

    tdf_num = tdf.select_dtypes(include='number')
    tdf_char = tdf.select_dtypes(exclude='number')

    for tcol in tdf_num.columns:
        tdf_num.loc[:,tcol] = tdf_num.loc[:,tcol].round(decimals=2)

    tdf = pd.concat([tdf_num, tdf_char], axis=1)

    tdf.to_csv(name + '.csv')

    with open(name + '.tex', 'w') as tf:
        tf.write(tdf.to_latex())


##############################################################################
##########################  SECTION 1: Description numeric variables
##########################  SECTION 1.1.1 THESIS
##############################################################################
        
col_char = [ 'word_count', 'char_count', 'avg_word' ]
df_char_described = df[col_char].describe()
df_char_described.name = "Numerical Description of the Blurb"

# Gen df for success descriptions
col_num = ['goal', 'success', 'frac_goal', 'backers_count', 'pledged', 'mean_pledged']
df_num = df[col_num]
described_num = df_num.describe()

# Change some columns to int for presentation
int_cols = ['goal', 'backers_count', 'pledged', 'mean_pledged']
described_num[int_cols]= described_num[int_cols].astype('int')

table_1_described_numeric = described_num
table_1_described_numeric.name = "Success Metrics of Campaigns"

# Count of categoric variables

df.main_cat.value_counts()
ym_start = df['ym_start'].value_counts().sort_index()
subcat_count = df.category_slug.value_counts()


"""
This function returns a df with freq and relative freq for a given column
It drop null rows
"""
def frequencies(tdf, tcol, drop_null_rows=True, save_dataframe=True):

    abs_freq = tdf[tcol].value_counts(sort=False)
    rel_freq = tdf[tcol].value_counts(sort=False, normalize=True)

    freqs = pd.concat([abs_freq, rel_freq], axis=1, )
    freqs.columns = ['Freq ' + tcol, '% ' + tcol]

    if drop_null_rows:
        freqs = freqs.loc[freqs['Freq ' + tcol] != 0, :]

    if save_dataframe:
        t_name = 'frequency_' + tcol

        save_df(tdf=freqs, name=t_name)

    return freqs


###################################### blurbs description
df.word_count.describe()
short_blurb=df.loc[df['word_count']<=40, 'word_count']

word_hist = short_blurb.hist(density=True, bins=20)
word_hist.set_xlabel('Nb Words per Blurb')
word_hist.set_ylabel('Relative Frequency')

word_hist.figure.savefig('1_histogram_blurb.png', dpi=300)

df_nlp = df[['fog', 'ira', 'flesh_score','sentiment']]
print(df_nlp.describe().to_latex())

# Compute frequence of countries
freq_countries = frequencies(tdf=df, tcol='location_country')
freq_countries.name = 'Frequence of Projects per Country'

# Compute frequence of locations
freq_location = frequencies(tdf=df, tcol='location')
freq_location = freq_location.sort_values('% location', ascending=False).iloc[:10,:]
freq_location.name = 'Most Frequent States and Countries'

# Compute frequence of years
freq_years = frequencies(tdf=df, tcol='y_start')
freq_years.name = 'Frequence of Projects per Year'

################################### Month start description
df['y_start_int'] = df['y_start'].astype('int')
# We keep only years for which we have every months, to keep each month comparable

df_2010_2018 = df.loc[ (df['y_start_int'] > 2009) & (df['y_start_int'] < 2019), :]

# Compute frequence of months
freq_months = frequencies(tdf=df_2010_2018 , tcol='m_start')
freq_months.name = 'Frequence of Projects per Month'


##############################################################################
##########################  SECTION 2: Description DURATION VARIABLE
##########################  SECTION 1.1.2 IN THESIS
##############################################################################

# For the description of the duration we only keep the year for which we have all the data
simple_duration_described = df_2010_2018['duration'].describe(percentiles=np.arange(0.1,1,0.1)).iloc[1:]
simple_duration_described.name = 'Distribution of the Duration of Projects'

# Compute the average duration of campaigns in days at each month
described_duration = df.groupby('ym_start')['duration'].describe()
described_duration = described_duration.drop(['count', 'mean', 'std'],axis=1)
described_duration = described_duration.iloc[:, -8:]

# Plot the average duration of projects for each month
duration_plot = described_duration.plot()

# Add a vertical line at policy change
duration_plot.axvline(x=27, label='Policy Change', ls='--')

# Save plot
duration_plot.figure.savefig('2_duration_distribution.png', dpi=300)

##############################################################################
##########################  SECTION 3: Description SERIAL CREATORS
##########################  SECTION 1.1.2 IN THESIS
##############################################################################

##################################### Compare variables per nb projects

# Extract the agregate information for each creator
df_creator = pd.read_pickle('df_creator.pkl')

# Variables to keep for the first table
df_creator_num = df_creator[['nb_projects', 'success', 'goal', 'frac_goal', 'mean_pledged', 'mean_months_per_project',
                             'mean_duration', 'cuts_nb_projects']]

described = df_creator_num.groupby('cuts_nb_projects').describe(percentiles=[0.5])

########################COMPUTE T-TEST TO COMPARE MEAN DURATION BY NB PROJECT


def gen_lst_of_arrays(df, binned_var, var):
    
    unique_binned_var = df[binned_var].unique()
    
    lst_arrays = list()
    
    for bin_value in unique_binned_var:
        
        # Select the appropriate value
        array_values = df.loc[df[binned_var]==bin_value, var].values
        
        # Append to list the array of values
        lst_arrays.append(array_values)
        
    return unique_binned_var, lst_arrays


idx_bins_lst, lst_arrays_duration = gen_lst_of_arrays(df=df_creator_num, binned_var='cuts_nb_projects', var='mean_duration')


# create df to store results

def gen_series_p_values(idx_bins, lst_arrays, increment):

    ttest_serie = pd.Series(index=idx_bins)
    
    ttest_serie.name = 'p-value ' + str(increment)
    
    increment = 1
    idx_bins = idx_bins_lst
    
    for i, bn in enumerate(idx_bins[:-increment]):
        
        bin_1 = lst_arrays[i]
        bin_2 = lst_arrays[i+increment]
        
    
        tstat_val = stats.ttest_ind(bin_1, bin_2, equal_var = False)
        
        ttest_serie.iat[i] = tstat_val[1]
        
    return ttest_serie
    
serie_p_values = gen_series_p_values(idx_bins=idx_bins_lst,
                                     lst_arrays=lst_arrays_duration,
                                     increment=1)
    
df_p_values = pd.DataFrame(serie_p_values.values, index=serie_p_values.index, columns=['t-test for mean difference'])

# create df to store results
# Keep only the mean of each information
super_creator_mean_described = described.loc[:,described.columns.get_level_values(1).isin({"mean"})]
# Keep only the median for each information
super_creator_med_described = described.loc[:,described.columns.get_level_values(1).isin({"50%"})]

# Concat the last two tables together
mean_med = pd.concat([super_creator_mean_described , super_creator_med_described], axis=1)

# Access the discretize value of nb of projects
count_nb_project = df_creator.cuts_nb_projects.value_counts()
# Sort creators by nb projects
count_nb_project.sort_index()

described_creator_level = pd.concat([count_nb_project, mean_med], axis=1)
described_creator_level.columns.values[0] = 'nb_projects'
described_creator_level.to_csv('3_creator_level.csv')

######################################### COMPARISON OF CATEGORIES OF PROJECTS

# Create a df of super creators
df_sc = df_creator.loc[df_creator['nb_projects'] >=5,:]

# Create a df of typical creators
df_mc = df_creator.loc[df_creator['nb_projects'] < 5,:]
df_mc = df_mc.sort_values('nb_projects', ascending=False)

# Create a df only with the numeric variables and describe
df_mc_numeric = df_mc.select_dtypes(include='number')
df_mc_numeric_described = df_mc_numeric.describe()

# Create df of only categoric variables
df_mc_cat = df_mc.select_dtypes(exclude='number')

# Compute market share of each main category for each sample
#  (normal creator vs super creators)
df_cat_count = df_mc.frequent_cat.value_counts(normalize=True)
sc_cat_count = df_sc.frequent_cat.value_counts(normalize=True)

# Create df with market share of both samples
count_main_cats = pd.concat([sc_cat_count, df_cat_count], axis=1)
count_main_cats.columns = ['Serial Creators', 'Typical Creators']
count_main_cats = count_main_cats.sort_values('Serial Creators', ascending=False)
count_main_cats.name = "Relative Frequency of Project Categories: Serial vs Typical Creators Comparison"

# Compute market share of each sub category for each sample
#  (normal creator vs super creators)
df_sub_cat_count = df_mc.frequent_sub_cat.value_counts(normalize=True)
sc_sub_cat_count = df_sc.frequent_sub_cat.value_counts(normalize=True)

# Create df with market share of both samples
count_sub_cats = pd.concat([sc_sub_cat_count, df_sub_cat_count], axis=1)
count_sub_cats.columns = ['Serial Creators', 'Typical Creators']

# Keep 15th most frequent sub cat for each sample
count_sub_cats_tot = count_sub_cats.sort_values('Typical Creators', ascending=False).iloc[:15, :]
count_sub_cats_sc = count_sub_cats.sort_values('Serial Creators', ascending=False).iloc[:15, :]

# Create a df with the 15th most frequent subcat of each sample
count_sub_cats = pd.concat([count_sub_cats_sc, count_sub_cats_tot], axis=0) # merge both table
count_sub_cats = count_sub_cats.sort_values('Serial Creators', ascending=False) # sort by serial creator
count_sub_cats = count_sub_cats.drop_duplicates() # since we merge both,some categories are top 15 for both sample
count_sub_cats.name = "Relative Frequency of Project Sub-categories: Serial vs Typical Creators Comparison"
save_df(count_sub_cats, '3_sub_cat_comparison')

##

# categories

cats = df[['main_cat', 'category_slug']]
cats_slug_unique = cats.category_slug.unique()
main_cats_unique = cats.main_cat.unique()
more_slug = [cat for cat in main_cats_unique if cat[:4] == 'more']

############################ Compute HHI
## Create empty DF
HHI_results = pd.DataFrame(index=['Nb Categories', 'Equal Shares',
                                  'Typical Creators', 'Serial Creators'],
                           columns=['sub categories', 'main categories'])

HHI_results.name = 'HHI per category'

# Assign Values if HHIs
HHI_results.loc['Typical Creators', 'sub categories'] = calculate_hhi(df_sub_cat_count).round(decimals=3)
HHI_results.loc['Serial Creators', 'sub categories'] = calculate_hhi(sc_sub_cat_count).round(decimals=3)

HHI_results.loc['Typical Creators', 'main categories'] = calculate_hhi(df_cat_count).round(decimals=3)
HHI_results.loc['Serial Creators', 'main categories'] = calculate_hhi(sc_cat_count).round(decimals=3)

HHI_results.iloc[0, :] = [len(cats_slug_unique), len(main_cats_unique)]

HHI_results.iloc[1, :] = 1/HHI_results.iloc[0, :]

HHI_results.columns = ['Sub Categories', 'Main Categories']

HHI_results.name = "Herfindahl-Hirschman Index for Categories and Sub-Categories: Comparison Typical vs Serial Creators"

############################################# Comparison of NLP super creator vs normal creators

# Select variables to compare
nlp_variables = ['creator_id', 'word_count', 'char_count', 'avg_word','frac_six_letters']

# Create df
df_nlp = df[nlp_variables]

df_indexes = df.index

# Select all projects that are from serial
super_creator_id_list = df_creator.loc[df_creator['nb_projects']>5, :].index

df_nlp_sc = df_nlp.loc[df['creator_id'].isin(super_creator_id_list), :]

df_nlp = df_nlp.drop(df_nlp_sc.index)

# Describe of both sample
df_nlp_described = df_nlp.describe()
df_nlp_sc_described = df_nlp_sc.describe()

nlp_comparison = pd.DataFrame(index=df_nlp_described.index)

# Create column names and df
for col in df_nlp_described.columns:
    total_name = 'TC ' + col
    super_creator_name = 'SC ' + col
    
    TC_col = df_nlp_described[col].copy()
    TC_col.name = total_name
    
    SC_col = df_nlp_sc_described[col].copy()
    SC_col.name = super_creator_name

    nlp_comparison = pd.concat([nlp_comparison, TC_col , SC_col], axis=1)


##############################################################################
##########################  SECTION 4: PRINTS TO LATEX
##########################   
##############################################################################

data_conf_mat= [['True Positive', 'False Negative'], ['False Positive', 'True Negative']]
conf_mat = pd.DataFrame(data_conf_mat, index=['Actual True', 'Actual False'], columns=['Predicted True', 'Predicted False'])
conf_mat.name = 'Confusion Matrix'

def print_to_latex(df, label=None):
    
    print(df.to_latex(caption=df.name, label=label))


# IN THE DESCRIPTION STATISTIC SECTIONS
################################### TABLE 2, DESCRIBED STATISTICS
# 1) Distribution of numeric variables

print(table_1_described_numeric.to_latex())

print(freq_years.to_latex(caption=freq_years.name))

print(freq_months.to_latex(caption=freq_months.name))

print(described_num.to_latex())

print(freq_countries.to_latex())

print(freq_location.to_latex())


print(simple_duration_described.to_latex())
# 2) Duration distribution the plot
    
# Inserted directly as image

# Creator level information

table_2 = described_creator_level[['nb_projects',('success', 'mean'), ('mean_duration', 'mean')]]

table_2.columns = ['nb projects', 'success rate', 'mean duration']

table_2['mean duration'] = table_2['mean duration'].map('{:,.1f}'.format)

table_2 = table_2.join(df_p_values)

table_2.name = "Average Success Rate and Duration of Campaigns in Function of Experience"


print(table_2.to_latex())

################################### TABLE 3, MORE DESCRIBED NUM PER NB PROJECTS

# Med per nb projects
table_3 = described_creator_level[[('goal', '50%'), ('frac_goal', '50%'), ('mean_pledged', '50%'), ('mean_months_per_project', '50%')]]
table_3.columns = ['goal', 'frac goal', 'mean pledge', 'months per projects' ]
table_3.loc[:, 'goal'] = table_3['goal'].values.astype('int')
table_3.name = "Median Goal, Fraction of Goal, Mean Pledge and Time Spent per Project in function of Experience"

print(table_3.to_latex())

################################### TABLE 4,5, categories of Serial Creators

print(count_main_cats.to_latex())
print(count_sub_cats.to_latex())
print(HHI_results.to_latex())
d19 = df.loc[df['year_creation'] == 2019,:]


################################### FINAL PRINTING ########################

print_to_latex(conf_mat)

print_to_latex(df_char_described)

print_to_latex(table_1_described_numeric)

print_to_latex(simple_duration_described)

print_to_latex(table_2)

print_to_latex(table_3)

# HHI and market shares
print_to_latex(count_main_cats)
print_to_latex(count_sub_cats)

print_to_latex(HHI_results)
