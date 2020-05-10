# Thesis

This repository contains the files to run the models and analysis for my thesis. The number at the beginning of each file name indicates in which chapter of the thesis are the results of the program.


## Code descriptions

- metric_modules.py

It is the only file not related to a specific chapter. It contains modules used in different programs.

- 1_data_description.py

This file has the codes to generate the tables and graphs for the explanatory data analysis.


- 3_1_causality.py

This file has all the model related with the analysis of the causal relationship between duration of campaigns and success of projects.

- 3_2_logistic_models.py

This file has the logistic models that evaluate the marginal impact of the independent variables on the probability of success.

- 3_3_ML_models.py

This file has the machine learning models and the comparison of the performance metrics.


## Prerequisites

### Datasets
You can download the datasets necessary for this project [here](https://drive.google.com/open?id=1RKA0iTNlEOFgg9uWKZSMRyNvNx7pwoO-). 

### Libraries

- pandas
- numpy
- sklearn
- lightgbm
- scipy
- matplotlib
- statsmodels
- linearmodels


## Authors

* **Victor Chabot** - *Initial work* -

## Acknowledgments

Thanks Mario!
