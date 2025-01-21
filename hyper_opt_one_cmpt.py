
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import plot_subject_levels
from scipy.integrate import solve_ivp
from tqdm import tqdm
from utils import one_compartment_model, objective_function
from sklearn.model_selection import GroupKFold
from utils import OneCompartmentModel, ObjectiveFunctionColumn
import numpy as np
from scipy.optimize import minimize
from joblib import dump, load
import os
from functools import partial
from utils import optimize_with_checkpoint_joblib
from sklearn.preprocessing import RobustScaler
import warnings
from tqdm import tqdm
from utils import huber_loss


df = pd.read_csv(r'/workspaces/miniconda/PKdata/data-raw/KI20160914/KI20160914.csv')


plot_subject_levels(df)





#prepare day 1 data
opt_df = df.dropna(subset = 'DV').copy()
opt_df['DV'] = opt_df['DV'].astype(pd.Float32Dtype())
opt_df = opt_df.loc[opt_df['DAY'] == 1, :]

#Within day 1 data, per subject identify the max concentration
#Drop time points occuring before the max, and set the time at that max conc to t=0
dfs = []
for c in opt_df['SUBJID'].drop_duplicates():
    work_df = opt_df.loc[opt_df['SUBJID'] == c, :].reset_index(drop = True)
    max_idx = work_df.loc[work_df['DV'] == work_df['DV'].max(), :].index[0]
    work_df = work_df.iloc[max_idx:, :]
    work_df['TIME'] = work_df['TIME'] - work_df['TIME'].min()
    dfs.append(work_df.copy())
work_df = pd.concat(dfs)


#plot the prepared data
plot_subject_levels(work_df)






scale_df = work_df.copy()
#scale_df[['DV']] = RobustScaler().fit_transform(scale_df[['DV']])
mgkg_scaler = RobustScaler()
age_scaler = RobustScaler()
wt_scaler = RobustScaler()

scale_df['MGKG'] = (scale_df['DOSR'] / scale_df['WT'])
scale_df['WT_scale'] = wt_scaler.fit_transform(scale_df[['WT']])
scale_df['MGKG_scale'] = mgkg_scaler.fit_transform(scale_df[['MGKG']])
scale_df['AGE_scale'] = age_scaler.fit_transform(scale_df[['AGE']])
scale_df['DOSR'] = scale_df['DOSR'] / 100



huber_deltas = np.linspace(3.25,10, 5)
cv_df = scale_df.copy().reset_index(drop = True)
X = cv_df.drop(columns = 'DV').copy()
y = cv_df['DV'].copy()
groups = cv_df['SUBJID'].copy()
allo_scaler = wt_scaler.transform([[70]])[0][0]





g_kfold = GroupKFold(n_splits=5)
g_kfold.get_n_splits(X, y, groups)




run_test_mod = True
if run_test_mod:
    warnings.filterwarnings("error", category= RuntimeWarning)

res = []
for idx, d in enumerate(tqdm(huber_deltas)):
    cv_error = []
    train_idx_history = []
    test_idx_history = [] 
    for i, (train_index, test_index) in enumerate(g_kfold.split(X, y, groups)):
        try:
            train_X = cv_df.iloc[train_index, :].copy()
            test_X = cv_df.iloc[test_index, :].copy()  

            mod_huber = OneCompartmentModel(dep_vars= {'k':[ ObjectiveFunctionColumn('AGE_scale'),
                                                        ObjectiveFunctionColumn('SEX')],
                                                'vd':[ObjectiveFunctionColumn('WT_scale',
                                                                                model_method='allometric', 
                                                                                allometric_norm_value=allo_scaler
                                                                                )]}, 
                                loss_function=huber_loss,
                                loss_params={'delta':d}
                                )
            mod_huber = mod_huber.fit(train_X, parallel=True,parallel_n_jobs=4 ,checkpoint_filename=f'huber_opt_d{d}_cv{i}.jb')
            test_preds = mod_huber.predict(test_X, parallel=True,parallel_n_jobs=4 )
            inner_mad = np.mean(np.abs(test_preds - test_X['DV'].values))
            cv_error.append(inner_mad)
        except RuntimeWarning as e:
            print(f"d = {d}:\n{e}\n")
            cv_error.append(None)
            raise e
        train_idx_history.append(train_index)
        test_idx_history.append(test_index)
    res.append(
        {
            'd':d, 
            'train_idx':train_idx_history, 
            'test_idx':test_idx_history,
            'cv_error':cv_error,
            'avg_cv_error':np.mean(cv_error) if all((i is not None for i in cv_error)) else None
        }
    )
    
    res_df = pd.DataFrame(res)
    with open('cv_log.jb', 'wb') as f:
        jb.dump(res_df, f)
res_df = pd.DataFrame(res)