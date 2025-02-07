import pandas as pd
import numpy as np
from scipy.stats import linregress

def estimate_subject_slope_cv(df, time_col = 'TIME', conc_col = 'CONC_ln', id_col = 'ID'):
    times = df[time_col].unique()
    results = []
    for start_idx, starttime in enumerate(times):
        start_idx_slopes = []
        if (start_idx + 1) < len(times):
            for end_idx, endtime in enumerate(times[start_idx+1:]):
                f = (df[time_col] >= starttime) & (df[time_col] <= endtime)
                work_df = df.loc[f, :]
                x = work_df[time_col].values
                y = work_df[conc_col].values
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                
                n = len(work_df[time_col].unique())
                k = 1  # Number of predictors (time)
                if n > 2:
                    adj_r2 = 1 - (1 - r_value**2) * (n - 1) / (n - k - 1)
                else:
                    adj_r2 = None
                start_idx_slopes.append(slope)
                if len(start_idx_slopes) > 1:
                    slope_cv = np.std(start_idx_slopes) / np.mean(start_idx_slopes)
                else:
                    slope_cv = None
                results.append({
                    'ID':work_df[id_col].unique()[0],
                    'start_index': start_idx,
                    'end_index': end_idx,
                    'start_time': starttime,
                    'end_time': endtime,
                    'slope': slope,
                    'start_idx_slope_cv':slope_cv,
                    'intercept': intercept,
                    'r_value': r_value,
                    'adj_r2': adj_r2,
                    'n_points': n
                })
    return pd.DataFrame(results)