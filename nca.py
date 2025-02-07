import pandas as pd
import numpy as np
from scipy.stats import linregress
from utils import safe_signed_log
from sklearn.metrics import auc


def estimate_subject_slope_cv(df, time_col = 'TIME', conc_col = 'CONC_ln', id_col = 'ID'):
    df = df.copy()
    sub_max_conc = df[conc_col].max()
    max_time = df.loc[df[conc_col] == sub_max_conc, time_col].values[-1]
    df['orig_conc'] = df[conc_col].copy()
    df[conc_col] =safe_signed_log(df[conc_col])
    times = df[time_col].unique()
    results = []
    for start_idx, starttime in enumerate(times):
        if start_idx == 3:
            testing = 1
        start_idx_slopes = []
        if (start_idx + 1) < len(times):
            for end_idx, endtime in enumerate(times[start_idx+1:]):
                f = (df[time_col] >= starttime) & (df[time_col] <= endtime)
                work_df = df.loc[f, :]
                x = work_df[time_col].values
                y = work_df[conc_col].values
                auc_y = work_df['orig_conc'].values
                section_auc = auc(x, auc_y)
                auc_per_time = section_auc / (endtime - starttime)
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
                    #'start_index': start_idx,
                    #'end_index': end_idx,
                    'auc_per_time':auc_per_time,
                    'start_time': starttime,
                    'end_time': endtime,
                    'slope': slope,
                    'start_idx_slope_cv':slope_cv,
                    'intercept': intercept,
                    'r_value': r_value,
                    'adj_r2': adj_r2,
                    'n_points': n, 
                    'max_conc':sub_max_conc, 
                    'max_conc_time':max_time,
                })
    return pd.DataFrame(results)

def analyze_pk_data(df, subject_col = 'subject', time_col = 'time', conc_col = 'conc'):  # df has columns 'subject', 'time', 'conc'
    results = {}
    for subject_id in df[subject_col].unique():
        subject_data = df[df[subject_col] == subject_id].sort_values(time_col)
        times = subject_data[time_col].values
        concs = subject_data[conc_col].values

        # 1. Calculate AUC/Δt
        auc_dt = []
        for i in range(1, len(times)):
            auc = np.trapz(concs[:i+1], times[:i+1])  # Trapezoidal rule
            dt = times[i] - times[0]
            auc_dt.append(auc / dt)

        # 2. Calculate Differences
        delta_auc_dt = np.diff(auc_dt)

        # 3. Detect Terminal Elimination Phase Start (Example: Rolling SD)
        rolling_sd = pd.Series(delta_auc_dt).rolling(window=3).std().values
        #  (Implement thresholding logic here to find the change point)
        #  Example: Find first point where rolling_sd drops below 50% of its max
        threshold = 0.5 * np.nanmax(rolling_sd)
        t_elim_index = np.where(rolling_sd < threshold)[0]
        t_elim_index = t_elim_index[0] if len(t_elim_index) > 0 else None
        t_elim = times[t_elim_index+1] if t_elim_index is not None else None


        # 4. Detect Zero Concentration Start (Example: Consecutive Zeros)
        t_zero_index = None
        if t_elim_index is not None:
            for i in range(t_elim_index, len(delta_auc_dt)):
                if np.all(np.abs(delta_auc_dt[i:i+3]) < 0.001): # Check 3 consecutive points
                    t_zero_index = i
                    break
        t_zero = times[t_zero_index+1] if t_zero_index is not None else None

        # 5. Calculate Elimination Rate Constant (k)
        if t_elim is not None and t_zero is not None:
            elim_phase_data = subject_data[(subject_data[time_col] >= t_elim) & (subject_data[time_col] <= t_zero)]
            if len(elim_phase_data) > 1 :
                slope, intercept, r_value, p_value, std_err = linregress(elim_phase_data[time_col], np.log(elim_phase_data[conc_col]))
                k = -slope
            else:
                k = np.nan
        else:
            k = np.nan

        results[subject_id] = {'t_elim': t_elim, 't_zero': t_zero, 'k': k}

    return results