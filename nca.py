import pandas as pd
import numpy as np
from scipy.stats import linregress
from utils import safe_signed_log
from sklearn.metrics import auc
from typing import List


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
    df_out = pd.DataFrame(results)
    df_out['abs_cv'] = np.abs(df_out['start_idx_slope_cv'])
    df_out['cv_sign'] = np.sign(df_out['start_idx_slope_cv'])
    signs = df_out.groupby('start_time')['cv_sign'].mean().reset_index().rename(columns = {'cv_sign':'start_time_mean_cv_sign'})
    start_idx_cv_mean = df_out.groupby('start_time')['abs_cv'].mean().reset_index().rename(columns = {'abs_cv':'start_time_mean_abs_cv'})
    start_idx_cv_std = df_out.groupby('start_time')['abs_cv'].std().reset_index().rename(columns = {'abs_cv':'start_time_std_mean_cv'})
    df_out = (df_out
            .merge(signs, how = 'left', on = 'start_time')
            .merge(start_idx_cv_mean, how = 'left', on = 'start_time')
            .merge(start_idx_cv_std, how = 'left', on = 'start_time')
            )
    
    return df_out

def identify_low_conc_zones(dfs:List[pd.DataFrame], low_frac = 0.01):
    subject_zero_zones = []
    for tmp in dfs:
        max_auc = tmp['auc_per_time'].max()
        #f1 = tmp['auc_per_time'] < max_auc*.01
        tmp.loc[tmp['auc_per_time'] < max_auc*low_frac, 'auc_per_time_gt_lim'] = 0
        tmp.loc[tmp['auc_per_time'] >= max_auc*low_frac, 'auc_per_time_gt_lim'] = 1
        ind = (tmp.groupby('start_time')['auc_per_time_gt_lim'].sum(
        ).reset_index().rename(columns={'auc_per_time_gt_lim': 'start_time_ind'}))
        ind.loc[ind['start_time_ind'] != 0] = 1
        tmp = tmp.merge(ind, how = 'left', on = 'start_time')
        f1 = tmp['start_time_ind'] == 0
        time_at_max_conc = tmp['max_conc_time'].values[0]
        f2 = tmp['start_time'] > time_at_max_conc
        tmp_f = tmp.loc[f1 & f2, :]
        brack = tmp_f.loc[tmp_f['start_time'] == tmp_f['start_time'].min(), :]

        subject_zero_zones.append(
            {
                'ID':brack['ID'].values[0], 
                'zero_window_time_start':brack['start_time'].values[0],
                'consecutive_zero_windows':len(brack)
                
            }
        )
    return pd.DataFrame(subject_zero_zones)

def estimate_k_halflife(dfs, zero_zone_df = None):
    zero_zone_df = identify_low_conc_zones(dfs) if zero_zone_df is None else zero_zone_df
    res = []
    for tmp in dfs:
        #id = tmp['ID'].values[0]
        tmp = tmp.merge(zero_zone_df, how = 'left', on = 'ID')#.copy()
        f1 = tmp['start_time'] < tmp['zero_window_time_start']
        f2 = tmp['end_time'] <= tmp['zero_window_time_start']
        tmp = tmp.loc[f1 & f2, :]
        f = tmp['start_time_std_mean_cv'] == tmp['start_time_std_mean_cv'].min()
        out_df = tmp.loc[f, :].copy()
        out_df['window_k_est'] = -1*tmp.loc[f, 'slope'].values
        out_df['geom_mean_k_est'] = np.exp(np.mean(safe_signed_log(out_df['window_k_est'])))
        out_df['window_halflife_est'] = 0.693/out_df['window_k_est']
        out_df['geom_mean_halflife_est'] = np.exp(np.mean(safe_signed_log(out_df['window_halflife_est'])))
        res.append(out_df.copy())
    return pd.concat(res).reset_index(drop = True)

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

def log_trapazoidal_section_auc(time, conc):
    tmp_auc = ((np.diff(conc)
            /(np.diff(safe_signed_log(conc))+1e-6))
    *(np.diff(time)))
    tmp_auc[np.isnan(tmp_auc)] = 0
    return tmp_auc

def section_auc(time, conc):
    avg_conc = (conc[:-1] + conc[1:]) / 2
    delta_t = np.diff(time)
    return avg_conc*delta_t

def auc_trapz_slope_is_pos(conc):
    tmp = np.sign(np.diff(conc))
    tmp[tmp >= 0] = True
    tmp[tmp < 0] = False
    return tmp.astype(bool)

def calculate_aucs(time, conc):
    t = log_trapazoidal_section_auc(time, conc)
    n = section_auc(time, conc)
    s = auc_trapz_slope_is_pos(conc)
    
    auc_res = [
        {
            'linup_logdown':np.sum(n[s]) + np.sum(t[~s]),
            'logup_lindown':np.sum(n[~s]) + np.sum(t[s]), 
            'linear_auc':np.sum(n), 
            'log_auc':np.sum(t)
        }
        ]
    
    return pd.DataFrame(auc_res)