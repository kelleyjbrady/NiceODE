import pandas as pd
import numpy as np
from scipy.stats import linregress
from utils import safe_signed_log
from sklearn.metrics import auc
from typing import List


def estimate_subject_slope_cv(df, time_col = 'TIME', conc_col = 'CONC_ln', id_col = 'ID',):
    df = df.copy()
    sub_max_conc = df[conc_col].max()
    max_time = df.loc[df[conc_col] == sub_max_conc, time_col].values[-1]
    #zero_window_start = df[zero_start_col].values[0]
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
                    #tmp_ar = np.array(start_idx_slopes)
                    start_idx_slopes_signs = np.mean(np.sign(start_idx_slopes))
                    
                else:
                    slope_cv = None
                    start_idx_slopes_signs = None
                results.append({
                    'ID':work_df[id_col].unique()[0],
                    #'start_index': start_idx,
                    #'end_index': end_idx,
                    'auc_per_time':auc_per_time,
                    'start_time': starttime,
                    'end_time': endtime,
                    'slope': slope,
                    'startidx_endidx_slope_cv':slope_cv,
                    'startidx_endidx_slope_sign':np.sign(slope),
                    'intercept': intercept,
                    'r_value': r_value,
                    'adj_r2': adj_r2,
                    'n_points': n, 
                    'max_conc':sub_max_conc, 
                    'max_conc_time':max_time,
                })
    df_out = pd.DataFrame(results)
    df_out['abs_cv'] = np.abs(df_out['startidx_endidx_slope_cv'])
    #df_out['cv_sign'] = np.sign(df_out['start_idx_slope_cv'])
    signs = df_out.groupby('start_time')['startidx_endidx_slope_sign'].mean().reset_index().rename(columns = {'startidx_endidx_slope_sign':'start_time_mean_slope_sign'})
    start_idx_cv_mean = df_out.groupby('start_time')['abs_cv'].mean().reset_index().rename(columns = {'abs_cv':'start_time_mean_abs_cv'})
    start_idx_cv_std = df_out.groupby('start_time')['abs_cv'].std().reset_index().rename(columns = {'abs_cv':'start_time_std_mean_cv'})
    df_out = (df_out
            .merge(signs, how = 'left', on = 'start_time')
            .merge(start_idx_cv_mean, how = 'left', on = 'start_time')
            .merge(start_idx_cv_std, how = 'left', on = 'start_time')
           )
    
    return df_out

def masked_signed_safe_log(x):
    s = np.zeros_like(x, dtype=np.float64)  # Initialize with zeros

    non_zero_mask = x != 0

    s[non_zero_mask] = np.sign(x[non_zero_mask]) * np.log(np.abs(x[non_zero_mask]))

    return s 


def indentify_low_conc_zones2(df,
                              time_col = 'TIME',
                              conc_col = 'CONC_ln',
                              id_col = 'ID', 
                              low_frac = 0.005
                              ):
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
                
                n = len(work_df[time_col].unique())
                results.append({
                    'ID':work_df[id_col].unique()[0],
                    #'start_index': start_idx,
                    #'end_index': end_idx,
                    'auc_per_time':auc_per_time,
                    'start_time': starttime,
                    'end_time': endtime,
                    'n_points': n, 
                    'max_conc':sub_max_conc, 
                    'max_conc_time':max_time,
                })
    tmp_res = pd.DataFrame(results)
    return identify_low_conc_zones([tmp_res], low_frac = low_frac)


def identify_low_conc_zones(dfs:List[pd.DataFrame], low_frac = 0.01):
    subject_zero_zones = []
    for tmp in dfs:
        id = tmp['ID'].values[0]
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
                'ID':id, 
                'zero_window_time_start':brack['start_time'].values[0] if len(brack) > 0 else np.inf,
                'consecutive_zero_windows':len(brack)
                
            }
        )
    return pd.DataFrame(subject_zero_zones)

def estimate_k_halflife(dfs, zero_zone_df = None, adj_r2_threshold = 0.8):
    zero_zone_df = identify_low_conc_zones(dfs) if zero_zone_df is None else zero_zone_df
    res = []
    adj_r2_ind_col = f"adj_r2_gte_{adj_r2_threshold}"
    for tmp in dfs:
        if tmp['ID'].values[0] == 'M9':
            debugging = True
        #id = tmp['ID'].values[0]
        tmp = tmp.merge(zero_zone_df, how = 'left', on = 'ID')#.copy()
        f1 = tmp['start_time'] < tmp['zero_window_time_start']
        f2 = tmp['end_time'] <= tmp['zero_window_time_start']
        tmp = tmp.loc[f1 & f2, :]
        
        tmp['startidx_endidx_slope_sign'] = np.sign(tmp['slope'])
        start_idx_avg_slope = (tmp
                                          .groupby('start_time')['startidx_endidx_slope_sign']
                                          .mean()
                                          .reset_index()
                                          .rename(columns = {'startidx_endidx_slope_sign':'startidx_avg_slope_sign'})
                                          
                                          )
        tmp = tmp.merge(start_idx_avg_slope, how = 'left', on = 'start_time')
        f3 = tmp['startidx_avg_slope_sign'] == -1
        tmp = tmp.loc[f3, :]
        tmp['adj_r2_threshold'] = adj_r2_threshold
        tmp.loc[tmp['adj_r2'] >= adj_r2_threshold, 'adj_r2_gte_threshold'] = 1
        tmp.loc[tmp['adj_r2'] < adj_r2_threshold, 'adj_r2_gte_threshold'] = 0
        start_idx_thresh = (tmp
                        .groupby('start_time')['adj_r2_gte_threshold']
                        .mean()
                        .reset_index()
                        .rename(columns = {'adj_r2_gte_threshold':'startidx_avg_adj_r2_gte_threshold'})
                        
                            )
        avg_adj_r2 = (tmp
                        .groupby('start_time')['adj_r2']
                        .mean()
                        .reset_index()
                        .rename(columns = {'adj_r2':'startidx_avg_adj_r2'})
                        
                            )
        tmp = tmp.merge(avg_adj_r2, how = 'left', on = 'start_time')
        tmp = tmp.merge(start_idx_thresh, how = 'left', on = 'start_time')
        f = tmp['startidx_avg_adj_r2_gte_threshold'] == 1 #the way this work is allowing the window to be wrong somtimes. Test on M9 to see an example. 
        good_liniearity_df = tmp.loc[f, :]
        if len(good_liniearity_df) > 0:
            tmp = good_liniearity_df.copy()
            f = tmp['startidx_avg_adj_r2'] == tmp['startidx_avg_adj_r2'].max()
            tmp = tmp.loc[f, :].copy()
            f = tmp['end_time'] == tmp['end_time'].max()
            tmp = tmp.loc[f, :].copy()
            

            #out_df['geom_mean_halflife_est'] = np.exp(np.mean(masked_signed_safe_log(out_df['window_halflife_est'])))
            tmp['method'] = 'adj_r2'
            
        if len(good_liniearity_df) == 0:
            max_end = tmp['end_time'].max()
            tmp = tmp.loc[tmp['end_time'] == max_end]
            max_start = tmp['start_time'].max()
            tmp = tmp.loc[tmp['start_time'] == max_start]
            tmp['method'] = 'final_nonzero_section'
            debugging = True
        out_df = tmp.copy()
        out_df['window_k_est'] = -1*out_df['slope'].values
        #out_df['geom_mean_k_est'] = np.exp(np.mean(masked_signed_safe_log(out_df['window_k_est'])))
        out_df['window_halflife_est'] = 0.693/out_df['window_k_est']
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
    
    wiggle = 0.0
    old_method = False
    if old_method:
        #orig_settings = np.seterr(divide='ignore')
        tmp_auc = ((np.diff(conc)
                /(np.diff(safe_signed_log(conc))+wiggle))
        *(np.diff(time)))
        #np.seterr(**orig_settings)
        tmp_auc[np.isnan(tmp_auc)] = 0
    else:
        conc_diff = np.diff(conc)
        log_conc_diff = np.diff(safe_signed_log(conc))
        time_diff = np.diff(time)
        tmp_auc = np.zeros_like(time_diff)
        nonzero_mask = log_conc_diff != 0
        tmp_auc[nonzero_mask] = (
            (conc_diff[nonzero_mask]/log_conc_diff[nonzero_mask])*time_diff[nonzero_mask]
            )
        #tmp_auc = s
    
    return tmp_auc

def section_auc(time, conc):
    avg_conc = (conc[:-1] + conc[1:]) / 2
    delta_t = np.diff(time)
    return avg_conc*delta_t

def auc_trapz_slope_is_pos(conc):
    tmp = np.sign(np.diff(conc))
    tmp[tmp > 0] = True
    tmp[tmp <= 0] = False
    return tmp.astype(bool)

def extend_auc_to_inf(time, conc, zero_start,terminal_k):
    final_conc = conc[time < zero_start][-1]
    final_time = time[time < zero_start][-1]
    auc_to_inf = final_conc/terminal_k
    
    
    auc_res = pd.DataFrame()
    auc_res['time_start'] = [final_time]
    auc_res['time_end'] = [np.inf]
    auc_res['conc_start'] = [final_conc]
    auc_res['conc_end'] = [0.0]
    auc_res['section_auc_log_trap'] = [auc_to_inf]
    auc_res['section_auc'] = [auc_to_inf]
    #auc_res['section_auc_alt'] = n_alt
    auc_res['section_slope_is_pos'] = [False]
   
    return auc_res

def extend_aumc_to_inf(time, conc, zero_start,terminal_k):
    final_time = time[time < zero_start][-1]
    final_conc = conc[time < zero_start][-1]
    aumc_to_inf = (final_conc * final_time / terminal_k) + (final_conc / (terminal_k**2))
    
    auc_res = pd.DataFrame()
    auc_res['time_start'] = [final_time]
    auc_res['time_end'] = [np.inf]
    auc_res['conc_start'] = [final_conc]
    auc_res['conc_end'] = [0.0]
    auc_res['section_auc_log_trap'] = [aumc_to_inf]
    auc_res['section_auc'] = [aumc_to_inf]
    #auc_res['section_auc_alt'] = n_alt
    auc_res['section_slope_is_pos'] = [False]
    return auc_res

def generate_auc_res_df(time, conc, log_trap_auc_comp, linear_auc_comp, auc_section_slope, ):
    auc_res = pd.DataFrame()
    auc_res['time_start'] = time[:-1] 
    auc_res['time_end'] = time[1:]
    auc_res['conc_start'] = conc[:-1]
    auc_res['conc_end'] = conc[1:]
    auc_res['section_auc_log_trap'] = log_trap_auc_comp
    auc_res['section_auc'] = linear_auc_comp
    #auc_res['section_auc_alt'] = n_alt
    auc_res['section_slope_is_pos'] = auc_section_slope
    s=auc_section_slope
    #auc_res['linup_logdown'] = np.sum(linear_auc_comp[s]) + np.sum(log_trap_auc_comp[~s])
    #auc_res['logup_lindown'] = np.sum(linear_auc_comp[~s]) + np.sum(log_trap_auc_comp[s])
    #auc_res['linear_auc'] = np.sum(linear_auc_comp)
    #auc_res['lin_auc_alt'] = n_alt
    #auc_res['log_auc'] = np.sum(log_trap_auc_comp)
    
    return auc_res

def calculate_aucs(time, conc, zero_start =None, terminal_k=None, ):

    
    t = log_trapazoidal_section_auc(time, conc)
    
    n = section_auc(time, conc)

    #n_alt = auc(time, conc)
    s = auc_trapz_slope_is_pos(conc)

    
    auc_res = generate_auc_res_df(time, conc, t, n, s)
    
    
    return auc_res