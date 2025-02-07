import pandas as pd
import numpy as np
from scipy.stats import linregress

def find_terminal_phase(df, min_points=4):
    """
    Finds the start of the terminal elimination phase in PK data using
    rolling window linear regressions and adjusted R-squared.

    Args:
        time (array-like): Time points.
        concentration (array-like): Corresponding drug concentrations.
        min_points (int): Minimum number of points required for a regression.

    Returns:
        tuple: (best_start_index, best_slope, best_intercept, best_adj_r2, results_df)
               - best_start_index: Index in the time/concentration arrays
                 corresponding to the start of the best-fit terminal phase.
               - best_slope: Slope of the best-fit line (negative of k_elim).
               - best_intercept: Intercept of the best-fit line.
               - best_adj_r2: Adjusted R-squared of the best-fit line.
               - results_df: DataFrame containing the results of all regressions.
               Returns (None, None, None, None, None) if no suitable terminal
               phase is found.
    """
    times = df['TIME'].unique()
    

    results = []
    for start_index in range(len(times) - min_points + 1):
        for end_index in range(start_index + min_points, len(times) + 1): # Iterate through end indices
            
            eval_times = times[start_index:end_index]
            #current_ln_conc = ln_concentration[start_index:end_index]
            x = df.loc[df['TIME'].isin(eval_times), 'TIME']
            y = df.loc[df['TIME'].isin(eval_times), 'CONC_ln']

            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = linregress(x, y)

            # Calculate adjusted R-squared
            n = len(eval_times)
            k = 1  # Number of predictors (time)
            adj_r2 = 1 - (1 - r_value**2) * (n - 1) / (n - k - 1)

            results.append({
                'ID':df['ID'].unique()[0],
                'start_index': start_index,
                'end_index': end_index,
                'start_time': eval_times[0],
                'end_time': eval_times[-1],
                'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'adj_r2': adj_r2,
                'n_points': n
            })

    results_df = pd.DataFrame(results)

    # Find the regression with the highest adjusted R-squared
    if not results_df.empty:
        best_row = results_df.loc[results_df['adj_r2'].idxmax()]
        best_start_index = int(best_row['start_index'])  # Ensure it's an integer
        best_slope = best_row['slope']
        best_intercept = best_row['intercept']
        best_adj_r2 = best_row['adj_r2']

        return results_df
    else:
         return None