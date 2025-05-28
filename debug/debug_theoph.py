#%%
import pandas as pd

df = pd.read_table(r"/workspaces/PK-Analysis/data/theoph.txt", ) 
# %%


df.loc[df['CONC'] == '.', 'CONC'] = 0
df.loc[df['AMT'] == '.', 'AMT'] = pd.NA
df[['CONC', 'TIME', 'AMT']] = df[['CONC', 'TIME', 'AMT']].astype(pd.Float64Dtype())
# %%

df['AMT'] = df['AMT'].ffill()
df['AMT'] = 320
#%%
from niceode.nca import NCA
nca_obj = NCA(
    subject_id_col='ID', 
    conc_col='CONC',
    time_col='TIME', 
    dose_col='AMT',
    data = df
)
# %%
noboot_tmp = nca_obj.estimate_all_nca_params(terminal_phase_adj_r2_thresh=0.85)
# %%
nca_obj.