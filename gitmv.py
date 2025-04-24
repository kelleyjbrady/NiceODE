#%%
import subprocess
import os
import glob

#%%
debug_mv_files = glob.glob("/workspaces/PK-Analysis/niceode/*")

# %%
for f in debug_mv_files:
    base = os.path.basename(f)
    write_path = os.path.join('src','niceode', base)
    subprocess.run(['git', 'mv', f, write_path])
# %%
