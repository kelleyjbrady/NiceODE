#%%
import subprocess
import os
import glob

#%%
debug_mv_files = glob.glob("/workspaces/PK-Analysis/debug_*.py")

# %%
for f in debug_mv_files:
    base = os.path.basename(f)
    write_path = os.path.join('debug', base)
    subprocess.run(['git', 'mv', f, write_path])
# %%
