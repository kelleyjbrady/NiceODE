set -e # Exit immediately if a command exits with a non-zero status.

echo "--- Installing Python dependencies ---"
cd /workspaces/PK-Analysis && poetry install

echo "--- Installing R dependencies ---"
R -e 'install.packages("nlmixr2", repos="https://cloud.r-project.org")'

echo "--- Post-start setup complete ---"