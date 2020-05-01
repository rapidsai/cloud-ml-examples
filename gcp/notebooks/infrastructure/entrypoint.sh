source /conda/etc/profile.d/conda.sh
conda activate rapids

echo "Running: entrypoint.py $@"
python entrypoint.py $@
