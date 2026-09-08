#!/bin/bash
#SBATCH --job-name=cs-flatten
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-255
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# --account and --output/--error are cluster-specific: sbatch does not expand
# $CS_ACCOUNT/$CS_CACHE inside an #SBATCH directive. Source env.sh, then pass
# them at submission if you want centralized logs and your own account:
#   source env.sh
#   sbatch --account="$CS_ACCOUNT" --output="$CS_CACHE/logs/flat_%A_%a.out" \
#          --error="$CS_CACHE/logs/flat_%A_%a.err" jobs/flatten_fmri.sbatch

# CPU-bound (neuromaps projection). Set --array=0-(N-1) where N = n_entries
# from build_fmri_index.py. Start with a small range to time one entry.

: "${CS_CODE:?CS_CODE not set — source env.sh in your submission shell first}"
cd "$CS_CODE"
source .venv/bin/activate

echo "$(date) — task $SLURM_ARRAY_TASK_ID on $(hostname)"
python src/pipeline/flatten_fmri.py --task-id "$SLURM_ARRAY_TASK_ID"
echo "$(date) — done"
