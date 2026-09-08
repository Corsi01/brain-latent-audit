#!/bin/bash
#SBATCH --job-name=cs-latents
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# --account and --output/--error are cluster-specific: sbatch does not expand
# $CS_ACCOUNT/$CS_CACHE inside an #SBATCH directive. Source env.sh, then pass
# them at submission if you want centralized logs and your own account:
#   source env.sh
#   sbatch --account="$CS_ACCOUNT" --output="$CS_CACHE/logs/lat_%A_%a.out" \
#          --error="$CS_CACHE/logs/lat_%A_%a.err" jobs/extract_latents.sbatch

# One array task per subject. IMPORTANT: set LAG from validate_lag.py output.
LAG=4

: "${CS_CODE:?CS_CODE not set — source env.sh in your submission shell first}"
export HF_HOME="$CS_CODE/hf_cache"
export HF_HUB_OFFLINE=1
cd "$CS_CODE"
source .venv/bin/activate

echo "$(date) — task $SLURM_ARRAY_TASK_ID on $(hostname), LAG=$LAG"
python src/pipeline/extract_latents.py --lag "$LAG" --task-id "$SLURM_ARRAY_TASK_ID"
echo "$(date) — done"
