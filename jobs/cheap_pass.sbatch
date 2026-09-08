#!/bin/bash
#SBATCH --job-name=clipsel-1hz
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --time=00:15:00
#SBATCH --array=0-93
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# --account and --output/--error are cluster-specific: sbatch does not expand
# $CS_ACCOUNT/$CS_CACHE inside an #SBATCH directive. Source env.sh, then pass
# them at submission if you want centralized logs and your own account:
#   source env.sh
#   sbatch --account="$CS_ACCOUNT" --output="$CS_CACHE/logs/1hz_%A_%a.out" \
#          --error="$CS_CACHE/logs/1hz_%A_%a.err" jobs/cheap_pass.sbatch

# --- each array task processes one run (~30-60s) ---

: "${CS_CODE:?CS_CODE not set — source env.sh in your submission shell first}"
cd "$CS_CODE"
source .venv/bin/activate

echo "$(date) — task $SLURM_ARRAY_TASK_ID on $(hostname)"
python src/pipeline/characterize_run.py --task-id "$SLURM_ARRAY_TASK_ID"
echo "$(date) — done"

# -----------------------------------------------------------------------
# NOTE: se la partizione non accetta --gres=gpu:0 o se vuoi evitare
# di occupare nodi GPU per lavoro CPU, verifica con:
#   sinfo -p boost_usr_prod -o "%P %G %a"
# e sostituisci con una partizione CPU se disponibile (es. lrd_all_serial).
#
# Alternativa sequenziale (un solo job, ~1.5h):
#   sbatch --array=0 jobs/cheap_pass.sbatch  # poi dentro usa --all
# -----------------------------------------------------------------------
