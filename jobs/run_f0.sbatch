#!/bin/bash
#SBATCH --job-name=f0_probe
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# --account is cluster-specific: sbatch does not expand $CS_ACCOUNT inside an
# #SBATCH directive. Source env.sh, then pass it at submission:
#   source env.sh
#   sbatch --account="$CS_ACCOUNT" jobs/run_f0.sbatch

set -euo pipefail

: "${CS_CODE:?CS_CODE not set — source env.sh in your submission shell first}"
: "${CS_CACHE:?CS_CACHE not set — source env.sh in your submission shell first}"

cd "$CS_CODE"
source .venv/bin/activate

mkdir -p "$CS_CACHE/f0"

# i nodi di calcolo non hanno rete: il checkpoint deve venire dalla cache locale
export HF_HOME="$CS_CODE/hf_cache"
export HF_HUB_OFFLINE=1

# 1) sonde attraverso l'encoder (serve GPU)
python src/audit/f0_probe.py \
    --flat-dir "$CS_CACHE/fmri_flat_1hz" \
    --out      "$CS_CACHE/f0/probes.npz" \
    --n-draws  64 \
    --n-real   256

# 2) analisi (CPU, gira anche in interattivo o su login node)
python src/audit/f0_analysis.py \
    --latents "$CS_CACHE/latents_lag4" \
    --probes  "$CS_CACHE/f0/probes.npz" \
    --ref     zeros \
    --out     "$CS_CACHE/f0/report.txt"
