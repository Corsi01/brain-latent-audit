# Copy to env.sh and fill in with your own paths — env.sh is gitignored and
# must never be committed (see CLAUDE.md: no absolute paths, no account names).
#
#   cp env.example.sh env.sh
#   source env.sh

export CS_CODE=/path/to/clip-selection            # this repo, checked out somewhere persistent
export CS_CACHE=/path/to/scratch/clip-selection    # scratch space: caches, extracted latents, logs
export CS_STIMULI=/path/to/algonauts2025/stimuli/movies   # CNeuroMod / Algonauts 2025 stimuli (usage agreement required)
export CS_FMRI=/path/to/algonauts2025/download             # CNeuroMod / Algonauts 2025 fMRI derivatives
export CS_ACCOUNT=your-cluster-account             # SLURM account, used by jobs/*.sbatch

mkdir -p "$CS_CACHE"/{cache_1hz,fmri_flat_1hz,shortlist_feats,logs}
