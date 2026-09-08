# CLAUDE.md

Context for agent sessions working on this repository.

## What this is

Research code for an audit of the latent space of CortexMAE-F, a
self-supervised foundation model for fMRI, evaluated on CNeuroMod /
Algonauts 2025 (4 subjects, 622 clip-subject latents).

Two connected pieces of work:

1. **Validation.** Does the frozen latent carry stimulus-locked signal, enough
   to serve as an auxiliary target in a stimulus-to-brain encoding model?
   Tested by inter-subject clip identification (26.6x chance, 39 SD above a
   permutation null).
2. **Audit.** During validation, subtracting the per-subject mean latent was
   found to raise identification by 40% while standardising lowered it. That
   asymmetry indicates an additive offset. Is it an intrinsic brain state, or
   the architectural anisotropy known in transformer representations? The
   audit separates the two, and finds the mean is not a direction but a
   low-dimensional subspace with components of different origin.

A written report accompanies the code and is the primary artefact; the code
exists to make the report verifiable.

## Current goal of these sessions

Turn this into a **standalone, public, presentable repository** for PhD
applications. It will not run without data and that is expected. It must be
readable, honest about what it does, and free of anything private.

Success looks like: a competent reader in this field opens the repo, reads the
README in ten minutes, and understands what was tested, what was found, and
what was ruled out.

## Hard constraints

**Never commit data.** No latents, no flat maps, no fMRI, no cache
directories, no model checkpoints. CNeuroMod data is governed by a usage
agreement and CortexMAE weights by their own licence. Derived scalar results
(correlations, effect sizes) are fine.

**Never commit paths or credentials.** No absolute paths, no cluster account
names, no usernames, no hostnames. Replace with environment variables or CLI
arguments with sensible defaults.

**Write .gitignore before the first commit.** Anything committed and later
removed stays in history.

**Do not alter analysis logic.** Refactoring for clarity is fine; changing
what a script computes is not. If a script looks wrong, say so, do not fix it
silently. The numbers in the report were produced by this code as it stands.

**Do not run git commands** (init, add, commit, remote, push) unless
explicitly asked in that session. Propose, then wait.

## Where things are

Source lives in `src/`. Data lives outside the repo under `$CS_CACHE`
(scratch) and is referenced through `$CS_CODE` / `$CS_CACHE` environment
variables set by `env.sh`. The model is loaded from a local HuggingFace cache
(`HF_HOME`, `HF_HUB_OFFLINE=1`) because compute nodes have no network.

### Script roles

**Data pipeline** (stimulus characterisation to latent extraction):
`build_run_index.py`, `build_fmri_index.py`, `characterize_run.py`,
`build_windows.py`, `select_clips.py`, `validate_lag.py`, `flatten_fmri.py`,
`extract_latents.py`

**Validation analyses** (the retrieval work):
`retrieval.py`, `retrieval_variants.py`, `drive_regression.py`,
`fair_comparison.py`, `pca_sweep.py`, `pca_loso.py`,
`probe_grain_and_mean.py`, `baseline_probe.py`, `final_table.py`

**Audit analyses** (the geometry work):
`f0_probe.py`, `f0_analysis.py` — controlled probes through the encoder,
decomposition against the null-input response.
`randinit_extract.py`, `randinit_compare.py`, `randinit_f0check.py` —
weight-randomised encoders under two schemes.
`input_retrieval.py` — the same identification protocol run on input flat
maps, as the honest baseline.
`map_stability.py`, `verify_patch_order.py`, `patch_to_surface.py` — the
patch-to-cortex chain and its two verifications.
`positive_control.py`, `coverage_check.py` — anatomical positive control and
the geometric confound.
`atlas_comparison.py`, `yeo_test.py` — connectivity gradients and Yeo
networks, with spin tests.
`diagnose_negative.py`, `validate_pcs.py`, `characterize_pcs.py` — why the
primary test failed, and the subspace decomposition.
`plot_maps.py` — rendering.

## Style

Analysis scripts are single-purpose and runnable from the command line with
`argparse`. Each has a docstring stating what question it answers and why the
controls in it are there. Preserve that: the docstrings carry the reasoning
and are part of what makes the repo readable.

Prefer explicit over clever. This code is read more than it is run.