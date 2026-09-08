#!/usr/bin/env python3
"""
validate_lag.py — Empirically verify the BOLD lag AND the stimulus↔fMRI alignment
                  BEFORE spending GPU time on CortexMAE.

Idea: both the stimulus features (cache_1hz/) and the flattened fMRI
(fmri_flat_1hz/) live on the same 1 Hz timeline with index 0 = run onset.
If the alignment and lag convention are right, then

    corr( stim_feature[t],  fmri[t + LAG] )

must peak at a POSITIVE lag of roughly 4-6 s (the haemodynamic delay), and the
pixels driving that peak must form a spatially coherent blob (visual cortex for
luminance, auditory cortex for RMS).

If the peak sits at lag 0, or at a negative lag, the alignment is wrong and
every downstream number is meaningless.

Usage:
    python src/validate_lag.py                       # a few runs, all subjects
    python src/validate_lag.py --max-lag 12 --n-runs 6
"""

import argparse, json, os, sys
import numpy as np
from pathlib import Path

FEATURES = ["luminance", "rms", "speech_coverage", "frame_diff"]


def load_stim_1hz(run_id: str, cache: Path) -> dict[str, np.ndarray] | None:
    p = cache / "cache_1hz" / f"{run_id}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    out = {}
    for f in FEATURES:
        if f in d:
            out[f] = d[f].astype(np.float64)
    # speech_coverage lives in the windows file, not the npz → recompute is
    # unnecessary here; luminance/rms/frame_diff are enough for the check.
    return out


def lag_profile(stim: np.ndarray, fmri: np.ndarray, max_lag: int):
    """
    For each lag L in 0..max_lag, correlate stim[t] with fmri[t+L] per pixel,
    and report the peak |r| across pixels (and the mean of the top 1%).

    Positive L = fMRI LATER than stimulus = the physically correct direction.
    """
    T = min(len(stim), len(fmri))
    prof = []
    for L in range(max_lag + 1):
        n = T - L
        if n < 60:
            prof.append((L, np.nan, np.nan))
            continue
        s = stim[:n]
        f = fmri[L : L + n]                     # <-- fMRI shifted FORWARD by L

        s = s - s.mean()
        sd = s.std()
        if sd < 1e-8:
            prof.append((L, np.nan, np.nan))
            continue
        s = s / sd

        f = f - f.mean(axis=0, keepdims=True)
        fs = f.std(axis=0, keepdims=True)
        f = np.divide(f, np.clip(fs, 1e-8, None), where=fs > 1e-8)

        r = (s[:, None] * f).mean(axis=0)       # (D,) per-pixel correlation
        r = np.nan_to_num(r)
        top = np.sort(np.abs(r))[-max(1, len(r) // 100):]
        prof.append((L, float(np.abs(r).max()), float(top.mean())))
    return prof


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-lag", type=int, default=10)
    p.add_argument("--n-runs",  type=int, default=4)
    args = p.parse_args()

    code  = Path(os.environ["CS_CODE"])
    cache = Path(os.environ["CS_CACHE"])
    fdir  = cache / "fmri_flat_1hz"

    npys = sorted(fdir.glob("*.npy"))
    if not npys:
        sys.exit(f"No flattened fMRI in {fdir} — run flatten_fmri.py first")

    print(f"Found {len(npys)} flattened runs. Checking up to {args.n_runs}.\n")
    print("Convention under test:  corr( stim[t] , fmri[t + LAG] )")
    print("Expected: peak at LAG ≈ 4-6 s.  Peak at 0 or negative → MISALIGNED.\n")

    agg = {f: [] for f in FEATURES}

    for npy in npys[: args.n_runs]:
        stem = npy.stem                          # sub-01_friends_s06e01a
        sub, run_id = stem.split("_", 1)

        stim = load_stim_1hz(run_id, cache)
        if not stim:
            print(f"{stem}: no stimulus cache → skip")
            continue

        fmri = np.load(npy).astype(np.float32)   # (T, 77763)
        print(f"{stem}:  fmri T={fmri.shape[0]}s  stim T={len(next(iter(stim.values())))}s")

        for fname, sig in stim.items():
            prof = lag_profile(sig, fmri, args.max_lag)
            valid = [(L, a, b) for L, a, b in prof if not np.isnan(a)]
            if not valid:
                continue
            best = max(valid, key=lambda x: x[2])
            agg[fname].append(best[0])
            bar = "  ".join(
                f"{L}:{b:.3f}" + ("*" if L == best[0] else " ")
                for L, a, b in valid
            )
            print(f"    {fname:16s} peak@{best[0]}s  (top1% |r|={best[2]:.3f})")
            print(f"        {bar}")
        print()

    print("=" * 60)
    print("SUMMARY — peak lag per feature across runs")
    for f, lags in agg.items():
        if not lags:
            continue
        med = int(np.median(lags))
        ok = "✓" if 3 <= med <= 7 else "✗ SUSPICIOUS"
        print(f"  {f:16s} median peak lag = {med}s   {ok}   (all: {lags})")
    print("=" * 60)
    print("\nIf medians land in 4-6 s → alignment + lag convention are correct.")
    print("Use that value as --lag in extract_latents.py.")


if __name__ == "__main__":
    main()
