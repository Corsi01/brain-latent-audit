#!/usr/bin/env python3
"""
drive_regression.py — Which sensory channel drives identifiability?

The category-based drive curve confounds channels: SPEECH_DYNAMIC has speech AND
motion AND high luminance, so when it retrieves well we cannot say which one did it.
The ordering *looks* like it tracks luminance, but "looks like" is not a measurement.

This regresses per-clip retrieval performance on the continuous stimulus axes:

    pct_rank_i  ~  luminance_i + frame_diff_i + rms_i + speech_coverage_i

The point is the word "controlling". A coefficient tells us how much a channel
contributes AT FIXED LEVELS OF THE OTHERS. If rms collapses to zero once luminance
is in the model, then audio only appeared to matter because it correlates with the
visual channel — it does not drive identification itself.

Reported per predictor:
  beta        standardised coefficient (all predictors z-scored → comparable)
  p           two-sided, via permutation of the residuals
  partial r   correlation with pct-rank after regressing out the other predictors
  ΔR²         drop in R² when that predictor alone is removed (unique variance)

NOTE the sign convention: pct-rank is a LOSS (0 = perfect, 0.5 = chance).
A NEGATIVE beta means MORE of that feature → BETTER retrieval.

Usage:
    source env.sh && python src/drive_regression.py --lag 4
    python src/drive_regression.py --lag 4 --pooling flat
"""

import argparse, json, os, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from retrieval import load_latents, evaluate

PREDICTORS = ["luminance", "frame_diff", "rms", "speech_coverage"]


def ols(X: np.ndarray, y: np.ndarray):
    """Least squares with intercept. X (n, p) already z-scored. Returns beta, r2, resid."""
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    resid = y - pred
    ss_res = (resid ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return coef[1:], r2, resid


def partial_corr(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> float:
    """corr(x, y) after regressing both on Z."""
    if Z.shape[1] == 0:
        return float(np.corrcoef(x, y)[0, 1])
    A = np.column_stack([np.ones(len(Z)), Z])
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lag", type=int, required=True)
    p.add_argument("--pooling", default="flat", choices=["spatial", "full", "flat"])
    p.add_argument("--n-perm", type=int, default=5000)
    args = p.parse_args()

    code  = Path(os.environ["CS_CODE"])
    cache = Path(os.environ["CS_CACHE"])

    # ---- per-clip retrieval performance ----
    X_lat, _, meta = load_latents(cache / f"latents_lag{args.lag}", pooling=args.pooling)
    ranks, ncand, clips = evaluate(X_lat, meta)
    pct = ranks / np.clip(ncand - 1, 1, None)

    by_clip = defaultdict(list)
    for r, c in zip(pct, clips):
        by_clip[c].append(r)
    clip_pct = {c: float(np.mean(v)) for c, v in by_clip.items()}

    # ---- stimulus predictors ----
    with open(code / "manifest" / "clip_manifest.json") as f:
        manifest = {c["clip_id"]: c for c in json.load(f)["clips"]}

    rows, y, labels, films = [], [], [], []
    for cid, val in clip_pct.items():
        c = manifest.get(cid)
        if c is None:
            continue
        dv = c["drive_vector"]
        if any(dv.get(k) is None for k in PREDICTORS):
            continue
        rows.append([float(dv[k]) for k in PREDICTORS])
        y.append(val)
        labels.append(c["content_label"])
        films.append(c["film_id"])

    Xr = np.array(rows)
    y  = np.array(y)
    n  = len(y)
    print(f"n = {n} clips   pooling = {args.pooling}   lag = {args.lag}")
    print(f"pct-rank: mean {y.mean():.3f}  sd {y.std():.3f}  (0 = perfect, 0.5 = chance)\n")

    # ---- collinearity among the predictors (read this BEFORE the betas) ----
    print("Predictor correlations — if these are high, the betas are unstable:")
    print(f"  {'':>16s} " + " ".join(f"{k[:8]:>9s}" for k in PREDICTORS))
    C = np.corrcoef(Xr.T)
    for i, k in enumerate(PREDICTORS):
        print(f"  {k:>16s} " + " ".join(f"{C[i, j]:9.2f}" for j in range(len(PREDICTORS))))
    print()

    # ---- z-score so betas are comparable ----
    Xz = (Xr - Xr.mean(0)) / np.clip(Xr.std(0), 1e-9, None)

    beta, r2_full, _ = ols(Xz, y)

    # permutation null for each beta
    rng = np.random.default_rng(0)
    null = np.zeros((args.n_perm, len(PREDICTORS)))
    for i in range(args.n_perm):
        null[i] = ols(Xz, rng.permutation(y))[0]
    pvals = [(np.abs(null[:, j]) >= abs(beta[j])).mean() for j in range(len(PREDICTORS))]

    print("=" * 78)
    print(f"FULL MODEL   R² = {r2_full:.3f}")
    print("  negative beta = more of this feature → BETTER retrieval")
    print("=" * 78)
    print(f"  {'predictor':>16s} {'beta':>8s} {'p':>8s} {'partial r':>10s} {'ΔR²':>8s}")
    print("  " + "-" * 56)

    for j, k in enumerate(PREDICTORS):
        others = np.delete(Xz, j, axis=1)
        pr = partial_corr(Xz[:, j], y, others)
        r2_drop = ols(others, y)[1]
        d = r2_full - r2_drop
        star = "***" if pvals[j] < 0.001 else "**" if pvals[j] < 0.01 else "*" if pvals[j] < 0.05 else ""
        print(f"  {k:>16s} {beta[j]:8.3f} {pvals[j]:8.4f} {pr:10.3f} {d:8.3f}  {star}")

    # ---- visual-only vs audio-only ----
    vis = [PREDICTORS.index("luminance"), PREDICTORS.index("frame_diff")]
    aud = [PREDICTORS.index("rms"), PREDICTORS.index("speech_coverage")]
    r2_vis = ols(Xz[:, vis], y)[1]
    r2_aud = ols(Xz[:, aud], y)[1]

    print("\n" + "=" * 78)
    print("CHANNEL DECOMPOSITION")
    print("=" * 78)
    print(f"  visual only (luminance + frame_diff)      R² = {r2_vis:.3f}")
    print(f"  audio  only (rms + speech_coverage)       R² = {r2_aud:.3f}")
    print(f"  both                                      R² = {r2_full:.3f}")
    print(f"\n  unique to audio  (full − visual only)     ΔR² = {r2_full - r2_vis:.3f}")
    print(f"  unique to visual (full − audio only)      ΔR² = {r2_full - r2_aud:.3f}")

    if r2_full - r2_vis < 0.02:
        print("\n  → audio adds essentially NOTHING once the visual channel is accounted for.")
        print("    The latent is stimulus-locked through the VISUAL channel only.")
    elif r2_full - r2_aud < 0.02:
        print("\n  → visual adds nothing once audio is accounted for. Surprising — check.")
    else:
        print("\n  → both channels carry unique variance.")

    # ---- film as a confound: does the effect survive within-film? ----
    print("\n" + "=" * 78)
    print("FILM CONTROL — same model, film means removed")
    print("=" * 78)
    y_w  = y.copy()
    Xz_w = Xz.copy()
    for f in set(films):
        m = np.array([fl == f for fl in films])
        if m.sum() > 1:
            y_w[m]     -= y_w[m].mean()
            Xz_w[m]    -= Xz_w[m].mean(0)
    beta_w, r2_w, _ = ols(Xz_w, y_w)
    print(f"  within-film R² = {r2_w:.3f}   (was {r2_full:.3f})")
    print(f"  {'predictor':>16s} {'beta':>8s}")
    for j, k in enumerate(PREDICTORS):
        print(f"  {k:>16s} {beta_w[j]:8.3f}")
    print("\n  If the betas keep their sign and magnitude here, the effect is not")
    print("  film identity — it holds inside each film.")


if __name__ == "__main__":
    main()
