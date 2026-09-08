#!/usr/bin/env python3
"""
pca_sweep.py — Find the smallest channel-PCA k that preserves stimulus-locking.

Context: PCA k=250 (364,000 d) BEAT the full latent (1,118,208 d) on pct-rank
(0.217 vs 0.240) while raising top-1. So ~20% of the variance was noise that was
dirtying the cosine. The question is how much further we can cut.

The number we want is NOT the k that maximises retrieval — it is the smallest k
that still preserves BOTH:
  (a) retrieval accuracy, and
  (b) the drive gradient.

(b) matters as much as (a). A representation could retrieve well on non-evoked
signal (the voxels do exactly this — they identify BLACK clips almost as well as
rich ones, gradient +0.066). If the gradient collapses as k shrinks, the surviving
components carry identification but not stimulus-locking — useless as an encoding
target, however good the headline number looks.

The PCA basis is fit ONCE and reused for every k (truncation is just taking fewer
columns), so the whole sweep costs one eigendecomposition.

Usage:
    python src/pca_sweep.py --lag 4
    python src/pca_sweep.py --lag 4 --ks 16 32 64 96 128 192 256 384
    python src/pca_sweep.py --lag 4 --center subject --whiten
"""

import argparse, os, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # validation/ — retrieval.py lives there
from retrieval import evaluate, summarize
from retrieval_variants import load_raw, fit_channel_pca, N_BLOCKS, EMB_DIM

# reference points measured earlier
REF = {
    "latent flat (1.12M)":  (0.240, 0.172, 0.220),
    "voxel full  (1.24M)":  (0.197, 0.116, 0.066),
    "CLS         (768)":    (0.407, 0.020, 0.067),
}


def eval_at_k(P, meta, evecs, evals, mean_fn, k, whiten):
    V = evecs[:, :k]
    if whiten:
        V = V / np.sqrt(np.clip(evals[:k], 1e-12, None))

    X = {}
    for s in sorted(P):
        m = mean_fn(s)
        X[s] = {c: ((v - m) @ V).ravel().astype(np.float32) for c, v in P[s].items()}

    ranks, ncand, clips = evaluate(X, meta)
    overall = summarize(ranks, ncand)

    by_cat = defaultdict(lambda: ([], []))
    for r, nc, c in zip(ranks, ncand, clips):
        lab = meta[c]["content_label"]
        by_cat[lab][0].append(r)
        by_cat[lab][1].append(nc)

    def pct(cat):
        if cat not in by_cat:
            return None
        r, nc = by_cat[cat]
        return summarize(np.array(r), np.array(nc))["pct"]

    black, dyn = pct("BLACK"), pct("SPEECH_DYNAMIC")
    grad = (black - dyn) if (black is not None and dyn is not None) else None

    del X
    return overall["pct"], overall["top1"], grad, dyn, black


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lag", type=int, required=True)
    p.add_argument("--ks", type=int, nargs="+",
                   default=[8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768])
    p.add_argument("--center", default="global", choices=["global", "subject"])
    p.add_argument("--whiten", action="store_true")
    args = p.parse_args()

    cache = Path(os.environ["CS_CACHE"])

    print(f"lag={args.lag}  center={args.center}  whiten={args.whiten}\n")

    P, _, meta = load_raw(cache / f"latents_lag{args.lag}")
    n_sp = next(iter(P[sorted(P)[0]].values())).shape[1]
    print(f"{len(P)} subjects × {len(meta)} clips   patches={n_sp}\n")

    evals, evecs, mean_fn = fit_channel_pca(P, center=args.center)
    cum = np.cumsum(evals) / evals.sum()

    print("\n" + "=" * 84)
    print("REFERENCE")
    print("=" * 84)
    print(f"  {'':>20s} {'pct-rank':>9s} {'top-1':>7s} {'gradient':>9s}")
    for name, (pc, t1, g) in REF.items():
        print(f"  {name:>20s} {pc:9.3f} {t1:7.3f} {g:+9.3f}")

    print("\n" + "=" * 84)
    print("SWEEP — smallest k that keeps BOTH accuracy and the drive gradient")
    print("=" * 84)
    print(f"  {'k':>4s} {'dims':>10s} {'var%':>6s} {'pct-rank':>9s} {'top-1':>7s} "
          f"{'gradient':>9s} {'DYN':>7s} {'BLACK':>7s}")
    print("  " + "-" * 74)

    rows = []
    for k in args.ks:
        if k > EMB_DIM:
            continue
        pc, t1, g, dyn, blk = eval_at_k(P, meta, evecs, evals, mean_fn, k, args.whiten)
        dims = N_BLOCKS * n_sp * k
        rows.append((k, dims, cum[k-1], pc, t1, g, dyn, blk))
        gs = f"{g:+9.3f}" if g is not None else "        —"
        print(f"  {k:4d} {dims:10,d} {cum[k-1]*100:5.1f}% {pc:9.3f} {t1:7.3f} "
              f"{gs} {dyn:7.3f} {blk:7.3f}", flush=True)

    # ---- pick the knee ----
    best = min(rows, key=lambda r: r[3])          # lowest pct-rank
    k_b, _, _, pc_b, t1_b, g_b, _, _ = best

    print("\n" + "=" * 84)
    print("VERDICT")
    print("=" * 84)
    print(f"  best pct-rank  : k={k_b}  ({pc_b:.3f}, top-1 {t1_b:.3f}, grad {g_b:+.3f})")

    # smallest k within 2% of best, with gradient still ≥ 80% of the best one
    ok = [r for r in rows
          if r[3] <= pc_b * 1.02 and r[5] is not None and g_b is not None
          and r[5] >= 0.8 * g_b]
    if ok:
        k_m, d_m, v_m, pc_m, t1_m, g_m, _, _ = min(ok, key=lambda r: r[0])
        print(f"  smallest viable: k={k_m}  ({d_m:,} dims, {v_m*100:.0f}% var)")
        print(f"                   pct-rank {pc_m:.3f}, top-1 {t1_m:.3f}, grad {g_m:+.3f}")
        print(f"\n  → {1118208/d_m:.1f}× smaller than the full latent, "
              f"{'better' if pc_m < 0.240 else 'worse'} retrieval")
        print(f"  → as an architecture this is a 1×1 conv, 768 → {k_m}, "
              f"PCA-initialised")

    print("\n  Read the gradient column as carefully as the accuracy column.")
    print("  A k that retrieves well but loses the gradient is identifying on")
    print("  non-evoked structure — exactly what the raw voxels do (grad +0.066).")


if __name__ == "__main__":
    main()
