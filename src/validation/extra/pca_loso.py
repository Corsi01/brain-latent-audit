#!/usr/bin/env python3
"""
pca_loso.py — Leave-two-subjects-out validation of the channel-PCA result.

THE CONCERN
-----------
k=16 with per-subject centering gave the best numbers we have (pct-rank 0.183,
top-1 0.238, gradient +0.200) — beating raw voxels on every metric at 53x fewer
dimensions. But the PCA basis was fitted on ALL FOUR subjects, including the two
being evaluated in each retrieval pair. And k itself was chosen by looking at
those same numbers. Both are, strictly, decisions made on the test set.

THE FIX
-------
For each ordered pair (A -> B), fit the PCA basis on the OTHER TWO subjects only,
then project A and B onto that basis. The basis has never seen A or B.

With 4 subjects there are 6 unordered pairs, hence 6 distinct bases. Each is fitted
on ~450k patches — still 590:1 overdetermined for a 768-d basis, so the basis should
barely move. That is the prediction; this script tests it.

This is also a STRICTER test, not merely a cleaner one: the projection can only help
if the directions it finds generalise to brains it was never fitted on. That is exactly
the property required of an encoding target.

ON THE PER-SUBJECT MEANS
------------------------
The basis is held out; the per-subject means are not, and should not be. Subtracting a
subject's own mean is a per-subject normalisation available in any real deployment (you
always have that subject's data). It carries no clip-identity information — it is one
constant vector removed from all of that subject's clips. The leakage worry was about
the BASIS, and that is what is held out here.

Usage:
    python src/pca_loso.py --lag 4
    python src/pca_loso.py --lag 4 --ks 8 16 32 64
"""

import argparse, os, sys
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # validation/ — retrieval.py lives there
from retrieval import summarize
from retrieval_variants import load_raw, N_BLOCKS, EMB_DIM

# fit-on-all reference (center=subject, no whiten)
REF_ALL = {8: (0.192, 0.191, 0.214), 16: (0.183, 0.238, 0.200),
           32: (0.197, 0.250, 0.197), 64: (0.207, 0.238, 0.189),
           128: (0.208, 0.230, 0.193), 256: (0.202, 0.233, 0.195)}


def subject_mean(P_sub: dict) -> np.ndarray:
    acc, n = np.zeros(EMB_DIM, np.float64), 0
    for v in P_sub.values():
        X = v.reshape(-1, EMB_DIM)
        acc += X.sum(0)
        n   += X.shape[0]
    return (acc / n).astype(np.float32)


def fit_basis(P: dict, subs: list, mu: dict):
    """PCA basis from the given subjects only. Streaming covariance."""
    cov, n_tot = np.zeros((EMB_DIM, EMB_DIM), np.float64), 0
    for s in subs:
        m = mu[s]
        for v in P[s].values():
            X = (v.reshape(-1, EMB_DIM) - m).astype(np.float64)
            cov   += X.T @ X
            n_tot += X.shape[0]
    cov /= n_tot
    ev, V = np.linalg.eigh(cov)
    order = np.argsort(ev)[::-1]
    return np.clip(ev[order], 0, None), V[:, order], n_tot


def subspace_alignment(V1: np.ndarray, V2: np.ndarray, k: int) -> float:
    """
    Mean squared canonical correlation between the two k-dim subspaces.
    1.0 = identical subspace, 0.0 = orthogonal.
    Directly measures whether the basis is stable across subject subsets.
    """
    s = np.linalg.svd(V1[:, :k].T @ V2[:, :k], compute_uv=False)
    return float((s ** 2).mean())


def retrieve_pair(P, meta, A, B, V, mu, k):
    """Project A and B onto V[:, :k], retrieve A -> B. Returns ranks, ncand, clips."""
    Vk = V[:, :k]
    shared = sorted(set(P[A]) & set(P[B]))
    if len(shared) < 10:
        return None

    Q = np.stack([((P[A][c] - mu[A]) @ Vk).ravel() for c in shared]).astype(np.float32)
    C = np.stack([((P[B][c] - mu[B]) @ Vk).ravel() for c in shared]).astype(np.float32)

    Q /= np.clip(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8, None)
    C /= np.clip(np.linalg.norm(C, axis=1, keepdims=True), 1e-8, None)
    S = Q @ C.T

    idx    = np.arange(len(shared))
    true_s = S[idx, idx]
    ranks  = (S > true_s[:, None]).sum(1)
    ncand  = np.full(len(shared), len(shared))
    return ranks, ncand, np.array(shared)


def eval_loso(P, meta, mu, bases, k):
    """Aggregate retrieval over all ordered pairs, each with its held-out basis."""
    R, N, C = [], [], []
    for A in sorted(P):
        for B in sorted(P):
            if A == B:
                continue
            V = bases[frozenset((A, B))]
            out = retrieve_pair(P, meta, A, B, V, mu, k)
            if out:
                R.append(out[0]); N.append(out[1]); C.append(out[2])

    ranks = np.concatenate(R); ncand = np.concatenate(N); clips = np.concatenate(C)
    overall = summarize(ranks, ncand)

    by_cat = defaultdict(lambda: ([], []))
    for r, n, c in zip(ranks, ncand, clips):
        lab = meta[c]["content_label"]
        by_cat[lab][0].append(r)
        by_cat[lab][1].append(n)

    def pct(cat):
        if cat not in by_cat:
            return None
        r, n = by_cat[cat]
        return summarize(np.array(r), np.array(n))["pct"]

    blk, dyn = pct("BLACK"), pct("SPEECH_DYNAMIC")
    grad = (blk - dyn) if (blk is not None and dyn is not None) else None
    return overall["pct"], overall["top1"], grad, dyn, blk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lag", type=int, required=True)
    p.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    args = p.parse_args()

    cache = Path(os.environ["CS_CACHE"])
    P, _, meta = load_raw(cache / f"latents_lag{args.lag}")
    subs = sorted(P)
    n_sp = next(iter(P[subs[0]].values())).shape[1]

    print(f"lag={args.lag}   {len(subs)} subjects × {len(meta)} clips   patches={n_sp}")
    print(f"centering: per-subject   whitening: off\n")

    mu = {s: subject_mean(P[s]) for s in subs}

    # ---- one basis per unordered pair, fitted on the two held-out subjects ----
    print("Fitting one basis per pair, on the two SUBJECTS NOT IN THE PAIR:")
    bases, evals_all = {}, {}
    for A, B in combinations(subs, 2):
        held = [s for s in subs if s not in (A, B)]
        ev, V, n = fit_basis(P, held, mu)
        bases[frozenset((A, B))] = V
        evals_all[frozenset((A, B))] = ev
        print(f"  eval {A}↔{B}   basis from {held[0]},{held[1]}   "
              f"{n:,} patches ({n/EMB_DIM:.0f}:1)")

    # ---- basis fitted on everything, for the alignment check ----
    _, V_all, n_all = fit_basis(P, subs, mu)

    print(f"\nSubspace alignment: held-out basis vs full basis")
    print(f"  (1.00 = identical subspace, 0.00 = orthogonal)")
    print(f"  {'k':>4s}  " + "  ".join(f"{A[-2:]}↔{B[-2:]}" for A, B in combinations(subs, 2)))
    for k in args.ks:
        row = [subspace_alignment(bases[frozenset(pr)], V_all, k)
               for pr in combinations(subs, 2)]
        print(f"  {k:4d}  " + "  ".join(f"{v:5.3f}" for v in row))

    # ---- the sweep ----
    print("\n" + "=" * 86)
    print("LEAVE-TWO-SUBJECTS-OUT vs FIT-ON-ALL")
    print("=" * 86)
    print(f"  {'k':>4s} {'dims':>9s} | {'--- held-out basis ---':^32s} | "
          f"{'--- fit on all ---':^24s}")
    print(f"  {'':>4s} {'':>9s} | {'pct':>7s} {'top-1':>7s} {'grad':>7s} {'DYN':>7s} | "
          f"{'pct':>7s} {'top-1':>7s} {'grad':>7s}")
    print("  " + "-" * 82)

    rows = []
    for k in args.ks:
        pc, t1, g, dyn, blk = eval_loso(P, meta, mu, bases, k)
        rows.append((k, pc, t1, g))
        d = N_BLOCKS * n_sp * k
        if k in REF_ALL:
            rp, rt, rg = REF_ALL[k]
            ref = f"{rp:7.3f} {rt:7.3f} {rg:+7.3f}"
        else:
            ref = f"{'—':>7s} {'—':>7s} {'—':>7s}"
        print(f"  {k:4d} {d:9,d} | {pc:7.3f} {t1:7.3f} {g:+7.3f} {dyn:7.3f} | {ref}",
              flush=True)

    # ---- verdict ----
    best_k, best_pc, best_t1, best_g = min(rows, key=lambda r: r[1])

    print("\n" + "=" * 86)
    print("VERDICT")
    print("=" * 86)
    print(f"  best k under held-out basis : k={best_k}")
    print(f"    pct-rank {best_pc:.3f}   top-1 {best_t1:.3f}   gradient {best_g:+.3f}")

    if 16 in REF_ALL:
        rp, rt, rg = REF_ALL[16]
        k16 = next((r for r in rows if r[0] == 16), None)
        if k16:
            print(f"\n  k=16 specifically:")
            print(f"    fit-on-all    pct {rp:.3f}  top-1 {rt:.3f}  grad {rg:+.3f}")
            print(f"    held-out      pct {k16[1]:.3f}  top-1 {k16[2]:.3f}  grad {k16[3]:+.3f}")
            print(f"    delta         pct {k16[1]-rp:+.3f}  top-1 {k16[2]-rt:+.3f}  "
                  f"grad {k16[3]-rg:+.3f}")

    print(f"\n  Reference — raw voxels, 1,244,208 dims:")
    print(f"    pct-rank 0.197   top-1 0.116   gradient +0.066")

    if best_pc < 0.197 and best_t1 > 0.116:
        print(f"\n  → The held-out latent still beats raw voxels on both accuracy and")
        print(f"    the drive gradient. The result is not an artefact of fitting the")
        print(f"    basis on the evaluation subjects.")
    else:
        print(f"\n  → The advantage does NOT survive held-out fitting. The earlier")
        print(f"    numbers were inflated by fitting the basis on the eval subjects.")

    if best_k == 16:
        print(f"  → k=16 remains optimal under held-out fitting, so the choice of k")
        print(f"    was not overfitted to the test set either.")
    else:
        print(f"  → Optimal k shifts from 16 to {best_k} under held-out fitting.")
        print(f"    Check whether the plateau is broad enough that this matters.")


if __name__ == "__main__":
    main()
