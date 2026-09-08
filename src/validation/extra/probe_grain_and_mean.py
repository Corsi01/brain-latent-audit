#!/usr/bin/env python3
"""
probe_grain_and_mean.py — Two cheap tests that decide whether to expand the dataset,
                          and what the latent's ceiling is as an encoding target.

TEST 1 — MEAN STABILITY (split-half).
  Andre's hypothesis: per-subject centering helps the latent but not the voxels not
  because the voxels lack a baseline, but because 622 clips are too few to estimate a
  reliable per-subject mean for the high-dimensional voxels — whereas the latent, being
  low-dimensional, gets a good mean even from few samples.

  Test: estimate each subject's mean from a random half of the clips, and again from the
  other half. Correlate the two. If the voxel mean is UNSTABLE (low split-half r) while
  the latent mean is STABLE (high r), Andre is right — the voxel centering fails from
  undersampling, and expanding the dataset would fix it. If BOTH means are stable, the
  voxel mean is well-estimated and centering fails for a different reason (no structured
  baseline to remove) — expansion would not help, and the "removable baseline" reading
  of the latent stands.

  Also reported: effective dimensionality of the per-subject mean across subjects — a
  low-dim mean (few components) is the signature of a coarse representation.

TEST 2 — GRAIN (within-category retrieval).
  Is the latent COARSE (separates visual vs auditory vs linguistic) or FINE (separates
  content WITHIN a channel — one visual scene from another)? The main retrieval mixes
  channels, so it cannot tell. Restrict the candidate pool to a single content category
  (e.g. VISUAL_NOSPEECH) and retrieve within it: if the latent still identifies the right
  clip above chance, it carries fine within-channel content; if it collapses to chance,
  it only distinguishes channels.

  This bounds the ceiling of the latent as an encoding target: a coarse latent has a low
  ceiling no matter how good the tower.

Usage:
    python src/probe_grain_and_mean.py --lag 4
"""

import argparse, os, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # validation/ — retrieval.py lives there
from retrieval import summarize
from retrieval_variants import (load_raw, load_voxels_full,
                                N_BLOCKS, EMB_DIM, NUM_FRAMES, FLAT_DIM)


# ---------------------------------------------------------------------------
#  TEST 1 — mean stability
# ---------------------------------------------------------------------------

def split_half_mean_stability(vecs_by_subj, name, n_rep=20, seed=0):
    """
    vecs_by_subj: {subject: (n_clips, D) array}
    For each subject and each of n_rep random splits, compute the mean over each half
    and correlate the two halves. Report mean cosine and Pearson r across reps/subjects.
    """
    rng = np.random.default_rng(seed)
    cos_all, r_all = [], []
    for s, X in vecs_by_subj.items():
        n = len(X)
        for _ in range(n_rep):
            perm = rng.permutation(n)
            h1, h2 = perm[:n//2], perm[n//2:]
            m1, m2 = X[h1].mean(0), X[h2].mean(0)
            cos = float(m1 @ m2 / (np.linalg.norm(m1)*np.linalg.norm(m2) + 1e-12))
            r   = float(np.corrcoef(m1, m2)[0, 1])
            cos_all.append(cos); r_all.append(r)
    print(f"  {name:<28s} split-half mean:  cosine={np.mean(cos_all):.4f}  "
          f"pearson_r={np.mean(r_all):.4f}   (1.0 = perfectly stable)")
    return np.mean(cos_all)


def mean_effective_dim(vecs_by_subj, name):
    """
    Effective dimensionality of the SET of per-subject mean vectors.
    Stack the 4 subject means, do PCA, report participation ratio.
    Low = the means live in a tiny subspace = coarse baseline.
    """
    M = np.stack([X.mean(0) for X in vecs_by_subj.values()])   # (4, D)
    M = M - M.mean(0)
    # singular values of the 4×D centered matrix
    sv = np.linalg.svd(M, compute_uv=False)
    ev = sv**2
    if ev.sum() < 1e-12:
        print(f"  {name:<28s} mean effective dim: ~0 (means identical)")
        return
    pr = (ev.sum()**2) / (ev**2).sum()          # participation ratio
    frac = ev / ev.sum()
    print(f"  {name:<28s} across-subject mean: participation ratio={pr:.2f} "
          f"of {len(ev)}   var per comp: {', '.join(f'{f:.2f}' for f in frac)}")


# ---------------------------------------------------------------------------
#  TEST 2 — within-category retrieval
# ---------------------------------------------------------------------------

def within_category_retrieval(X, meta, category):
    """
    Retrieve only among clips of ONE content category. Candidate pool = that category's
    clips in subject B. Above chance ⇒ the latent separates content WITHIN the channel.
    """
    clips_in_cat = [c for c in meta if meta[c]["content_label"] == category]
    subs = sorted(X)
    R, N = [], []
    for A in subs:
        for B in subs:
            if A == B:
                continue
            shared = sorted(set(X[A]) & set(X[B]) & set(clips_in_cat))
            if len(shared) < 5:
                continue
            Q  = np.stack([X[A][c] for c in shared]).astype(np.float32)
            Cc = np.stack([X[B][c] for c in shared]).astype(np.float32)
            Q  /= np.clip(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8, None)
            Cc /= np.clip(np.linalg.norm(Cc, axis=1, keepdims=True), 1e-8, None)
            S = Q @ Cc.T
            idx = np.arange(len(shared))
            ts = S[idx, idx]
            R.append((S > ts[:, None]).sum(1))
            N.append(np.full(len(shared), len(shared)))
    if not R:
        return None
    ranks = np.concatenate(R); ncand = np.concatenate(N)
    o = summarize(ranks, ncand)
    pct = (ranks / np.clip(ncand - 1, 1, None)).mean()
    return {"n_pool": int(ncand[0]), "n_query": len(ranks),
            "pct": pct, "top1": o["top1"], "chance1": float(np.mean(1/ncand))}


# ---------------------------------------------------------------------------

def center_B(P):
    """per-patch 768-mean centering, returns flat vectors"""
    X = {}
    for s in P:
        allp = np.concatenate([v.reshape(-1, EMB_DIM) for v in P[s].values()], 0)
        m = allp.mean(0)
        X[s] = {c: (v.reshape(-1, EMB_DIM) - m).ravel().astype(np.float32)
                for c, v in P[s].items()}
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lag", type=int, required=True)
    args = ap.parse_args()
    code, cache = Path(os.environ["CS_CODE"]), Path(os.environ["CS_CACHE"])

    P, _, meta = load_raw(cache / f"latents_lag{args.lag}")
    subs = sorted(P)

    # ---- assemble arrays ----
    lat = {s: np.stack([P[s][c].reshape(-1) for c in sorted(P[s])]) for s in subs}
    clip_order = {s: sorted(P[s]) for s in subs}

    print("=" * 74)
    print("TEST 1 — MEAN STABILITY (split-half)")
    print("  Andre's H: voxel centering fails because 622 clips are too few for a")
    print("  reliable high-dim voxel mean. If so: voxel mean unstable, latent stable.")
    print("=" * 74)

    print("\n  Per-subject mean, estimated on random halves, correlated:")
    split_half_mean_stability(lat, "latent")

    V = load_voxels_full(code, cache, args.lag, set(meta))
    vox = {s: np.stack([V[s][c] for c in sorted(V[s])]) for s in V}
    split_half_mean_stability(vox, "voxels")

    print("\n  Effective dimensionality of the baseline across subjects:")
    mean_effective_dim(lat, "latent")
    mean_effective_dim(vox, "voxels")
    del V, vox

    print("\n  Reading:")
    print("    voxel r low, latent r high  → Andre right: undersampling, EXPAND helps")
    print("    both r high                 → voxel mean is fine; centering fails for")
    print("                                  another reason; expansion won't fix it")

    # ---- TEST 2 ----
    print("\n" + "=" * 74)
    print("TEST 2 — GRAIN (within-category retrieval)")
    print("  Coarse (separates channels) vs fine (separates content within a channel)?")
    print("  Above chance within a category ⇒ fine. At chance ⇒ coarse.")
    print("=" * 74)

    XB = center_B(P)
    cats = ["VISUAL_NOSPEECH", "SPEECH_STATIC", "SPEECH_DYNAMIC",
            "AUDIO_NOSPEECH", "LOW_DRIVE"]
    print(f"\n  {'category':<18s} {'pool':>5s} {'nq':>5s} {'pct':>7s} "
          f"{'top1':>7s} {'chance':>7s}  verdict")
    print("  " + "-" * 66)
    for cat in cats:
        r = within_category_retrieval(XB, meta, cat)
        if r is None:
            print(f"  {cat:<18s}  (too few clips)")
            continue
        fine = r["pct"] < 0.42
        verdict = "FINE (content within channel)" if fine else "coarse-ish / at chance"
        print(f"  {cat:<18s} {r['n_pool']:5d} {r['n_query']:5d} {r['pct']:7.3f} "
              f"{r['top1']:7.3f} {r['chance1']:7.3f}  {verdict}")

    print("\n  Note: pools are small (≤30), so these are underpowered — a pct clearly")
    print("  below 0.5 is meaningful, but a null here is 'not proven fine', not 'proven")
    print("  coarse'. This is the test that most benefits from expanding the clip set.")


if __name__ == "__main__":
    main()
