#!/usr/bin/env python3
"""
fair_comparison.py — Level the playing field: SAME treatment (per-subject centering,
                     NO PCA on either side) for latent and voxels. Then stress-test
                     the drive gradient two more ways.

WHY NO PCA HERE
---------------
At k=768 the channel-PCA is a pure rotation (centre + rotate, no truncation), and cosine
is invariant to rotation. So "per-subject centered, k=768" already IS "per-subject centered,
no PCA". Applying a real (truncated) PCA to the voxels is not comparable: with only 622
windows the voxel sample-PCA has rank <=621, and matching either dimensionality (23,296) or
variance to the latent's channel-PCA is ill-posed — the latent keeps 364 spatial tokens that
the flattened voxels do not have. So we drop PCA entirely and compare like for like:
per-subject centering, full dimension, both sides.

THREE EXPERIMENTS
-----------------
1. FAIR GRADIENT.  Latent vs voxels (pooled and unpooled), all per-subject centered, full dim.
   Does per-subject centering raise the VOXEL drive gradient up to the latent's? If yes, the
   "voxels identify on non-evoked structure" claim was a preprocessing artefact. If no, it holds.

2. REGRESSION GRADIENT.  The categorical gradient leans on the black-screen category, which is
   audio-contaminated. The continuous regression on speech + motion (within-film demeaned) is
   the defensible version. Reported for latent and voxels.

3. TEMPORAL SHUFFLE.  Permute the latent's 4 temporal blocks within each clip. If retrieval
   holds, block ORDER is inert and pooling over T is justified; if it drops, order matters.

Usage:
    python src/fair_comparison.py --lag 4
"""

import argparse, os, sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "extra"))  # retrieval_variants.py lives there
from retrieval import summarize
from retrieval_variants import load_raw, load_voxels_full, N_BLOCKS, EMB_DIM, NUM_FRAMES, FLAT_DIM

PRED = ["luminance", "frame_diff", "rms", "speech_coverage"]


# ---------------------------------------------------------------------------
#  Retrieval + reporting
# ---------------------------------------------------------------------------

def cosine_eval(X, meta):
    subs = sorted(X)
    R, N, C = [], [], []
    for A in subs:
        for B in subs:
            if A == B:
                continue
            shared = sorted(set(X[A]) & set(X[B]))
            if len(shared) < 10:
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
            C.append(np.array(shared))
    return np.concatenate(R), np.concatenate(N), np.concatenate(C)


def _pct_both(ranks, ncand):
    """Two percentile conventions, to expose any normalization mismatch."""
    p_nm1 = (ranks / np.clip(ncand - 1, 1, None)).mean()   # rank/(N-1)
    p_n   = (ranks / np.clip(ncand,     1, None)).mean()   # rank/N
    return p_nm1, p_n


def report(X, meta, label):
    ranks, ncand, clips = cosine_eval(X, meta)
    o = summarize(ranks, ncand)
    by = defaultdict(lambda: ([], []))
    for r, n, c in zip(ranks, ncand, clips):
        by[meta[c]["content_label"]][0].append(r)
        by[meta[c]["content_label"]][1].append(n)
    def pct(cat):
        if cat not in by: return None
        r, n = by[cat]; return summarize(np.array(r), np.array(n))["pct"]
    blk, dyn = pct("BLACK"), pct("SPEECH_DYNAMIC")
    grad = (blk - dyn) if (blk is not None and dyn is not None) else None
    gs = f"{grad:+.3f}" if grad is not None else "   —"
    bs = f"{blk:.3f}"  if blk  is not None else "  —"
    ds = f"{dyn:.3f}"  if dyn  is not None else "  —"
    p1, p2 = _pct_both(ranks, ncand)
    print(f"  {label:<38s} pct={p1:.3f}  top1={o['top1']:.3f}  "
          f"grad={gs}  (BLACK={bs} DYN={ds})")
    return {"pct": p1, "top1": o["top1"], "grad": grad, "black": blk, "dyn": dyn}


# ---------------------------------------------------------------------------
#  Per-subject centering
# ---------------------------------------------------------------------------

def flatten(D):
    return {s: {c: v.reshape(-1).astype(np.float32) for c, v in D[s].items()} for s in D}


def subject_center(X):
    Xc = {}
    for s in X:
        M = np.mean(np.stack(list(X[s].values())), axis=0)
        Xc[s] = {c: v - M for c, v in X[s].items()}
    return Xc


# ---------------------------------------------------------------------------
#  Regression gradient
# ---------------------------------------------------------------------------

def regression_gradient(X, meta, code, label):
    ranks, ncand, clips = cosine_eval(X, meta)
    pct = ranks / np.clip(ncand - 1, 1, None)
    per = defaultdict(list)
    for p, c in zip(pct, clips):
        per[c].append(p)
    cp = {c: float(np.mean(v)) for c, v in per.items()}

    with open(code / "manifest" / "clip_manifest.json") as f:
        man = {c["clip_id"]: c for c in json.load(f)["clips"]}

    rows, y, films = [], [], []
    for cid, val in cp.items():
        c = man.get(cid)
        if c is None: continue
        dv = c["drive_vector"]
        if any(dv.get(k) is None for k in PRED): continue
        rows.append([dv[k] for k in PRED]); y.append(val); films.append(c["film_id"])
    Xr = np.array(rows); y = np.array(y)

    for f in set(films):
        m = np.array([ff == f for ff in films])
        if m.sum() > 1:
            y[m] -= y[m].mean(); Xr[m] -= Xr[m].mean(0)
    Xz = (Xr - Xr.mean(0)) / np.clip(Xr.std(0), 1e-9, None)

    def partial(j):
        others = np.delete(Xz, j, 1)
        A = np.column_stack([np.ones(len(Xz)), others])
        rx = Xz[:, j] - A @ np.linalg.lstsq(A, Xz[:, j], rcond=None)[0]
        ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        return float(np.corrcoef(rx, ry)[0, 1])

    # permutation p for the two channels of interest
    def perm_p(j, r_obs, n=2000):
        rng = np.random.default_rng(0)
        cnt = 0
        others = np.delete(Xz, j, 1)
        A = np.column_stack([np.ones(len(Xz)), others])
        ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        rx = Xz[:, j] - A @ np.linalg.lstsq(A, Xz[:, j], rcond=None)[0]
        for _ in range(n):
            rp = rng.permutation(ry)
            if abs(np.corrcoef(rx, rp)[0, 1]) >= abs(r_obs):
                cnt += 1
        return cnt / n

    sp, mo = partial(3), partial(1)
    print(f"  {label:<24s} speech r={sp:+.3f} (p={perm_p(3, sp):.3f})   "
          f"motion r={mo:+.3f} (p={perm_p(1, mo):.3f})")


# ---------------------------------------------------------------------------
#  Temporal shuffle
# ---------------------------------------------------------------------------

def temporal_shuffle(P, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for s in P:
        out[s] = {c: v[rng.permutation(N_BLOCKS)] for c, v in P[s].items()}
    return out


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lag", type=int, required=True)
    ap.add_argument("--shuffle-seeds", type=int, default=3)
    args = ap.parse_args()

    code  = Path(os.environ["CS_CODE"])
    cache = Path(os.environ["CS_CACHE"])

    P, _, meta = load_raw(cache / f"latents_lag{args.lag}")
    subs = sorted(P); n_sp = next(iter(P[subs[0]].values())).shape[1]
    print(f"lag={args.lag}  {len(subs)} subj × {len(meta)} clips  patches={n_sp}\n")

    # =====================================================================
    print("=" * 78)
    print("1. FAIR GRADIENT — per-subject centered, full dimension, NO PCA on either side")
    print("=" * 78)

    Xlat = subject_center(flatten({s: {c: P[s][c] for c in P[s]} for s in P}))
    r_lat = report(Xlat, meta, f"latent centered ({N_BLOCKS}×{n_sp}×{EMB_DIM})")

    print()
    V = load_voxels_full(code, cache, args.lag, set(meta))
    Xv_full = subject_center({s: {c: V[s][c] for c in V[s]} for s in V})
    r_vox_full = report(Xv_full, meta, f"voxels UNPOOLED centered ({NUM_FRAMES}×{FLAT_DIM})")
    del Xv_full

    # pooled 4×4
    Xv_pool = {}
    for s in V:
        Xv_pool[s] = {}
        for c, v in V[s].items():
            w = v.reshape(NUM_FRAMES, FLAT_DIM).reshape(N_BLOCKS, 4, FLAT_DIM).mean(1)
            Xv_pool[s][c] = w.reshape(-1)
    Xv_pool = subject_center(Xv_pool)
    r_vox_pool = report(Xv_pool, meta, f"voxels POOLED 4×4 centered ({N_BLOCKS}×{FLAT_DIM})")
    del V, Xv_pool

    print("\n  VERDICT on the main claim (all per-subject centered, like for like):")
    lg = r_lat["grad"]
    for name, r in [("voxels unpooled", r_vox_full), ("voxels pooled", r_vox_pool)]:
        vg = r["grad"]
        if vg is None or lg is None:
            continue
        ratio = vg / lg
        if ratio < 0.5:
            tag = "CLAIM HOLDS — voxel gradient stays far below latent's"
        elif ratio < 0.8:
            tag = "CLAIM SURVIVES, weakened — report both"
        else:
            tag = "CLAIM FAILS — centering closed the gap"
        print(f"    latent grad {lg:+.3f}  vs  {name} grad {vg:+.3f}  "
              f"(ratio {ratio:.2f}) → {tag}")

    # =====================================================================
    print("\n" + "=" * 78)
    print("2. REGRESSION GRADIENT — continuous drive, within-film demeaned")
    print("   (does not depend on the audio-contaminated black-screen category)")
    print("=" * 78)
    regression_gradient(Xlat, meta, code, "latent centered")
    V = load_voxels_full(code, cache, args.lag, set(meta))
    Xv_full = subject_center({s: {c: V[s][c] for c in V[s]} for s in V})
    regression_gradient(Xv_full, meta, code, "voxels unpooled centered")
    del V, Xv_full
    print("  (negative r = more of that feature → better retrieval)")

    # =====================================================================
    print("\n" + "=" * 78)
    print("3. TEMPORAL SHUFFLE — do the 4 blocks' ORDER carry signal?")
    print("=" * 78)
    r_ord = report(Xlat, meta, "blocks in order")
    ds = []
    for seed in range(args.shuffle_seeds):
        Xsh = subject_center(flatten(temporal_shuffle(P, seed=seed)))
        r_sh = report(Xsh, meta, f"blocks shuffled (seed {seed})")
        ds.append(r_sh["pct"] - r_ord["pct"])
    d = float(np.mean(ds))
    print(f"\n  Δ pct-rank (shuffled − ordered), mean of {args.shuffle_seeds} seeds = {d:+.3f}")
    if abs(d) < 0.01:
        print("    → order is inert. Pooling / reducing over the 4 temporal blocks is justified.")
    else:
        print("    → order carries signal. The 4 temporal blocks must be kept distinct.")


if __name__ == "__main__":
    main()
