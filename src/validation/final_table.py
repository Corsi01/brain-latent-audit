#
# !/usr/bin/env python3
"""
final_table.py — One coherent table of every representation, every centering,
                 every k, all in the SAME pct-rank convention (rank/(N-1), N=155).

Resolves the confusion that produced two different pct-rank values for what looked
like the same setting. They were NOT the same setting — they were two different
per-subject centerings:

  A  per-clip-vector centering: subtract a 279k-dim mean (one value per (patch,channel)).
     Removes the subject's baseline AND their mean cortical topography. Stronger, but the
     mean is estimated from only ~155 clips (155 samples for a 279k vector — noisy), and it
     has no clean 1x1-conv analogue for the encoding head (it is a spatial subtraction).

  B  per-patch-channel centering: subtract a 768-dim mean (one value per channel, broadcast
     over all 364 patches). Removes the shared baseline state only. Estimated from 905k
     patches (robust), and it IS a 1x1-conv-compatible operation.

Channel-PCA (768 -> k) is applied AFTER centering, on the channel axis, for both.

Everything uses retrieval.evaluate (N=155 candidates) and rank/(N-1), so every number
in the output — and in the report — is comparable.

Usage:
    python src/final_table.py --lag 4
    python src/final_table.py --lag 4 --loso        # add leave-two-subjects-out columns
"""

import argparse, os, sys, json
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "extra"))  # retrieval_variants.py lives there
from retrieval import evaluate, summarize
from retrieval_variants import (load_raw, load_voxels_full,
                                N_BLOCKS, EMB_DIM, NUM_FRAMES, FLAT_DIM)


def metrics(X, meta):
    r, n, c = evaluate(X, meta)
    o = summarize(r, n)
    by = defaultdict(lambda: ([], []))
    for rr, nn, cc in zip(r, n, c):
        by[meta[cc]["content_label"]][0].append(rr)
        by[meta[cc]["content_label"]][1].append(nn)
    def p(cat):
        if cat not in by: return None
        a, b = by[cat]; return (np.array(a) / np.clip(np.array(b) - 1, 1, None)).mean()
    blk, dyn = p("BLACK"), p("SPEECH_DYNAMIC")
    g = (blk - dyn) if (blk is not None and dyn is not None) else None
    return o["pct"], o["top1"], g


def row(label, X, meta, dim):
    pc, t1, g = metrics(X, meta)
    gs = f"{g:+.3f}" if g is not None else "   —  "
    print(f"  {label:<34s} {dim:>12,d}  {pc:.3f}  {t1:.3f}  {gs}")
    return pc, t1, g


# ---- centerings ----
def center_B(P):  # per-patch-channel, 768-dim mean per subject
    X = {}
    for s in P:
        allp = np.concatenate([v.reshape(-1, EMB_DIM) for v in P[s].values()], 0)
        m = allp.mean(0)
        X[s] = {c: (v.reshape(-1, EMB_DIM) - m) for c, v in P[s].items()}   # (4,nsp,768)
    return X

def center_A(P):  # per-clip-vector, full-dim mean per subject
    X = {}
    for s in P:
        M = np.mean(np.stack([v.reshape(-1) for v in P[s].values()]), 0)
        X[s] = {c: (v.reshape(-1) - M) for c, v in P[s].items()}            # flat 279k
    return X


def channel_pca_basis(Xc):
    """PCA over 768 channels from centered (4,nsp,768) arrays. Streaming cov."""
    cov, n = np.zeros((EMB_DIM, EMB_DIM)), 0
    for s in Xc:
        for v in Xc[s].values():
            M = v.reshape(-1, EMB_DIM)
            cov += M.T @ M; n += M.shape[0]
    cov /= n
    ev, V = np.linalg.eigh(cov)
    o = np.argsort(ev)[::-1]
    return V[:, o]


def project_B(Xc, V, k):
    return {s: {c: (v.reshape(-1, EMB_DIM) @ V[:, :k]).ravel().astype(np.float32)
                for c, v in Xc[s].items()} for s in Xc}


def flat(P):
    return {s: {c: v.reshape(-1).astype(np.float32) for c, v in P[s].items()} for s in P}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lag", type=int, required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    args = ap.parse_args()
    code, cache = Path(os.environ["CS_CODE"]), Path(os.environ["CS_CACHE"])

    P, C, meta = load_raw(cache / f"latents_lag{args.lag}")
    nsp = next(iter(P[sorted(P)[0]].values())).shape[1]
    print(f"lag={args.lag}  {len(P)} subj × {len(meta)} clips  patches={nsp}")
    print(f"pct convention: rank/(N-1), N=155 candidates\n")

    hdr = f"  {'representation':<34s} {'dims':>12s}  {'pct':>5s}  {'top1':>5s}  {'grad':>6s}"
    print(hdr); print("  " + "-" * 68)

    # ---- baselines ----
    row("latent raw (no centering)", flat(P), meta, N_BLOCKS*nsp*EMB_DIM)

    # CLS
    if all(C[s][c] is not None for s in C for c in C[s]):
        Xc = {s: {c: C[s][c] for c in C[s]} for s in C}
        row("CLS token", Xc, meta, EMB_DIM)

    # spatial pooled
    Xsp = {s: {c: P[s][c].mean(1).ravel().astype(np.float32) for c in P[s]} for s in P}
    row("latent spatial-pooled", Xsp, meta, N_BLOCKS*EMB_DIM)

    print("  " + "-" * 68)
    print("  CENTERING B (per-patch, 768-mean; 1×1-conv compatible):")
    XB = center_B(P)
    row("  B, full dim", flat(XB), meta, N_BLOCKS*nsp*EMB_DIM)
    VB = channel_pca_basis(XB)
    for k in args.ks:
        row(f"  B + PCA k={k}", project_B(XB, VB, k), meta, N_BLOCKS*nsp*k)

    print("  " + "-" * 68)
    print("  CENTERING A (per-clip-vector, 279k-mean; stronger, no conv analogue):")
    XA = center_A(P)
    row("  A, full dim", XA, meta, N_BLOCKS*nsp*EMB_DIM)
    # A + channel PCA: reshape A back to (4,nsp,768) to PCA the channel axis
    XA_shaped = {s: {c: XA[s][c].reshape(N_BLOCKS, nsp, EMB_DIM) for c in XA[s]} for s in XA}
    VA = channel_pca_basis(XA_shaped)
    for k in args.ks:
        row(f"  A + PCA k={k}", project_B(XA_shaped, VA, k), meta, N_BLOCKS*nsp*k)

    # ---- voxels ----
    print("  " + "-" * 68)
    print("  VOXELS:")
    V = load_voxels_full(code, cache, args.lag, set(meta))
    Xv = {s: {c: V[s][c] for c in V[s]} for s in V}
    row("  raw, 16 frames", Xv, meta, NUM_FRAMES*FLAT_DIM)
    # per-subject centered (vector centering = A-style, the only option for voxels)
    Xvc = {}
    for s in Xv:
        M = np.mean(np.stack(list(Xv[s].values())), 0)
        Xvc[s] = {c: (v - M).astype(np.float32) for c, v in Xv[s].items()}
    row("  centered, 16 frames", Xvc, meta, NUM_FRAMES*FLAT_DIM)
    del V, Xv, Xvc

    print("\n  Note: voxel centering is vector-style (like A) — voxels have no channel axis.")


if __name__ == "__main__":
    main()
