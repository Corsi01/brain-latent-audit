#!/usr/bin/env python3
"""
baseline_probe.py — Discriminate the two hypotheses for WHY per-subject centering
                    helps the latent (+40% top-1) but not the voxels (~0%).

H1  CortexMAE reintroduces a per-subject SCALE difference. If so, the fix is
    standardisation (subtract mean AND divide by std), and centering alone would
    be incomplete. Additive centering recovering most of the gain argues AGAINST H1.

H2  The latent's per-subject MEAN vector encodes the intrinsic/baseline brain state.
    Removing it isolates the evoked deviation. If so:
      - the mean vector should itself be rich and structured (not near-zero, not a
        scalar offset),
      - it should be SIMILAR across subjects (baseline cortex is broadly shared),
      - removing it should hurt no-stimulus clips (BLACK) more than evoked ones,
      - and standardisation should add little beyond centering.

TESTS
  A. Magnitude of the per-subject mean vs the residual. If the mean dominates,
     it is a large structured component, not numerical noise.
  B. Cosine similarity between subjects' mean vectors. High → shared baseline (H2).
  C. Centering vs standardisation vs nothing, on latent AND voxels. If standardise
     ≈ center on the latent, the effect is additive (H2), not multiplicative (H1).
  D. Is the removed mean itself stimulus-decodable? Project each clip onto the
     subject-mean direction and see if that scalar carries drive information. If the
     mean direction is pure baseline, it should NOT track stimulus drive.

Usage:
    python src/baseline_probe.py --lag 4
"""

import argparse, os, sys, json
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "extra"))  # retrieval_variants.py lives there
from retrieval import summarize
from retrieval_variants import (load_raw, load_voxels_full,
                                N_BLOCKS, EMB_DIM, NUM_FRAMES, FLAT_DIM)


def flatten(D):
    return {s: {c: v.reshape(-1).astype(np.float32) for c, v in D[s].items()} for s in D}


def stats(X):
    return {s: (np.mean(np.stack(list(X[s].values())), 0),
                np.std(np.stack(list(X[s].values())), 0)) for s in X}


def transform(X, mode, st):
    """mode: raw | center | standardize"""
    out = {}
    for s in X:
        mu, sd = st[s]
        out[s] = {}
        for c, v in X[s].items():
            if mode == "raw":
                out[s][c] = v
            elif mode == "center":
                out[s][c] = v - mu
            else:
                out[s][c] = (v - mu) / np.clip(sd, 1e-6, None)
    return out


def cosine_eval(X, meta):
    subs = sorted(X); R, N, C = [], [], []
    for A in subs:
        for B in subs:
            if A == B: continue
            shared = sorted(set(X[A]) & set(X[B]))
            if len(shared) < 10: continue
            Q  = np.stack([X[A][c] for c in shared]).astype(np.float32)
            Cc = np.stack([X[B][c] for c in shared]).astype(np.float32)
            Q  /= np.clip(np.linalg.norm(Q, 1, keepdims=True).T, 1e-8, None) if False else \
                  np.clip(np.linalg.norm(Q, axis=1, keepdims=True), 1e-8, None)
            Cc /= np.clip(np.linalg.norm(Cc, axis=1, keepdims=True), 1e-8, None)
            S = Q @ Cc.T
            idx = np.arange(len(shared)); ts = S[idx, idx]
            R.append((S > ts[:, None]).sum(1)); N.append(np.full(len(shared), len(shared)))
            C.append(np.array(shared))
    return np.concatenate(R), np.concatenate(N), np.concatenate(C)


def quick(X, meta, label):
    r, n, c = cosine_eval(X, meta)
    o = summarize(r, n)
    pct = (r / np.clip(n - 1, 1, None)).mean()
    by = defaultdict(lambda: ([], []))
    for rr, nn, cc in zip(r, n, c):
        by[meta[cc]["content_label"]][0].append(rr); by[meta[cc]["content_label"]][1].append(nn)
    def p(cat):
        if cat not in by: return None
        a, b = by[cat]; return (np.array(a) / np.clip(np.array(b)-1,1,None)).mean()
    blk, dyn = p("BLACK"), p("SPEECH_DYNAMIC")
    g = (blk - dyn) if (blk and dyn) else None
    print(f"    {label:<28s} pct={pct:.3f} top1={o['top1']:.3f} "
          f"grad={g:+.3f}" if g else f"    {label:<28s} pct={pct:.3f} top1={o['top1']:.3f}")
    return pct, o["top1"], blk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lag", type=int, required=True)
    args = ap.parse_args()
    code, cache = Path(os.environ["CS_CODE"]), Path(os.environ["CS_CACHE"])

    P, _, meta = load_raw(cache / f"latents_lag{args.lag}")
    Xlat = flatten(P)
    subs = sorted(Xlat)
    st_lat = stats(Xlat)

    # ---- A. magnitude of the mean vs residual ----
    print("=" * 72)
    print("A. Is the per-subject mean a large structured component?")
    print("=" * 72)
    for s in subs:
        mu, _ = st_lat[s]
        vecs = np.stack(list(Xlat[s].values()))
        resid = vecs - mu
        mnorm = np.linalg.norm(mu)
        rnorm = np.linalg.norm(resid, axis=1).mean()
        print(f"    {s}: ||mean||={mnorm:8.2f}   mean||residual||={rnorm:8.2f}   "
              f"ratio={mnorm/rnorm:.2f}")
    print("    (ratio >> 1 → the mean dominates: a large structured vector, not noise)")

    # ---- B. are the subject means similar to each other? ----
    print("\n" + "=" * 72)
    print("B. Cosine similarity between subjects' mean vectors")
    print("=" * 72)
    means = {s: st_lat[s][0] for s in subs}
    for a, b in combinations(subs, 2):
        ca = means[a] / np.linalg.norm(means[a])
        cb = means[b] / np.linalg.norm(means[b])
        print(f"    {a} · {b} = {float(ca @ cb):.3f}")
    print("    (high → a shared baseline state across subjects, supporting H2)")

    # ---- C. center vs standardize vs raw, latent and voxels ----
    print("\n" + "=" * 72)
    print("C. raw vs center vs standardize")
    print("   H1 (scale): standardize should beat center.  H2 (baseline): center ≈ standardize.")
    print("=" * 72)
    print("  LATENT:")
    for mode in ("raw", "center", "standardize"):
        quick(transform(Xlat, mode, st_lat), meta, mode)

    print("  VOXELS (unpooled):")
    V = load_voxels_full(code, cache, args.lag, set(meta))
    Xvox = {s: {c: V[s][c] for c in V[s]} for s in V}
    st_vox = stats(Xvox)
    for mode in ("raw", "center", "standardize"):
        quick(transform(Xvox, mode, st_vox), meta, mode)
    del V, Xvox

    # ---- D. does the removed mean direction carry stimulus drive? ----
    print("\n" + "=" * 72)
    print("D. Does the subject-mean DIRECTION track stimulus drive?")
    print("   H2 predicts NO — the mean is baseline, orthogonal-ish to evoked signal.")
    print("=" * 72)
    with open(code / "manifest" / "clip_manifest.json") as f:
        man = {c["clip_id"]: c for c in json.load(f)["clips"]}

    # for each clip, projection of (centered) latent onto the subject-mean unit vector,
    # averaged over subjects; correlate with speech+motion
    proj_by_clip = defaultdict(list)
    for s in subs:
        mu = means[s]; u = mu / np.linalg.norm(mu)
        for c, v in Xlat[s].items():
            proj_by_clip[c].append(float(v @ u))
    y_proj = {c: np.mean(vs) for c, vs in proj_by_clip.items()}

    rows, drive = [], []
    for c, pv in y_proj.items():
        m = man.get(c)
        if m is None: continue
        dv = m["drive_vector"]
        if dv.get("speech_coverage") is None or dv.get("frame_diff") is None: continue
        rows.append(pv); drive.append([dv["speech_coverage"], dv["frame_diff"]])
    rows = np.array(rows); drive = np.array(drive)
    for i, nm in enumerate(["speech", "motion"]):
        r = np.corrcoef(rows, drive[:, i])[0, 1]
        print(f"    corr(mean-projection, {nm}) = {r:+.3f}")
    print("    (near zero → the removed mean is baseline, not stimulus — supports H2)")


if __name__ == "__main__":
    main()
