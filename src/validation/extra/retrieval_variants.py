#!/usr/bin/env python3
"""
retrieval_variants.py — Three representations the previous runs left open.

  1. VOXEL FULL   — raw flat map, all 16 frames, NO temporal binning (1,244,208 d)
                    The earlier voxel baseline binned 16 frames → 4 blocks to match
                    the latent's temporal structure. But binning is destructive for
                    voxels in a way it is not for the latent (the encoder attends over
                    its 4 frames; averaging them just throws the dynamics away).
                    So the old baseline HANDICAPPED the voxels — and they still won on
                    pct-rank. This measures the unhandicapped number.

  2. CLS          — the class token (768 d). The one place CortexMAE actually does
                    compress: a single global summary vector. patch_embeds does NOT
                    compress (1.24M in → 1.12M out, ~10% reduction), so if we want to
                    know what a compact brain code buys us, this is it.

  3. CHANNEL PCA  — PCA over the 768 feature channels, patches as samples.
                    ~905k samples to estimate a 768-d basis: overdetermined 1180:1.
                    Keeps all 364 patches intact (spatial structure is the signal —
                    unpooled beat pooled 2:1 on top-1) and compresses only the channel
                    axis. Architecturally this is a 1×1 conv: it drops straight into
                    a readout head.

Usage:
    python src/retrieval_variants.py --lag 4 --k 250
    python src/retrieval_variants.py --lag 4 --k 250 --center subject --whiten
"""

import argparse, json, os, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # validation/ — retrieval.py lives there
from retrieval import evaluate, summarize, print_row

CATEGORY_ORDER = [
    "BLACK",
    "LOW_DRIVE",
    "AUDIO_NOSPEECH",
    "VISUAL_NOSPEECH",
    "SPEECH_STATIC",
    "SPEECH_DYNAMIC",
]

N_BLOCKS   = 4
EMB_DIM    = 768
NUM_FRAMES = 16
FLAT_DIM   = 77763


# =====================================================================
#  Loading
# =====================================================================

def load_raw(lat_dir: Path):
    """Load patch_embeds (4, n_sp, 768) and cls (768,) per (subject, clip)."""
    P, C, meta = defaultdict(dict), defaultdict(dict), {}
    files = sorted(lat_dir.glob("*.npz"))
    if not files:
        sys.exit(f"No latents in {lat_dir}")

    for f in files:
        d = np.load(f, allow_pickle=True)
        sub, cid = str(d["subject"]), str(d["clip_id"])
        p = d["patch_embeds"]
        n_sp = p.shape[0] // N_BLOCKS
        P[sub][cid] = p.reshape(N_BLOCKS, n_sp, EMB_DIM).astype(np.float32)

        cls = d["cls_embeds"]
        C[sub][cid] = cls.reshape(-1).astype(np.float32) if cls.size else None

        if cid not in meta:
            meta[cid] = {
                "film_id": str(d["film_id"]), "run_id": str(d["run_id"]),
                "content_label": str(d["content_label"]), "start_s": int(d["start_s"]),
            }
    return dict(P), dict(C), meta


def load_voxels_full(code: Path, cache: Path, lag: int, clip_ids: set):
    """Raw flat map, all 16 frames, NO binning → 16 × 77763 = 1,244,208 d."""
    with open(code / "manifest" / "clip_manifest.json") as f:
        clips = {c["clip_id"]: c for c in json.load(f)["clips"]}

    fdir = cache / "fmri_flat_1hz"
    V = defaultdict(dict)

    by_run = defaultdict(list)
    for cid in clip_ids:
        c = clips.get(cid)
        if c is not None:
            by_run[c["run_id"]].append(c)

    subs = sorted({p.stem.split("_", 1)[0] for p in fdir.glob("*.npy")})
    for sub in subs:
        for run_id, run_clips in by_run.items():
            npy = fdir / f"{sub}_{run_id}.npy"
            if not npy.exists():
                continue
            fmri = np.load(npy).astype(np.float32)
            for c in run_clips:
                t0 = c["start_s"] + lag          # same lag convention throughout
                t1 = t0 + NUM_FRAMES
                if t1 > fmri.shape[0]:
                    continue
                V[sub][c["clip_id"]] = fmri[t0:t1].ravel().astype(np.float32)
            del fmri
    return dict(V)


# =====================================================================
#  Channel PCA
# =====================================================================

def fit_channel_pca(P: dict, center: str = "global"):
    """
    PCA over the 768 channels, treating every (clip, block, patch) as one sample.
    Returns (eigvals desc, eigvecs, mean_fn).

    Accumulates the 768×768 covariance in one streaming pass — never materialises
    the ~905k × 768 stack.
    """
    subs = sorted(P)

    # --- means ---
    if center == "subject":
        mu = {}
        for s in subs:
            acc, n = np.zeros(EMB_DIM, np.float64), 0
            for v in P[s].values():
                acc += v.reshape(-1, EMB_DIM).sum(0)
                n   += v.reshape(-1, EMB_DIM).shape[0]
            mu[s] = (acc / n).astype(np.float32)
        mean_fn = lambda s: mu[s]
        print(f"  centering: per-subject (removes each subject's constant offset)")
    else:
        acc, n = np.zeros(EMB_DIM, np.float64), 0
        for s in subs:
            for v in P[s].values():
                acc += v.reshape(-1, EMB_DIM).sum(0)
                n   += v.reshape(-1, EMB_DIM).shape[0]
        g = (acc / n).astype(np.float32)
        mean_fn = lambda s: g
        print(f"  centering: global")

    # --- covariance (streaming) ---
    cov, n_tot = np.zeros((EMB_DIM, EMB_DIM), np.float64), 0
    for s in subs:
        m = mean_fn(s)
        for v in P[s].values():
            X = v.reshape(-1, EMB_DIM) - m
            cov   += X.T.astype(np.float64) @ X.astype(np.float64)
            n_tot += X.shape[0]
    cov /= n_tot

    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    evals = np.clip(evals, 0, None)

    print(f"  fitted on {n_tot:,} patch samples for a {EMB_DIM}-d basis "
          f"({n_tot/EMB_DIM:.0f}:1 overdetermined)")
    return evals, evecs, mean_fn


def variance_report(evals: np.ndarray, k: int):
    tot = evals.sum()
    cum = np.cumsum(evals) / tot
    print(f"\n  variance explained at k={k}: {cum[k-1]:.1%}")
    for target in (0.90, 0.95, 0.99):
        kk = int(np.searchsorted(cum, target) + 1)
        print(f"    {target:.0%} reached at k = {kk:3d}   "
              f"({N_BLOCKS} × 364 × {kk} = {N_BLOCKS*364*kk:,} dims)")
    for kk in (16, 32, 64, 128, 256, 512):
        if kk <= len(cum):
            print(f"    k={kk:3d} → {cum[kk-1]:.1%}")


def project(P: dict, evecs: np.ndarray, mean_fn, k: int,
            evals: np.ndarray | None = None, whiten: bool = False):
    """(4, n_sp, 768) → flattened (4 × n_sp × k). Spatial structure preserved."""
    V = evecs[:, :k]
    if whiten and evals is not None:
        V = V / np.sqrt(np.clip(evals[:k], 1e-12, None))

    out = defaultdict(dict)
    for s in sorted(P):
        m = mean_fn(s)
        for cid, v in P[s].items():
            z = (v - m) @ V                       # (4, n_sp, k)
            out[s][cid] = z.ravel().astype(np.float32)
    return dict(out)


# =====================================================================
#  Reporting
# =====================================================================

def run_block(X, meta, title, dim):
    print("\n" + "=" * 78)
    print(f"{title}   —   {dim:,} dimensions")
    print("=" * 78)
    ranks, ncand, clips = evaluate(X, meta)
    overall = summarize(ranks, ncand, "ALL CLIPS")
    print_row(overall)

    by_cat = defaultdict(lambda: ([], []))
    for r, nc, c in zip(ranks, ncand, clips):
        lab = meta[c]["content_label"]
        by_cat[lab][0].append(r)
        by_cat[lab][1].append(nc)

    print()
    for cat in CATEGORY_ORDER:
        if cat in by_cat:
            r, nc = by_cat[cat]
            print_row(summarize(np.array(r), np.array(nc), cat))

    if "BLACK" in by_cat and "SPEECH_DYNAMIC" in by_cat:
        b = summarize(np.array(by_cat["BLACK"][0]), np.array(by_cat["BLACK"][1]))["pct"]
        s = summarize(np.array(by_cat["SPEECH_DYNAMIC"][0]),
                      np.array(by_cat["SPEECH_DYNAMIC"][1]))["pct"]
        overall["grad"] = b - s
        print(f"\n  drive gradient (BLACK − SPEECH_DYNAMIC) = {b - s:+.3f}")
    return overall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lag", type=int, required=True)
    p.add_argument("--k", type=int, default=250)
    p.add_argument("--center", default="global", choices=["global", "subject"])
    p.add_argument("--whiten", action="store_true")
    p.add_argument("--skip-voxel", action="store_true")
    args = p.parse_args()

    code  = Path(os.environ["CS_CODE"])
    cache = Path(os.environ["CS_CACHE"])

    print(f"lag={args.lag}  k={args.k}  center={args.center}  whiten={args.whiten}")

    P, C, meta = load_raw(cache / f"latents_lag{args.lag}")
    subs = sorted(P)
    n_sp = next(iter(P[subs[0]].values())).shape[1]
    print(f"{len(subs)} subjects × {len(meta)} clips   "
          f"patch_embeds = ({N_BLOCKS}, {n_sp}, {EMB_DIM})\n")

    results = {}

    # ---- 1. voxels, full 16 frames ----
    if not args.skip_voxel:
        V = load_voxels_full(code, cache, args.lag, set(meta))
        results["voxel_full"] = run_block(
            V, meta, "VOXELS — full 16 frames, no binning", NUM_FRAMES * FLAT_DIM)
        del V

    # ---- 2. CLS ----
    if all(C[s][c] is not None for s in C for c in C[s]):
        Xc = {s: {c: C[s][c] for c in C[s]} for s in C}
        results["cls"] = run_block(Xc, meta, "CLS TOKEN — global summary", EMB_DIM)
        del Xc
    else:
        print("\n⚠ cls_embeds empty — skipping")

    # ---- 3. channel PCA ----
    print("\n" + "=" * 78)
    print(f"CHANNEL PCA — 768 → {args.k}, all {n_sp} patches kept")
    print("=" * 78)
    evals, evecs, mean_fn = fit_channel_pca(P, center=args.center)
    variance_report(evals, args.k)

    Xp = project(P, evecs, mean_fn, args.k, evals, args.whiten)
    results[f"pca{args.k}"] = run_block(
        Xp, meta, f"CHANNEL PCA k={args.k}", N_BLOCKS * n_sp * args.k)

    # ---- summary ----
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  {'representation':>22s} {'dims':>12s} {'pct-rank':>9s} {'top-1':>7s} {'grad':>7s}")
    print("  " + "-" * 62)
    ref = [("latent flat  (earlier)", 1118208, 0.240, 0.172, 0.220),
           ("latent pooled(earlier)",    3072, 0.306, 0.083, 0.165),
           ("voxel binned (earlier)",  311052, 0.202, 0.117, None)]
    for name, d, pct, t1, g in ref:
        gs = f"{g:+.3f}" if g else "     —"
        print(f"  {name:>22s} {d:12,d} {pct:9.3f} {t1:7.3f} {gs:>7s}")
    print("  " + "-" * 62)
    dims = {"voxel_full": NUM_FRAMES * FLAT_DIM, "cls": EMB_DIM,
            f"pca{args.k}": N_BLOCKS * n_sp * args.k}
    for name, s in results.items():
        g = s.get("grad")
        gs = f"{g:+.3f}" if g is not None else "     —"
        print(f"  {name:>22s} {dims[name]:12,d} {s['pct']:9.3f} {s['top1']:7.3f} {gs:>7s}")


if __name__ == "__main__":
    main()
