#!/usr/bin/env python3
"""
positive_control.py — la catena patch -> corteccia e' anatomicamente corretta?

L'ordinamento dei token e' verificato al 100%, ma questo dimostra solo che
l'indicizzazione e' coerente. Non dimostra che la corrispondenza patch -> vertici
fsLR sia anatomicamente giusta: un errore di orientamento della flat map (assi
scambiati, emisferi invertiti, flip verticale) passerebbe indenne quel test.

Il controllo che lo smaschera: un modello di encoding per patch verso feature di
stimolo di cui si conosce gia' l'anatomia.

    frame_diff (motion visivo) -> deve accendere le aree visive
    rms / speech_coverage      -> devono accendere le aree uditive

Se le due mappe si accendono nello stesso posto, o in posti sbagliati, la catena
e' rotta e la mappa di partecipazione non vale nulla. E' un test doppio: non solo
"si accende qualcosa", ma "due cose diverse si accendono in due posti diversi".

I descrittori del manifest sono per blocco (4 blocchi da 4 s), quindi si allineano
uno-a-uno con i 4 blocchi temporali del latente: 622 x 4 = 2488 campioni.

Uso:
    python positive_control.py
"""
import argparse
import glob
import json
import os

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

H, W, PATCH = 224, 560, 16
GH, GW = H // PATCH, W // PATCH
N_T, N_SP, D = 4, 364, 768
FEATURES = ["frame_diff", "luminance", "rms", "speech_coverage"]


def main():
    ap = argparse.ArgumentParser()
    cache = os.environ.get("CS_CACHE", ".")
    code = os.environ.get("CS_CODE", "..")
    ap.add_argument("--latents", default=os.path.join(cache, "latents_lag4"))
    ap.add_argument("--manifest", default=os.path.join(code, "manifest", "clip_manifest.json"))
    ap.add_argument("--outdir", default=os.path.join(cache, "f0", "surface"))
    ap.add_argument("--alpha", type=float, default=1e4)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------------------------------------------- 1. dati
    feats = {}
    for c in json.load(open(args.manifest))["clips"]:
        blocks = c.get("blocks") or []
        if len(blocks) == N_T:
            feats[c["clip_id"]] = np.array(
                [[b.get(f, np.nan) for f in FEATURES] for b in blocks], np.float64)

    files = sorted(glob.glob(os.path.join(args.latents, "*.npz")))
    Z, Y, subj, run = [], [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        cid = str(z["clip_id"])
        if cid not in feats:
            continue
        Z.append(np.asarray(z["patch_embeds"], np.float32).reshape(N_T, N_SP, D))
        Y.append(feats[cid])
        subj.append(str(z["subject"]))
        run.append(str(z["run_id"]))
    Z = np.stack(Z)                                   # [n, 4, 364, 768]
    Y = np.stack(Y)                                   # [n, 4, F]
    subj, run = np.array(subj), np.array(run)
    print(f"{len(Z)} clip x {N_T} blocchi = {len(Z)*N_T} campioni")
    print(f"feature: {FEATURES}")

    # centering B per soggetto: toglie la direzione di baseline
    for s in np.unique(subj):
        m = Z[subj == s].mean(axis=(0, 1, 2))
        Z[subj == s] -= m[None, None, None, :]

    n = len(Z)
    Yf = Y.reshape(n * N_T, len(FEATURES))
    groups = np.repeat(run, N_T)
    keep = ~np.isnan(Yf).any(axis=1)
    print(f"campioni utilizzabili: {keep.sum()} / {len(Yf)}")
    Yf = (Yf[keep] - Yf[keep].mean(0)) / (Yf[keep].std(0) + 1e-9)
    groups = groups[keep]
    print(f"run distinti (fold): {len(np.unique(groups))}")

    # ---------------------------------------------------- 2. encoding per patch
    print(f"\n=== ENCODING PER PATCH (ridge alpha={args.alpha:g}, "
          f"{args.folds} fold per run) ===")
    gkf = GroupKFold(n_splits=args.folds)
    splits = list(gkf.split(np.zeros(len(Yf)), groups=groups))
    R = np.zeros((N_SP, len(FEATURES)))

    for p in range(N_SP):
        X = Z[:, :, p, :].reshape(n * N_T, D)[keep]
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        pred = np.zeros_like(Yf)
        for tr, te in splits:
            m = Ridge(alpha=args.alpha).fit(X[tr], Yf[tr])
            pred[te] = m.predict(X[te])
        for j in range(len(FEATURES)):
            R[p, j] = np.corrcoef(pred[:, j], Yf[:, j])[0, 1]
        if (p + 1) % 100 == 0:
            print(f"   {p+1}/{N_SP}", flush=True)

    print("\n   feature          R medio    R max    patch migliori (indice)")
    for j, f in enumerate(FEATURES):
        top = np.argsort(-R[:, j])[:5]
        print(f"   {f:16s} {R[:, j].mean():7.3f}  {R[:, j].max():7.3f}   {list(top)}")

    print("\n   sovrapposizione fra mappe (correlazione fra patch):")
    for j in range(len(FEATURES)):
        for k in range(j + 1, len(FEATURES)):
            r = np.corrcoef(R[:, j], R[:, k])[0, 1]
            print(f"      {FEATURES[j]:16s} vs {FEATURES[k]:16s}  r = {r:6.3f}")
    print("   frame_diff e rms devono correlare POCO: aree diverse.")
    print("   Se correlano molto, o la catena e' rotta o le feature sono collineari.")

    # -------------------------------------------------- 3. proiezione superficie
    from cortex_mae import transforms
    Rs = transforms._FLAT_RESAMPLER
    mask = np.asarray(Rs.mask_)
    m = mask.reshape(GH, PATCH, GW, PATCH).transpose(0, 2, 1, 3).reshape(GH * GW, -1)
    valid = m.any(axis=1)
    rank = np.cumsum(valid) - 1
    patch_of_pixel = np.full((H, W), -1, np.int32)
    for pi in np.flatnonzero(valid):
        gr, gc = divmod(pi, GW)
        patch_of_pixel[gr*PATCH:(gr+1)*PATCH, gc*PATCH:(gc+1)*PATCH] = rank[pi]
    patch_of_pixel[~mask] = -1

    def to_surface(v364):
        img = np.zeros((H, W), np.float32)
        ok = patch_of_pixel >= 0
        img[ok] = v364[patch_of_pixel[ok]]
        return np.asarray(Rs.inverse(img[None])).reshape(-1)

    out = {f"R_{f}": R[:, j] for j, f in enumerate(FEATURES)}
    out.update({f"surf_{f}": to_surface(R[:, j]) for j, f in enumerate(FEATURES)})
    out["cover"] = m.sum(axis=1)[valid] / (PATCH * PATCH)

    fout = os.path.join(args.outdir, "positive_control.npz")
    np.savez_compressed(fout, **out)
    print(f"\nsalvato in {fout}")
    print("Ispeziona le mappe su superficie: frame_diff deve concentrarsi in")
    print("corteccia occipitale, rms/speech in corteccia temporale superiore.")


if __name__ == "__main__":
    main()
