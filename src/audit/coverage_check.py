#!/usr/bin/env python3
"""
coverage_check.py — il pattern bordo-centro e' biologia o geometria?

I patch ad alta partecipazione stanno sul bordo della flat map, e i patch di
bordo sono anche quelli meno coperti da corteccia. Se partecipazione e copertura
correlano, il pattern e' un artefatto di come e' fatta la griglia, non una
proprieta' del cervello.

Tre controlli:
  1. correlazione partecipazione ~ copertura del patch
  2. correlazione partecipazione ~ distanza dal bordo della maschera
  3. partecipazione dopo aver regredito via entrambe: il pattern sopravvive?

E in piu' rifa' il controllo positivo con le feature z-scorate DENTRO ciascun run.
Nel primo tentativo rms aveva R medio negativo: le feature variano molto fra
episodi, i fold tagliano per run, quindi il modello non puo' usare quella varianza
mentre il test set ce l'ha ancora. Z-scorare per run misura solo la modulazione
entro-episodio, che e' quello che interessa.

Uso:
    python coverage_check.py                    # solo controlli 1-3
    python coverage_check.py --redo-encoding    # rifa' anche l'encoding
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import ndimage

H, W, PATCH = 224, 560, 16
GH, GW = H // PATCH, W // PATCH
N_T, N_SP, D = 4, 364, 768
FEATURES = ["frame_diff", "luminance", "rms", "speech_coverage"]


def patch_geometry():
    from cortex_mae import transforms
    R = transforms._FLAT_RESAMPLER
    mask = np.asarray(R.mask_)
    m = mask.reshape(GH, PATCH, GW, PATCH).transpose(0, 2, 1, 3).reshape(GH * GW, -1)
    valid = m.any(axis=1)
    ids = np.flatnonzero(valid)
    cover = m.sum(axis=1)[valid] / (PATCH * PATCH)

    # distanza euclidea dal bordo della maschera, mediata sui pixel del patch
    dist = ndimage.distance_transform_edt(mask)
    dpatch = np.zeros(len(ids))
    for i, p in enumerate(ids):
        gr, gc = divmod(int(p), GW)
        blk = dist[gr*PATCH:(gr+1)*PATCH, gc*PATCH:(gc+1)*PATCH]
        msk = mask[gr*PATCH:(gr+1)*PATCH, gc*PATCH:(gc+1)*PATCH]
        dpatch[i] = blk[msk].mean() if msk.any() else 0.0
    return mask, valid, ids, cover, dpatch


def main():
    ap = argparse.ArgumentParser()
    cache = os.environ.get("CS_CACHE", ".")
    code = os.environ.get("CS_CODE", "..")
    sd = os.path.join(cache, "f0", "surface")
    ap.add_argument("--maps", default=os.path.join(
        cache, "f0", "maps", "participation_maps.npz"))
    ap.add_argument("--latents", default=os.path.join(cache, "latents_lag4"))
    ap.add_argument("--manifest", default=os.path.join(
        code, "manifest", "clip_manifest.json"))
    ap.add_argument("--outdir", default=sd)
    ap.add_argument("--redo-encoding", action="store_true")
    ap.add_argument("--alpha", type=float, default=1e4)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    mask, valid, ids, cover, dist = patch_geometry()
    print(f"{len(ids)} patch validi")
    print(f"copertura: min {cover.min():.3f}  mediana {np.median(cover):.3f}  "
          f"max {cover.max():.3f}")
    print(f"patch sotto 25% di copertura: {(cover <= 0.25).sum()}")

    part = np.load(args.maps, allow_pickle=True)
    subs = [str(s) for s in part["subjects"]]
    P = np.mean([part[f"cos_{s}"] for s in subs], axis=0)

    print("\n=== 1-2. LA PARTECIPAZIONE E' SPIEGATA DALLA GEOMETRIA? ===")
    r_cov = np.corrcoef(P, cover)[0, 1]
    r_dist = np.corrcoef(P, dist)[0, 1]
    print(f"   partecipazione ~ copertura        r = {r_cov:+.3f}")
    print(f"   partecipazione ~ distanza dal bordo r = {r_dist:+.3f}")
    print(f"   copertura ~ distanza dal bordo    r = {np.corrcoef(cover, dist)[0,1]:+.3f}")
    print("   |r| alto -> il pattern bordo-centro e' geometria della griglia")
    print("   |r| basso -> e' una proprieta' dei dati")

    print("\n=== 3. PARTECIPAZIONE RESIDUA DOPO AVER TOLTO LA GEOMETRIA ===")
    X = np.column_stack([np.ones(len(P)), cover, dist,
                         cover**2, dist**2, cover*dist])
    beta, *_ = np.linalg.lstsq(X, P, rcond=None)
    resid = P - X @ beta
    r2 = 1 - resid.var() / P.var()
    print(f"   varianza spiegata da copertura+distanza (con termini quadratici): "
          f"{r2:.3f}")
    print(f"   partecipazione residua: sd {resid.std():.4f} "
          f"(originale {P.std():.4f})")
    # la mappa residua replica ancora tra soggetti?
    Ps = {s: np.asarray(part[f"cos_{s}"]) for s in subs}
    res_s = {s: Ps[s] - X @ np.linalg.lstsq(X, Ps[s], rcond=None)[0] for s in subs}
    import itertools
    cc = [np.corrcoef(res_s[a], res_s[b])[0, 1]
          for a, b in itertools.combinations(subs, 2)]
    print(f"   replica tra soggetti della mappa RESIDUA: "
          f"{min(cc):.4f} - {max(cc):.4f}")
    print("   Se resta alta, la struttura non geometrica e' riproducibile:")
    print("   e' quella da confrontare con gli atlanti.")

    out = {"cover": cover, "dist_edge": dist, "part_group": P,
           "part_resid": resid, "valid_ids": ids}

    # ------------------------------------------------------------ 4. encoding
    if args.redo_encoding:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GroupKFold

        print("\n=== 4. ENCODING CON FEATURE Z-SCORATE PER RUN ===")
        feats = {}
        for c in json.load(open(args.manifest))["clips"]:
            b = c.get("blocks") or []
            if len(b) == N_T:
                feats[c["clip_id"]] = np.array(
                    [[bb.get(f, np.nan) for f in FEATURES] for bb in b], np.float64)

        Z, Y, subj, run = [], [], [], []
        for f in sorted(glob.glob(os.path.join(args.latents, "*.npz"))):
            z = np.load(f, allow_pickle=True)
            cid = str(z["clip_id"])
            if cid not in feats:
                continue
            Z.append(np.asarray(z["patch_embeds"], np.float32).reshape(N_T, N_SP, D))
            Y.append(feats[cid]); subj.append(str(z["subject"]))
            run.append(str(z["run_id"]))
        Z = np.stack(Z); Y = np.stack(Y)
        subj, run = np.array(subj), np.array(run)
        for s in np.unique(subj):
            Z[subj == s] -= Z[subj == s].mean(axis=(0, 1, 2))[None, None, None, :]

        n = len(Z)
        Yf = Y.reshape(n * N_T, len(FEATURES))
        groups = np.repeat(run, N_T)
        keep = ~np.isnan(Yf).any(axis=1)
        Yf, groups = Yf[keep], groups[keep]

        # z-score DENTRO ciascun run: e' la correzione
        for g in np.unique(groups):
            sel = groups == g
            if sel.sum() < 3:
                continue
            Yf[sel] = (Yf[sel] - Yf[sel].mean(0)) / (Yf[sel].std(0) + 1e-9)
        print(f"   {keep.sum()} campioni, {len(np.unique(groups))} run")

        splits = list(GroupKFold(n_splits=args.folds).split(
            np.zeros(len(Yf)), groups=groups))
        R = np.zeros((N_SP, len(FEATURES)))
        for p in range(N_SP):
            Xp = Z[:, :, p, :].reshape(n * N_T, D)[keep]
            Xp = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-9)
            pred = np.zeros_like(Yf)
            for tr, te in splits:
                pred[te] = Ridge(alpha=args.alpha).fit(Xp[tr], Yf[tr]).predict(Xp[te])
            for j in range(len(FEATURES)):
                R[p, j] = np.corrcoef(pred[:, j], Yf[:, j])[0, 1]
            if (p + 1) % 100 == 0:
                print(f"   {p+1}/{N_SP}", flush=True)

        print("\n   feature          R medio   R max")
        for j, f in enumerate(FEATURES):
            print(f"   {f:16s} {R[:, j].mean():7.3f} {R[:, j].max():7.3f}")
        print("\n   sovrapposizione fra mappe:")
        for j in range(len(FEATURES)):
            for k in range(j + 1, len(FEATURES)):
                print(f"      {FEATURES[j]:16s} vs {FEATURES[k]:16s}  "
                      f"r = {np.corrcoef(R[:, j], R[:, k])[0,1]:+.3f}")
        print("\n   partecipazione vs mappe di encoding (il controllo NEGATIVO):")
        for j, f in enumerate(FEATURES):
            print(f"      partecipazione vs {f:16s} r = "
                  f"{np.corrcoef(P, R[:, j])[0,1]:+.3f}   "
                  f"(residua: {np.corrcoef(resid, R[:, j])[0,1]:+.3f})")
        print("   r NEGATIVO = la partecipazione evita le aree sensoriali.")
        out.update({f"R_{f}": R[:, j] for j, f in enumerate(FEATURES)})

    fout = os.path.join(args.outdir, "coverage_check.npz")
    np.savez_compressed(fout, **out)
    print(f"\nsalvato in {fout}")


if __name__ == "__main__":
    main()
