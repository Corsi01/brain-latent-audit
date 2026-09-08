#!/usr/bin/env python3
"""
diagnose_negative.py — il risultato Yeo negativo e' reale o autoinflitto?

Quattro sospetti, in ordine di gravita'.

  A. La regressione geometrica ha rimosso il DMN.
     La distanza dal bordo della flat map NON e' anatomicamente neutra: il bordo
     corrisponde a corteccia mediale, insula, poli temporali — e il DMN vive in
     buona parte sulla superficie mediale. Regredendola si potrebbe aver rimosso
     proprio la variabilita' che portava il segnale.
     -> quanto correlano copertura e distanza con l'appartenenza al DMN?
     -> test Yeo su varianti di correzione: nessuna / solo copertura / entrambe

  B. La risoluzione dei patch attenua l'effetto.
     Patch quadrati da 16x16 px straddiano piu' reti Yeo: la media per parcella
     mescola patch di reti diverse. Attenuazione, non bias — ma con un effetto
     vero del 2-3% puo' bastare a cancellarlo.
     -> purezza di rete per patch; test ristretto ai patch puri

  C. La scelta del coseno.
     cos e' invariante alla scala locale, ma l'eterogeneita' del tessuto dentro
     il patch potrebbe dominare l'allineamento. proj e norm non sono mai state
     testate contro Yeo.
     -> contrasto Default vs sensorie per tutte e tre le mappe

  D. m_B non e' una direzione sola.
     Se e' una miscela (fisiologica + vascolare + rumore di acquisizione),
     proiettare tutto su un asse le confonde.
     -> PCA sui vettori medi per patch; le prime componenti separatamente

Uso:
    python diagnose_negative.py                 # A, B, C
    python diagnose_negative.py --n-perm 1000   # con spin test
"""
import argparse
import os
import sys

import numpy as np

H, W, PATCH = 224, 560, 16
GH, GW = H // PATCH, W // PATCH
NV, N_SP, D, N_T = 64984, 364, 768, 4
SENSORY = ["Vis", "SomMot"]
TARGET = "Default"


def patch_geometry():
    from cortex_mae import transforms
    R = transforms._FLAT_RESAMPLER
    mask = np.asarray(R.mask_)
    m = mask.reshape(GH, PATCH, GW, PATCH).transpose(0, 2, 1, 3).reshape(GH*GW, -1)
    valid = m.any(axis=1)
    rank = np.cumsum(valid) - 1
    pop = np.full((H, W), -1, np.int32)
    for p in np.flatnonzero(valid):
        gr, gc = divmod(int(p), GW)
        pop[gr*PATCH:(gr+1)*PATCH, gc*PATCH:(gc+1)*PATCH] = rank[p]
    pop[~mask] = -1
    return R, pop, m.sum(axis=1)[valid] / (PATCH*PATCH)


def vertex_to_patch(R, pop):
    img = np.where(pop >= 0, pop, -1).astype(np.float32)
    v = np.asarray(R.inverse(img[None])).reshape(-1)
    out = np.full(NV, -1, np.int32)
    n = min(len(v), NV)
    out[:n] = np.round(v[:n]).astype(np.int32)
    return out


def contrast(vals, nets):
    """Default - (Vis + SomMot); ritorna (diff, d di Cohen)."""
    a = vals[nets == TARGET]
    b = vals[np.isin(nets, SENSORY)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    pooled = np.sqrt((a.var(ddof=1)*(len(a)-1) + b.var(ddof=1)*(len(b)-1)) /
                     (len(a)+len(b)-2))
    diff = a.mean() - b.mean()
    return diff, diff / pooled if pooled > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    cache = os.environ.get("CS_CACHE", ".")
    code = os.environ.get("CS_CODE", "..")
    sd = os.path.join(cache, "f0", "surface")
    ap.add_argument("--check", default=os.path.join(sd, "coverage_check.npz"))
    ap.add_argument("--maps", default=os.path.join(
        cache, "f0", "maps", "participation_maps.npz"))
    ap.add_argument("--latents", default=os.path.join(cache, "latents_lag4"))
    ap.add_argument("--probes", default=os.path.join(cache, "f0", "probes.npz"))
    ap.add_argument("--parc", default=os.path.join(
        code, "atlases", "fslr32k",
        "Schaefer2018_400Parcels_7Networks_order.dlabel.nii"))
    ap.add_argument("--outdir", default=sd)
    ap.add_argument("--n-perm", type=int, default=0)
    ap.add_argument("--skip-pca", action="store_true")
    args = ap.parse_args()

    import nibabel as nib

    R, pop, cover = patch_geometry()
    v2p = vertex_to_patch(R, pop)
    d = np.load(args.check)
    dist = np.asarray(d["dist_edge"])
    P_raw = np.asarray(d["part_group"])
    P_res = np.asarray(d["part_resid"])

    img = nib.load(args.parc)
    parc = np.asarray(img.get_fdata()).squeeze().astype(int)
    labels = dict(img.header.get_axis(0).label[0])
    net_of = {k: n.split("_")[2] for k, (n, _) in labels.items()
              if k != 0 and len(n.split("_")) > 2}

    # ---- rete dominante per PATCH, con purezza --------------------------
    net_names = sorted(set(net_of.values()))
    ni = {n: i for i, n in enumerate(net_names)}
    counts = np.zeros((N_SP, len(net_names)))
    for v in range(NV):
        p = v2p[v]
        k = parc[v]
        if p >= 0 and k in net_of:
            counts[p, ni[net_of[k]]] += 1
    tot = counts.sum(axis=1)
    has = tot > 0
    dom = np.full(N_SP, "", object)
    purity = np.zeros(N_SP)
    dom[has] = np.array(net_names)[counts[has].argmax(axis=1)]
    purity[has] = counts[has].max(axis=1) / tot[has]
    print(f"patch con etichetta di rete: {int(has.sum())}/{N_SP}")
    print(f"purezza: mediana {np.median(purity[has]):.3f}  "
          f"sopra 0.6: {int((purity > 0.6).sum())}  "
          f"sopra 0.8: {int((purity > 0.8).sum())}")
    print("   distribuzione per rete (patch dominanti):")
    for n in net_names:
        print(f"      {n:12s} {int((dom == n).sum()):3d}")

    # ================================================== A. la geometria e il DMN
    print("\n" + "="*64)
    print("A. LA REGRESSIONE GEOMETRICA HA RIMOSSO IL DMN?")
    print("="*64)
    is_dmn = (dom == TARGET)[has]
    is_sen = np.isin(dom, SENSORY)[has]
    cv, dv = cover[has], dist[has]
    print(f"   copertura:  DMN {cv[is_dmn].mean():.3f}   "
          f"sensorie {cv[is_sen].mean():.3f}   "
          f"r(cover, e' DMN) = {np.corrcoef(cv, is_dmn.astype(float))[0,1]:+.3f}")
    print(f"   dist.bordo: DMN {dv[is_dmn].mean():.3f}   "
          f"sensorie {dv[is_sen].mean():.3f}   "
          f"r(dist, e' DMN)  = {np.corrcoef(dv, is_dmn.astype(float))[0,1]:+.3f}")
    print("   |r| alto -> regredire la geometria rimuove il DMN per costruzione")
    print("   |r| basso -> la correzione e' innocente, il negativo e' reale")

    # varianti di correzione
    print("\n   CONTRASTO Default - sensorie, per variante di correzione")
    print(f"   (a livello di PATCH, {int(has.sum())} patch; ipotesi: positivo)")
    X_none = np.ones((N_SP, 1))
    X_cov = np.column_stack([np.ones(N_SP), cover, cover**2])
    X_both = np.column_stack([np.ones(N_SP), cover, dist,
                              cover**2, dist**2, cover*dist])
    variants = {"nessuna correzione": X_none,
                "solo copertura": X_cov,
                "copertura + distanza (usata)": X_both}
    part = np.load(args.maps, allow_pickle=True)
    subs = [str(s) for s in part["subjects"]]
    base = {"cos": np.mean([part[f"cos_{s}"] for s in subs], axis=0),
            "proj": np.mean([part[f"proj_{s}"] for s in subs], axis=0),
            "norm": np.mean([part[f"norm_{s}"] for s in subs], axis=0)}

    print(f"\n   {'mappa':6s} {'correzione':30s} {'diff':>9s} {'d':>7s}")
    res = {}
    for mk, mv in base.items():
        for vk, X in variants.items():
            beta, *_ = np.linalg.lstsq(X, mv, rcond=None)
            r = mv - X @ beta
            diff, dd = contrast(r[has], dom[has])
            flag = "  <-- ipotesi confermata" if diff > 0 else ""
            print(f"   {mk:6s} {vk:30s} {diff:+9.4f} {dd:+7.3f}{flag}")
            res[f"{mk}_{vk}"] = (diff, dd)

    # ================================================== B. purezza
    print("\n" + "="*64)
    print("B. LA RISOLUZIONE ATTENUA L'EFFETTO?")
    print("="*64)
    print(f"   {'soglia':10s} {'n patch':>8s} {'diff':>9s} {'d':>7s}")
    for thr in (0.0, 0.5, 0.6, 0.7, 0.8):
        sel = has & (purity > thr)
        if sel.sum() < 20:
            continue
        beta, *_ = np.linalg.lstsq(X_both, P_res, rcond=None)
        r = P_res - X_both @ beta
        diff, dd = contrast(r[sel], dom[sel])
        print(f"   >{thr:.1f}      {int(sel.sum()):8d} {diff:+9.4f} {dd:+7.3f}")
    print("   Se l'effetto cresce con la purezza, e' attenuazione da risoluzione.")

    # ================================================== D. m_B e' una sola direzione?
    if not args.skip_pca:
        print("\n" + "="*64)
        print("D. m_B E' UNA MISCELA DI PIU' COMPONENTI?")
        print("="*64)
        import glob
        files = sorted(glob.glob(os.path.join(args.latents, "*.npz")))
        acc, cnt = {}, {}
        for f in files:
            z = np.load(f, allow_pickle=True)
            s = str(z["subject"])
            e = np.asarray(z["patch_embeds"], np.float32).reshape(N_T, N_SP, D).mean(0)
            acc[s] = acc.get(s, 0.0) + e
            cnt[s] = cnt.get(s, 0) + 1
        MA = np.mean([acc[s] / cnt[s] for s in acc], axis=0)      # [364, 768]
        f0 = np.asarray(np.load(args.probes)["zeros"], np.float32)
        f0 = f0.reshape(N_T, N_SP, D).mean(0)
        A = MA - f0
        Ac = A - A.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(Ac, full_matrices=False)
        ev = S**2 / (S**2).sum()
        print(f"   varianza per componente (dopo rimozione della media per patch):")
        print("      " + "  ".join(f"PC{i+1} {ev[i]:.3f}" for i in range(6)))
        print(f"\n   contrasto Default - sensorie, per componente principale:")
        print(f"   {'comp':6s} {'var%':>6s} {'diff':>9s} {'d':>7s}")
        for i in range(6):
            sc = U[:, i] * S[i]
            diff, dd = contrast(sc[has], dom[has])
            flag = "  <--" if abs(dd) > 0.5 else ""
            print(f"   PC{i+1:<4d} {ev[i]:6.3f} {diff:+9.4f} {dd:+7.3f}{flag}")
        print("   Se una PC secondaria separa nettamente le reti, m_B e' una")
        print("   miscela e proiettare su un asse solo confonde le componenti.")
        res["pca_ev"] = ev[:10]

    np.savez_compressed(os.path.join(args.outdir, "diagnose_negative.npz"),
                        dom=dom.astype(str), purity=purity, has=has,
                        **{k: np.asarray(v) for k, v in res.items()})
    print(f"\nsalvato in {args.outdir}/diagnose_negative.npz")


if __name__ == "__main__":
    main()
