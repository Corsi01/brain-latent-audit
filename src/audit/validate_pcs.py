#!/usr/bin/env python3
"""
validate_pcs.py — PC3 e PC5 sono reali o un artefatto da confronti multipli?

La diagnostica ha trovato che il vettore medio non e' una direzione singola: lo
spettro delle componenti della variazione fra patch e' piatto (PC1 solo 14%), e
due componenti secondarie separano DMN da reti sensoriali con d > 1.2 nella
direzione predetta, mentre la proiezione su un asse solo da' d = -0.17 perche'
componenti con relazioni opposte si cancellano.

Sono state selezionate DOPO aver guardato sei componenti. Servono tre controlli.

  1. SPIN TEST per componente. Su 364 patch spazialmente autocorrelati, d = 1.2
     senza null non significa quello che sembra. Correzione per le 6 componenti
     esaminate: la soglia non e' 0.05 ma 0.05/6.

  2. REPLICA CROSS-SOGGETTO. La SVD e' fatta sui vettori medi mediati sui 4
     soggetti. Si rifa' su 2 soggetti e si testa sugli altri 2: la componente
     deve riemergere con lo stesso allineamento E lo stesso contrasto. E' il
     controllo che distingue una direzione reale da struttura trovata nel rumore.

  3. STABILITA' SPLIT-HALF sui campioni, per componente. Le PC ad alta varianza
     sono stabili per costruzione; quelle a bassa varianza possono non esserlo,
     e PC5 e' al 3.8%.

Nota di nomenclatura: PC3 e PC5 non sono "la direzione media". Sono componenti
della VARIAZIONE FRA PATCH del vettore medio (la media per patch e' rimossa
prima della SVD). Oggetto diverso, va chiamato diversamente nel paper.

Uso:
    python validate_pcs.py --n-perm 1000
"""
import argparse
import glob
import itertools
import os

import numpy as np

H, W, PATCH = 224, 560, 16
GH, GW = H // PATCH, W // PATCH
NV, N_SP, D, N_T = 64984, 364, 768, 4
SENSORY = ["Vis", "SomMot"]
TARGET = "Default"
N_PC = 6


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
    return R, pop


def to_surface(v364, R, pop):
    img = np.full((H, W), np.nan, np.float32)
    ok = pop >= 0
    img[ok] = np.asarray(v364, np.float64)[pop[ok]]
    s = np.asarray(R.inverse(np.nan_to_num(img, nan=0.0)[None])).reshape(-1)
    out = np.zeros(NV)
    n = min(len(s), NV)
    out[:n] = s[:n]
    return out


def contrast(vals, nets):
    a, b = vals[nets == TARGET], vals[np.isin(nets, SENSORY)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    pooled = np.sqrt((a.var(ddof=1)*(len(a)-1) + b.var(ddof=1)*(len(b)-1)) /
                     (len(a)+len(b)-2))
    diff = a.mean() - b.mean()
    return diff, (diff/pooled if pooled > 0 else np.nan)


def svd_of(MA, f0):
    """[364,768] -> (scores [364,k], basis [k,768], varianza spiegata)."""
    A = MA - f0
    A = A - A.mean(axis=0, keepdims=True)      # rimuove la media PER PATCH
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U * S, Vt, S**2 / (S**2).sum()


def main():
    ap = argparse.ArgumentParser()
    cache = os.environ.get("CS_CACHE", ".")
    code = os.environ.get("CS_CODE", "..")
    sd = os.path.join(cache, "f0", "surface")
    ap.add_argument("--latents", default=os.path.join(cache, "latents_lag4"))
    ap.add_argument("--probes", default=os.path.join(cache, "f0", "probes.npz"))
    ap.add_argument("--diag", default=os.path.join(sd, "diagnose_negative.npz"))
    ap.add_argument("--outdir", default=sd)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    diag = np.load(args.diag, allow_pickle=True)
    dom = np.asarray(diag["dom"]).astype(str)
    R, pop = patch_geometry()

    # ---- medie per patch, per soggetto -----------------------------------
    files = sorted(glob.glob(os.path.join(args.latents, "*.npz")))
    per_sub, per_sub_files = {}, {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        s = str(z["subject"])
        e = np.asarray(z["patch_embeds"], np.float32).reshape(N_T, N_SP, D).mean(0)
        per_sub.setdefault(s, []).append(e)
        per_sub_files.setdefault(s, []).append(f)
    subs = sorted(per_sub)
    MA_sub = {s: np.mean(per_sub[s], axis=0) for s in subs}
    f0 = np.asarray(np.load(args.probes)["zeros"], np.float32)
    f0 = f0.reshape(N_T, N_SP, D).mean(0)
    MA_all = np.mean([MA_sub[s] for s in subs], axis=0)
    print(f"{len(files)} latenti, {len(subs)} soggetti")

    scores, basis, ev = svd_of(MA_all, f0)
    print(f"varianza: " + "  ".join(f"PC{i+1} {ev[i]:.3f}" for i in range(N_PC)))

    # ================================================== 1. spin test
    print("\n" + "="*66)
    print("1. SPIN TEST PER COMPONENTE")
    print("="*66)
    print(f"   {N_PC} componenti esaminate -> soglia corretta 0.05/{N_PC} = "
          f"{0.05/N_PC:.4f}")
    from neuromaps import nulls
    import nibabel as nib
    parc = np.asarray(nib.load(os.path.join(
        code, "atlases", "fslr32k",
        "Schaefer2018_400Parcels_7Networks_order.dlabel.nii")
    ).get_fdata()).squeeze().astype(int)

    print(f"\n   {'comp':6s} {'var%':>6s} {'d':>7s} {'p':>8s}  esito")
    spin = {}
    for i in range(N_PC):
        v = scores[:, i]
        _, dd = contrast(v, dom)
        p = np.nan
        if args.n_perm > 0:
            try:
                surf = to_surface(v, R, pop)
                rot = nulls.alexander_bloch(surf, atlas="fsLR", density="32k",
                                            n_perm=args.n_perm, seed=args.seed)
                # riporta ogni rotazione ai patch mediando per patch
                null_d = []
                v2p = np.round(np.asarray(R.inverse(
                    np.where(pop >= 0, pop, -1).astype(np.float32)[None])
                ).reshape(-1)).astype(int)
                v2p_full = np.full(NV, -1, int)
                n = min(len(v2p), NV)
                v2p_full[:n] = v2p[:n]
                for j in range(rot.shape[1]):
                    pv = np.array([rot[:, j][v2p_full == k].mean()
                                   if (v2p_full == k).sum() > 0 else np.nan
                                   for k in range(N_SP)])
                    ok = np.isfinite(pv)
                    _, ddj = contrast(pv[ok], dom[ok])
                    if np.isfinite(ddj):
                        null_d.append(ddj)
                null_d = np.array(null_d)
                if len(null_d) > 50:
                    p = float((np.abs(null_d) >= abs(dd)).mean())
            except Exception as e:
                print(f"      PC{i+1}: spin fallito ({type(e).__name__}: {e})")
        mark = ("SUPERA" if p < 0.05/N_PC else
                "nominale" if p < 0.05 else "no") if np.isfinite(p) else "?"
        print(f"   PC{i+1:<4d} {ev[i]:6.3f} {dd:+7.3f} {p:8.4f}  {mark}")
        spin[i] = (dd, p)

    # ================================================== 2. cross-soggetto
    print("\n" + "="*66)
    print("2. REPLICA CROSS-SOGGETTO (base su 2 soggetti, test sugli altri 2)")
    print("="*66)
    print("   La componente deve riemergere con lo stesso ALLINEAMENTO e lo")
    print("   stesso CONTRASTO. Il segno delle PC e' arbitrario: allineato.")
    print(f"\n   {'split':22s} {'comp':6s} {'|cos| basi':>11s} {'d held-out':>12s}")
    cross = {}
    for fit in itertools.combinations(subs, 2):
        held = [s for s in subs if s not in fit]
        MA_f = np.mean([MA_sub[s] for s in fit], axis=0)
        MA_h = np.mean([MA_sub[s] for s in held], axis=0)
        _, B_f, _ = svd_of(MA_f, f0)
        Ah = MA_h - f0
        Ah = Ah - Ah.mean(axis=0, keepdims=True)
        for i in (2, 4):          # PC3 e PC5, indici 0-based
            # allineamento delle basi
            c = abs(float(np.dot(B_f[i] / np.linalg.norm(B_f[i]),
                                 basis[i] / np.linalg.norm(basis[i]))))
            sc_h = Ah @ B_f[i]
            _, dd = contrast(sc_h, dom)
            sgn = np.sign(np.dot(B_f[i], basis[i]))
            print(f"   {'+'.join(fit):22s} PC{i+1:<4d} {c:11.3f} "
                  f"{dd*sgn:+12.3f}")
            cross.setdefault(i, []).append((c, dd*sgn))
    for i in (2, 4):
        cs = np.array(cross[i])
        print(f"   PC{i+1}: |cos| medio {cs[:,0].mean():.3f}   "
              f"d held-out {cs[:,1].mean():+.3f} "
              f"(min {cs[:,1].min():+.3f}, max {cs[:,1].max():+.3f})")
    print("   |cos| basso -> la componente non e' la stessa nei due gruppi:")
    print("   il confronto di d non ha significato.")

    # ================================================== 3. split-half
    print("\n" + "="*66)
    print("3. STABILITA' SPLIT-HALF SUI CAMPIONI")
    print("="*66)
    rng = np.random.default_rng(args.seed)
    print(f"   {'comp':6s} {'r(scores)':>11s} {'|cos| basi':>11s}")
    for i in range(N_PC):
        rs, cs = [], []
        for _ in range(10):
            h = {}
            for s in subs:
                idx = rng.permutation(len(per_sub[s]))
                h.setdefault(0, []).append(
                    np.mean([per_sub[s][j] for j in idx[:len(idx)//2]], axis=0))
                h.setdefault(1, []).append(
                    np.mean([per_sub[s][j] for j in idx[len(idx)//2:]], axis=0))
            s1, b1, _ = svd_of(np.mean(h[0], axis=0), f0)
            s2, b2, _ = svd_of(np.mean(h[1], axis=0), f0)
            cs.append(abs(float(np.dot(b1[i]/np.linalg.norm(b1[i]),
                                       b2[i]/np.linalg.norm(b2[i])))))
            rs.append(abs(float(np.corrcoef(s1[:, i], s2[:, i])[0, 1])))
        print(f"   PC{i+1:<4d} {np.mean(rs):11.4f} {np.mean(cs):11.4f}")
    print("   Le PC a bassa varianza possono non essere stabili: PC5 e' al 3.8%.")

    np.savez_compressed(os.path.join(args.outdir, "validate_pcs.npz"),
                        scores=scores[:, :N_PC], basis=basis[:N_PC], ev=ev[:N_PC],
                        spin_d=np.array([spin[i][0] for i in range(N_PC)]),
                        spin_p=np.array([spin[i][1] for i in range(N_PC)]))
    print(f"\nsalvato in {args.outdir}/validate_pcs.npz")
    print("\nRicorda: PC3 e PC5 sono state selezionate DOPO aver guardato sei")
    print("componenti. Anche se superano, restano un risultato POST-HOC e vanno")
    print("riportate come tali, non come predizione.")


if __name__ == "__main__":
    main()
