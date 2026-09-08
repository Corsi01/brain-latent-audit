#!/usr/bin/env python3
"""
yeo_test.py — la partecipazione e' piu' alta nel DMN che nelle reti sensoriali?

E' il TEST PRIMARIO, e il primo test davvero indipendente dal gradiente di
Margulies. Una differenza fra reti discrete e' anche piu' interpretabile di una
correlazione continua, e meno sensibile alla soppressione lineare che rende
grezza e residua cosi' diverse nel test sul gradiente.

Parcellazione: Schaefer-400 7Networks in fsLR-32k dal repo CBIG. Verificato che
il file abbia shape (64984,) con lo stesso ordinamento della mappa, quindi
nessun ricampionamento. Le etichette hanno forma 7Networks_{LH,RH}_{Rete}_{n}.

Ipotesi pre-specificata (dal rapporto di luglio, prima di questi dati):
    partecipazione(Default) > partecipazione(Vis, SomMot)

Due mappe in parallelo, come per il gradiente:
    residua  = dopo regressione di copertura e distanza dal bordo  [PRIMARIA]
    grezza   = cos, media soggetti

Statistica:
  - a livello di PARCELLA (400), non di vertice: i vertici dentro una parcella
    non sono indipendenti
  - null = spin test, che preserva l'autocorrelazione spaziale
  - riportati sia il test sia il rapporto fra medie, perche' "significativo" e
    "grande" sono affermazioni diverse e la modulazione qui e' del ~10%

Uso:
    python yeo_test.py                      # tutte le 364 parcelle di patch
    python yeo_test.py --min-cover 0.25     # esclude i patch di bordo
"""
import argparse
import os
import sys

import numpy as np

H, W, PATCH = 224, 560, 16
GH, GW = H // PATCH, W // PATCH
NV = 64984
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


def to_surface(v364, R, pop, drop=None):
    v = np.asarray(v364, np.float64).copy()
    if drop is not None:
        v[drop] = np.nan
    img = np.full((H, W), np.nan, np.float32)
    ok = pop >= 0
    img[ok] = v[pop[ok]]
    bad = np.isnan(img)
    s = np.asarray(R.inverse(np.nan_to_num(img, nan=0.0)[None])).reshape(-1)
    b = np.asarray(R.inverse(bad[None].astype(np.float32))).reshape(-1) > 0.5
    out = np.full(NV, np.nan)
    n = min(len(s), NV)
    out[:n] = s[:n]
    out[:n][b[:n]] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    cache = os.environ.get("CS_CACHE", ".")
    code = os.environ.get("CS_CODE", "..")
    sd = os.path.join(cache, "f0", "surface")
    ap.add_argument("--check", default=os.path.join(sd, "coverage_check.npz"))
    ap.add_argument("--parc", default=os.path.join(
        code, "atlases", "fslr32k",
        "Schaefer2018_400Parcels_7Networks_order.dlabel.nii"))
    ap.add_argument("--outdir", default=sd)
    ap.add_argument("--min-cover", type=float, default=0.0)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import nibabel as nib
    try:
        from neuromaps import nulls
    except ImportError:
        nulls = None

    # ------------------------------------------------------------- parcelle
    img = nib.load(args.parc)
    parc = np.asarray(img.get_fdata()).squeeze().astype(int)
    labels = dict(img.header.get_axis(0).label[0])
    if parc.shape[0] != NV:
        sys.exit(f"parcellazione {parc.shape}, attesa ({NV},)")

    net_of = {}
    for k, (name, _) in labels.items():
        if k == 0:
            continue
        parts = name.split("_")          # 7Networks_LH_Vis_1
        net_of[k] = parts[2] if len(parts) > 2 else "?"
    nets = sorted(set(net_of.values()))
    print(f"parcellazione: {parc.shape}, {len(net_of)} parcelle, reti: {nets}")

    # ---------------------------------------------------------------- mappe
    R, pop, cover = patch_geometry()
    d = np.load(args.check)
    maps = {"residua": np.asarray(d["part_resid"]),
            "grezza": np.asarray(d["part_group"])}
    drop = cover < args.min_cover if args.min_cover > 0 else None
    if drop is not None:
        print(f"esclusi {int(drop.sum())} patch sotto {args.min_cover:.0%} copertura")
    surf = {k: to_surface(v, R, pop, drop) for k, v in maps.items()}

    def parcel_means(v):
        """media per parcella -> (valori, reti) sulle parcelle con dati."""
        vals, who = [], []
        for k in sorted(net_of):
            sel = (parc == k) & np.isfinite(v)
            if sel.sum() >= 20:
                vals.append(v[sel].mean())
                who.append(net_of[k])
        return np.array(vals), np.array(who)

    results = {}
    for name, v in surf.items():
        print(f"\n{'='*62}\n=== {name.upper()} ===")
        pv, pn = parcel_means(v)
        print(f"   {len(pv)} parcelle con dati sufficienti")

        print(f"\n   {'rete':10s} {'n':>4s} {'media':>9s} {'sd':>8s}")
        stats_by_net = {}
        for net in nets:
            s = pv[pn == net]
            if len(s) == 0:
                continue
            stats_by_net[net] = s
            print(f"   {net:10s} {len(s):4d} {s.mean():9.4f} {s.std():8.4f}")

        # --- il contrasto pre-specificato
        if TARGET in stats_by_net and all(x in stats_by_net for x in SENSORY):
            dmn = stats_by_net[TARGET]
            sen = np.concatenate([stats_by_net[x] for x in SENSORY])
            diff = dmn.mean() - sen.mean()
            pooled = np.sqrt((dmn.var(ddof=1)*(len(dmn)-1) +
                              sen.var(ddof=1)*(len(sen)-1)) /
                             (len(dmn)+len(sen)-2))
            print(f"\n   CONTRASTO PRE-SPECIFICATO  Default - (Vis + SomMot)")
            print(f"      Default  {dmn.mean():+.4f}  (n={len(dmn)})")
            print(f"      Sensorie {sen.mean():+.4f}  (n={len(sen)})")
            print(f"      differenza {diff:+.4f}   d di Cohen {diff/pooled:+.3f}")
            if name == "grezza":
                rng = abs(pv.max() - pv.min())
                print(f"      ampiezza relativa: {abs(diff)/abs(pv.mean()):.1%} "
                      f"del livello medio, {abs(diff)/rng:.1%} del range")
            results[f"{name}_diff"] = diff

            # --- spin test sul contrasto
            if nulls is not None and args.n_perm > 0:
                try:
                    vv = np.where(np.isfinite(v), v, 0.0)
                    rot = nulls.alexander_bloch(vv, atlas="fsLR", density="32k",
                                                n_perm=args.n_perm, seed=args.seed)
                    null_d = []
                    for i in range(rot.shape[1]):
                        pvi, _ = parcel_means(np.where(np.isfinite(v),
                                                       rot[:, i], np.nan))
                        if len(pvi) != len(pn):
                            continue
                        a = pvi[pn == TARGET].mean()
                        b = np.concatenate([pvi[pn == x] for x in SENSORY]).mean()
                        null_d.append(a - b)
                    null_d = np.array(null_d)
                    if len(null_d) > 10:
                        p = (np.abs(null_d) >= abs(diff)).mean()
                        print(f"      spin test: p = {p:.4f} "
                              f"({len(null_d)} rotazioni valide, "
                              f"null sd {null_d.std():.4f})")
                        results[f"{name}_p"] = p
                    else:
                        print("      spin test: troppe rotazioni scartate")
                except Exception as e:
                    print(f"      spin test fallito: {type(e).__name__}: {e}")

        # --- ranking completo delle reti
        order = sorted(stats_by_net, key=lambda k: -stats_by_net[k].mean())
        print(f"\n   ranking (alta -> bassa partecipazione): {' > '.join(order)}")
        results[f"{name}_ranking"] = np.array(order)
        for net, s in stats_by_net.items():
            results[f"{name}_net_{net}"] = s

    np.savez_compressed(os.path.join(args.outdir, "yeo_test.npz"),
                        **{k: np.asarray(v) for k, v in results.items()},
                        min_cover=args.min_cover)
    print(f"\nsalvato in {args.outdir}/yeo_test.npz")
    print("\nRicorda: l'ipotesi pre-specificata era Default > sensorie. Se l'esito")
    print("e' di segno opposto va riportato come predizione fallita, non")
    print("riformulato a posteriori.")


if __name__ == "__main__":
    main()
