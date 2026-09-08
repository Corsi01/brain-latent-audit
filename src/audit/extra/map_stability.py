#!/usr/bin/env python3
"""
map_stability.py — quale mappa portare sulla superficie corticale?

La proiezione non normalizzata confonde due quantita' distinte:

    proj_i = || A_i ||  x  cos( A_i , u_res )
             ^ ampiezza     ^ allineamento
             locale          alla direzione condivisa

dove  A_i = M_A[i] - f0[i]  e  u_res = unit(m_B - proiezione su f(0)).

L'ampiezza e' in buona parte una proprieta' banale della flat map (posizione
del patch, densita' di vertici, vicinanza al bordo). L'allineamento e' cio' che
"partecipazione" dovrebbe significare: e' invariante alla scala locale.

Se proj correla ~1 con norm, la mappa e' essenzialmente una mappa di ampiezza e
va sostituita con il coseno prima di costruire la catena verso i vertici.

Ruoli dei due centering, per chiarezza:
    centering B (m_B, [768])       -> definisce la DIREZIONE su cui si proietta
    centering A (M_A, [364, 768])  -> e' l'oggetto MAPPATO
    f(0) per token                 -> sottratto patch-wise, ha una sua topografia
(sottrarre A da M_A darebbe identicamente zero: non e' un'opzione)

Uso:
    python map_stability.py --latents $CS_CACHE/latents_lag4 \
                            --probes  $CS_CACHE/f0/probes.npz
"""
import argparse
import glob
import itertools
import os

import numpy as np

N_TOK, D = 1456, 768
N_T, N_P = 4, 364


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def to_grid(x, time_major=True):
    """[1456, 768] -> [364, 768] mediando sui blocchi temporali."""
    if time_major:
        return x.reshape(N_T, N_P, D).mean(axis=0)
    return x.reshape(N_P, N_T, D).mean(axis=1)


def cv(m):
    return abs(float(m.std() / m.mean())) if m.mean() != 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    cache = os.environ.get("CS_CACHE", ".")
    ap.add_argument("--latents", default=os.path.join(cache, "latents_lag4"))
    ap.add_argument("--probes", default=os.path.join(cache, "f0", "probes.npz"))
    ap.add_argument("--outdir", default=os.path.join(cache, "f0", "maps"))
    ap.add_argument("--space-major", action="store_true")
    ap.add_argument("--n-splits", type=int, default=20)
    args = ap.parse_args()

    tm = not args.space_major
    os.makedirs(args.outdir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.latents, "*.npz")))
    if not files:
        raise SystemExit(f"nessun latente in {args.latents}")
    Z, subj = [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        Z.append(np.asarray(z["patch_embeds"], np.float32))
        subj.append(str(z["subject"]))
    Z, subj = np.stack(Z), np.array(subj)
    subs = sorted(set(subj))
    print(f"{len(Z)} latenti, {len(subs)} soggetti\n")

    f0_grid = to_grid(np.asarray(np.load(args.probes)["zeros"], np.float32), tm)
    u0 = unit(f0_grid.mean(axis=0))

    KINDS = ["proj", "norm", "cos", "proj_raw"]

    def maps_of(Zsub):
        """Ritorna dict di mappe [364]."""
        MA = to_grid(Zsub.mean(axis=0), tm)
        mB = Zsub.mean(axis=(0, 1))
        u_res = unit(mB - np.dot(mB, u0) * u0)
        A = MA - f0_grid
        nrm = np.linalg.norm(A, axis=1)
        prj = A @ u_res
        return {
            "proj": prj,                       # ampiezza x allineamento
            "norm": nrm,                       # sola ampiezza locale
            "cos": prj / (nrm + 1e-8),         # solo allineamento
            "proj_raw": MA @ u_res,            # senza sottrazione di f(0)
        }

    rng = np.random.default_rng(0)
    M = {s: maps_of(Z[subj == s]) for s in subs}

    # ------------------------------------------------------------------ 1
    print("=== 1. STABILITA' SPLIT-HALF, per tipo di mappa ===")
    print(f"   {'soggetto':10s} " + " ".join(f"{k:>10s}" for k in KINDS))
    for s in subs:
        idx = np.where(subj == s)[0]
        acc = {k: [] for k in KINDS}
        for _ in range(args.n_splits):
            p = rng.permutation(idx)
            m1 = maps_of(Z[p[:len(p) // 2]])
            m2 = maps_of(Z[p[len(p) // 2:]])
            for k in KINDS:
                acc[k].append(np.corrcoef(m1[k], m2[k])[0, 1])
        print(f"   {s:10s} " + " ".join(f"{np.mean(acc[k]):10.4f}" for k in KINDS))
    print()

    # ------------------------------------------------------------------ 2
    print("=== 2. REPLICA TRA SOGGETTI ===")
    for k in KINDS:
        cc = [np.corrcoef(M[a][k], M[b][k])[0, 1]
              for a, b in itertools.combinations(subs, 2)]
        print(f"   {k:10s} r = {min(cc):.4f} - {max(cc):.4f}")
    print()

    # ------------------------------------------------------------------ 3
    print("=== 3. LA DIAGNOSI: proj e' solo ampiezza? ===")
    for s in subs:
        m = M[s]
        r_pn = np.corrcoef(m["proj"], m["norm"])[0, 1]
        r_pc = np.corrcoef(m["proj"], m["cos"])[0, 1]
        r_nc = np.corrcoef(m["norm"], m["cos"])[0, 1]
        print(f"   {s}: r(proj,norm) = {r_pn:6.3f}   r(proj,cos) = {r_pc:6.3f}"
              f"   r(norm,cos) = {r_nc:6.3f}")
    print("   r(proj,norm) vicino a 1 -> proj e' una mappa di ampiezza:")
    print("                              porta `cos` sulla superficie, non `proj`.")
    print("   r(proj,cos) piu' alto    -> proj e' gia' dominata dall'allineamento.\n")

    # ------------------------------------------------------------------ 4
    print("=== 4. AMPIEZZA DELLA MODULAZIONE ===")
    for s in subs:
        m = M[s]
        print(f"   {s}: " + "   ".join(
            f"{k} media {m[k].mean():8.3f} CV {cv(m[k]):.3f}" for k in ("proj", "cos")))
    c = M[subs[0]]["cos"]
    print(f"\n   coseno: min {c.min():.3f}  max {c.max():.3f}  "
          f"(range {c.max()-c.min():.3f})")
    print("   Se tutti i coseni sono alti e simili, la partecipazione e' globale:")
    print("   risultato di per se', ma argomento contro una lettura rete-specifica.")
    f0m = f0_grid @ u0
    print(f"   topografia di f(0): CV = {cv(f0m):.3f}\n")

    out = os.path.join(args.outdir, "participation_maps.npz")
    np.savez_compressed(
        out,
        subjects=np.array(subs), f0_map=f0m,
        **{f"{k}_{s}": M[s][k] for s in subs for k in KINDS},
    )
    print(f"mappe [364] salvate in {out}")
    print("chiavi: " + ", ".join(f"{k}_<soggetto>" for k in KINDS))


if __name__ == "__main__":
    main()