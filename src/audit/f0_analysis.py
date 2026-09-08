#!/usr/bin/env python
"""
f0_analysis.py — decompone la media latente m_s rispetto alle sonde e testa
quale componente porta il guadagno di retrieval.

Nessun modello richiesto: lavora sui 622 .npz di latents_lag4 + probes.npz.

Domanda:  m_s = f(0) + m_s^bio  — quanto pesa ciascun termine, e quale dei due
serve al retrieval?

Uso:
    python f0_analysis.py \
        --latents $CS_CACHE/latents_lag4 \
        --probes  $CS_CACHE/f0/probes.npz \
        --out     $CS_CACHE/f0/report.txt
"""
import argparse
import glob
import os
import itertools
import numpy as np

N_TOK, D = 1456, 768
N_T, N_P = 4, 364          # blocchi temporali x patch spaziali


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cos(a, b):
    return float(np.dot(unit(np.ravel(a)), unit(np.ravel(b))))


def load_latents(d):
    files = sorted(glob.glob(os.path.join(d, "*.npz")))
    if not files:
        raise FileNotFoundError(f"nessun .npz in {d}")
    Z, meta = [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        e = np.asarray(z["patch_embeds"], dtype=np.float32)
        if e.shape != (N_TOK, D):
            raise ValueError(f"{f}: forma {e.shape}")
        Z.append(e)
        meta.append({k: (z[k].item() if z[k].ndim == 0 else z[k])
                     for k in z.files if k != "patch_embeds"})
    return np.stack(Z), meta          # [S, 1456, 768]


def retrieval(Z, subj, clip, metric_dim=None):
    """
    Retrieval inter-soggetto same-clip. Ritorna (top1, pct_rank).
    Z : [S, F] gia' appiattito e trasformato.
    """
    subj, clip = np.asarray(subj), np.asarray(clip)
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    subs = np.unique(subj)
    t1, pr = [], []
    for a, b in itertools.permutations(subs, 2):
        ia, ib = np.where(subj == a)[0], np.where(subj == b)[0]
        ca, cb = clip[ia], clip[ib]
        common = np.intersect1d(ca, cb)
        if len(common) < 5:
            continue
        ia = ia[np.isin(ca, common)][np.argsort(ca[np.isin(ca, common)])]
        ib = ib[np.isin(cb, common)][np.argsort(cb[np.isin(cb, common)])]
        S = Zn[ia] @ Zn[ib].T                      # [n, n]
        n = S.shape[0]
        order = np.argsort(-S, axis=1)
        rank = np.argmax(order == np.arange(n)[:, None], axis=1)
        t1.append((rank == 0).mean())
        pr.append((rank / (n - 1)).mean())
    return float(np.mean(t1)), float(np.mean(pr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref", default="meanmap",
                    help="sonda di riferimento per f(0): meanmap o zeros")
    ap.add_argument("--time-major", action="store_true", default=True,
                    help="reshape 1456 -> (4, 364); disattiva se space-major")
    args = ap.parse_args()

    lines = []
    def P(s=""):
        print(s)
        lines.append(s)

    Z, meta = load_latents(args.latents)
    subj = np.array([m["subject"] for m in meta])
    clip = np.array([m["clip_id"] for m in meta])
    P(f"latenti: {Z.shape[0]}  soggetti: {sorted(set(subj))}  "
      f"clip distinte: {len(set(clip))}")
    for s in sorted(set(subj)):
        P(f"   {s}: {(subj == s).sum()} clip")

    probes = np.load(args.probes)
    if args.ref not in probes:
        raise SystemExit(f"sonda '{args.ref}' assente: {list(probes.keys())}")

    # ---- medie per soggetto ------------------------------------------------
    subs = sorted(set(subj))
    mB = {s: Z[subj == s].mean(axis=(0, 1)) for s in subs}          # [768]
    mA = {s: Z[subj == s].mean(axis=0) for s in subs}               # [1456, 768]
    resid_norm = {s: np.linalg.norm(Z[subj == s] - mB[s], axis=2).mean()
                  for s in subs}

    P("\n=== 1. RIPRODUZIONE DEI NUMERI NOTI ===")
    for s in subs:
        P(f"   {s}: ||m_B|| = {np.linalg.norm(mB[s]):8.3f}   "
          f"rapporto media/residuo = {np.linalg.norm(mB[s])/resid_norm[s]:.3f}"
          f"   (atteso ~1.45)")
    cs = [cos(mB[a], mB[b]) for a, b in itertools.combinations(subs, 2)]
    P(f"   coseno inter-soggetto di m_B: {min(cs):.4f} - {max(cs):.4f}"
      f"   (atteso 0.97-0.99)")

    # ---- confronto con le sonde -------------------------------------------
    P("\n=== 2. CONFRONTO CON LE SONDE  <-- IL TEST ===")
    probe_B = {}
    for name in ("zeros", "gaussian", "shuffled", "real"):
        if name not in probes:
            continue
        probe_B[name] = np.asarray(probes[name]).mean(axis=0)        # [768]
    for name, p in probe_B.items():
        cc = [cos(mB[s], p) for s in subs]
        ratio = np.linalg.norm(p) / np.mean([np.linalg.norm(mB[s]) for s in subs])
        P(f"   cos(m_B, {name:9s}) = {min(cc):.4f} - {max(cc):.4f}"
          f"    ||{name}||/||m_B|| = {ratio:.3f}")
    P("\n   Lettura:")
    P("     cos(m_B, zeros/meanmap) alto  -> m_B e' bias di rete f(0).")
    P("                                      L'interpretazione 'stato cerebrale' cade.")
    P("     cos(m_B, gaussian) alto ma zeros basso -> e' il termine di curvatura:")
    P("                                      dipende dalla varianza dell'input,")
    P("                                      non dal contenuto. Ancora niente biologia.")
    P("     tutti bassi                   -> m_B non e' spiegato dall'architettura.")
    P("                                      Il claim biologico resta in gioco.")

    # ---- decomposizione ----------------------------------------------------
    u0 = unit(probe_B[args.ref])
    P(f"\n=== 3. DECOMPOSIZIONE  m_s = proj_f0 + residuo   (ref = {args.ref}) ===")
    res = {}
    for s in subs:
        proj = np.dot(mB[s], u0) * u0
        r = mB[s] - proj
        res[s] = r
        frac = np.linalg.norm(r) / np.linalg.norm(mB[s])
        P(f"   {s}: ||residuo||/||m_B|| = {frac:.4f}   "
          f"varianza spiegata da f(0) = {1 - frac**2:.3f}")

    cr = [cos(res[a], res[b]) for a, b in itertools.combinations(subs, 2)]
    P(f"\n   coseno inter-soggetto del RESIDUO: {min(cr):.4f} - {max(cr):.4f}")
    P("   Se resta alto (>0.9), il residuo e' una struttura condivisa tra individui")
    P("   e non spiegata dall'architettura: e' li' che vive il claim biologico.")
    P("   Se crolla, la condivisione 0.97-0.99 era interamente f(0).")

    # ---- stabilita' split-half del residuo ---------------------------------
    P("\n=== 4. STABILITA' SPLIT-HALF (per soggetto) ===")
    rng = np.random.default_rng(0)
    for s in subs:
        idx = np.where(subj == s)[0]
        rng.shuffle(idx)
        h1, h2 = idx[:len(idx)//2], idx[len(idx)//2:]
        m1, m2 = Z[h1].mean(axis=(0, 1)), Z[h2].mean(axis=(0, 1))
        r1 = m1 - np.dot(m1, u0) * u0
        r2 = m2 - np.dot(m2, u0) * u0
        P(f"   {s}: m_B  r = {np.corrcoef(m1, m2)[0,1]:.4f}   "
          f"residuo  r = {np.corrcoef(r1, r2)[0,1]:.4f}")
    P("   La stabilita' di m_B e' attesa altissima (stima di una costante).")
    P("   Quella del residuo e' il numero informativo.")

    # ---- topografia di partecipazione --------------------------------------
    P("\n=== 5. TOPOGRAFIA: partecipazione per patch ===")
    for s in subs[:1]:
        MA = mA[s].reshape(N_T, N_P, D) if args.time_major else \
             mA[s].reshape(N_P, N_T, D).transpose(1, 0, 2)
        MA = MA.mean(axis=0)                                   # [364, 768]
        p_f0 = MA @ u0
        ur = unit(res[s])
        p_res = MA @ ur
        P(f"   {s}: partecipazione a f(0)   media {p_f0.mean():.3f}  "
          f"sd {p_f0.std():.3f}  CV {abs(p_f0.std()/p_f0.mean()):.3f}")
        P(f"       partecipazione al residuo media {p_res.mean():.3f}  "
          f"sd {p_res.std():.3f}  CV {abs(p_res.std()/p_res.mean()):.3f}")
        np.save(os.path.join(os.path.dirname(args.probes),
                             f"participation_{s}.npy"),
                np.stack([p_f0, p_res]))
    P("   Una mappa piatta (CV bassa) non ha topografia -> niente da mappare su Yeo.")
    P("   Salvate le mappe [2, 364] per la proiezione sulla flat map.")

    # ---- retrieval: quale componente serve? --------------------------------
    P("\n=== 6. RETRIEVAL: quale centering porta il guadagno? ===")
    Zf = Z.reshape(Z.shape[0], -1)
    conds = {}
    conds["raw"] = Zf.copy()
    Zc = Z.copy()
    for s in subs:
        Zc[subj == s] -= mB[s][None, None, :]
    conds["centering B (m_s)"] = Zc.reshape(Z.shape[0], -1)
    Zf0 = Z - (Z @ u0)[..., None] * u0[None, None, :]
    conds["solo direzione f(0)"] = Zf0.reshape(Z.shape[0], -1)
    Zr = Z.copy()
    for s in subs:
        ur = unit(res[s])
        Zr[subj == s] -= (Zr[subj == s] @ ur)[..., None] * ur[None, None, :]
    conds["solo residuo biologico"] = Zr.reshape(Z.shape[0], -1)

    for name, X in conds.items():
        t1, pr = retrieval(X, subj, clip)
        P(f"   {name:24s}  top-1 = {t1:.4f}   pct rank = {pr:.4f}")
    P("\n   Se 'solo direzione f(0)' recupera quasi tutto il guadagno di")
    P("   'centering B', allora il +40% e' rimozione di anisotropia architetturale,")
    P("   non separazione di uno stato cerebrale.")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\nreport salvato in {args.out}")


if __name__ == "__main__":
    main()
