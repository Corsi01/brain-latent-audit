#!/usr/bin/env python3
"""
randinit_compare.py — confronta latenti addestrati vs. encoder a pesi casuali.

Quattro domande, in ordine di importanza:

  Q1  La direzione media m_B e' la stessa?      cos(m_B^trained, m_B^random)
      alto -> m_B e' statistica dell'input filtrata dall'architettura.
      basso -> il pretraining costruisce quella direzione.

  Q2  Il residuo condiviso tra soggetti sopravvive senza pretraining?
      Se anche la rete casuale da' coseno inter-soggetto ~0.98, la
      condivisione e' una proprieta' dei dati, non del modello.

  Q3  Il centering serve anche senza pretraining?
      Se il +40% c'e' anche a pesi casuali, e' un effetto geometrico
      generico, non una separazione appresa.

  Q4  Il retrieval regge senza pretraining?
      Controllo per l'INTERO progetto: una proiezione casuale conserva in
      parte le distanze, quindi un po' di retrieval e' atteso. La domanda e'
      quanto del 26x sopravvive.

Uso:
    python randinit_compare.py --trained $CS_CACHE/latents_lag4 \
                               --random  $CS_CACHE/latents_lag4_permuted0
"""
import argparse
import glob
import itertools
import os
import numpy as np

N_TOK, D = 1456, 768


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cos(a, b):
    return float(np.dot(unit(np.ravel(a)), unit(np.ravel(b))))


def load(d):
    files = sorted(glob.glob(os.path.join(d, "*.npz")))
    if not files:
        raise FileNotFoundError(d)
    Z, subj, clip = [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        Z.append(np.asarray(z["patch_embeds"], np.float32))
        subj.append(str(z["subject"]))
        clip.append(str(z["clip_id"]))
    return np.stack(Z), np.array(subj), np.array(clip)


def retrieval(X, subj, clip):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    t1, pr = [], []
    for a, b in itertools.permutations(sorted(set(subj)), 2):
        ia, ib = np.where(subj == a)[0], np.where(subj == b)[0]
        common = np.intersect1d(clip[ia], clip[ib])
        if len(common) < 5:
            continue
        ia = ia[np.isin(clip[ia], common)]
        ia = ia[np.argsort(clip[ia])]
        ib = ib[np.isin(clip[ib], common)]
        ib = ib[np.argsort(clip[ib])]
        S = Xn[ia] @ Xn[ib].T
        n = S.shape[0]
        rank = np.argmax(np.argsort(-S, axis=1) == np.arange(n)[:, None], axis=1)
        t1.append((rank == 0).mean())
        pr.append((rank / (n - 1)).mean())
    return float(np.mean(t1)), float(np.mean(pr)), n


def summarize(tag, Z, subj, clip, out):
    subs = sorted(set(subj))
    mB = {s: Z[subj == s].mean(axis=(0, 1)) for s in subs}
    resid = {s: np.linalg.norm(Z[subj == s] - mB[s], axis=2).mean() for s in subs}
    ratios = [np.linalg.norm(mB[s]) / resid[s] for s in subs]
    cs = [cos(mB[a], mB[b]) for a, b in itertools.combinations(subs, 2)]
    out(f"\n--- {tag} ---")
    out(f"   ||m_B||/residuo: {min(ratios):.3f} - {max(ratios):.3f}")
    out(f"   coseno inter-soggetto di m_B: {min(cs):.4f} - {max(cs):.4f}")

    raw = Z.reshape(len(Z), -1)
    Zc = Z.copy()
    for s in subs:
        Zc[subj == s] -= mB[s][None, None, :]
    t1r, prr, n = retrieval(raw, subj, clip)
    t1c, prc, _ = retrieval(Zc.reshape(len(Z), -1), subj, clip)
    gain = (t1c / t1r - 1) * 100 if t1r > 0 else float("nan")
    out(f"   retrieval raw      top-1 {t1r:.4f}  ({t1r*n:.1f}x caso)  "
        f"pct {prr:.4f}")
    out(f"   retrieval centrato top-1 {t1c:.4f}  ({t1c*n:.1f}x caso)  "
        f"pct {prc:.4f}")
    out(f"   guadagno dal centering: {gain:+.1f}%")
    return mB, subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trained", required=True)
    ap.add_argument("--random", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lines = []
    def P(s=""):
        print(s)
        lines.append(s)

    Zt, st, ct = load(args.trained)
    Zr, sr, cr = load(args.random)
    P(f"addestrato: {len(Zt)} latenti   casuale: {len(Zr)} latenti")
    if not (np.array_equal(st, sr) and np.array_equal(ct, cr)):
        P("   !! gli insiemi (soggetto, clip) NON coincidono: confronto non pulito")

    mBt, subs = summarize("ENCODER ADDESTRATO", Zt, st, ct, P)
    mBr, _ = summarize("ENCODER A PESI CASUALI", Zr, sr, cr, P)

    P("\n=== Q1. E' la stessa direzione? ===")
    cc = [cos(mBt[s], mBr[s]) for s in subs]
    P(f"   cos(m_B addestrato, m_B casuale): {min(cc):.4f} - {max(cc):.4f}")
    P("   alto  -> m_B e' statistica dell'input filtrata dall'architettura")
    P("   basso -> il pretraining costruisce quella direzione")

    P("\n=== Lettura complessiva ===")
    P("   Se la rete casuale riproduce coseno inter-soggetto, guadagno da")
    P("   centering e retrieval, allora nulla di quanto osservato richiede il")
    P("   pretraining, e il claim va riformulato come proprieta' dei dati.")
    P("   Se li riproduce solo in parte, la differenza quantifica il")
    P("   contributo del pretraining ed e' il risultato del paper.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        open(args.out, "w").write("\n".join(lines) + "\n")
        print(f"\nreport in {args.out}")


if __name__ == "__main__":
    main()
