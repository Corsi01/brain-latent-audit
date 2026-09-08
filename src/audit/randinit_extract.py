#!/usr/bin/env python3
"""
randinit_extract.py — ri-estrae i latenti con l'encoder a PESI CASUALI.

Perche': m_B non e' spiegato da f(0) ne' dal termine di curvatura. Resta una
sola spiegazione alternativa al pretraining: che m_B sia il prodotto della
statistica spaziale dei dati cerebrali passata attraverso una qualunque rete
con quella architettura, addestrata o no.

Questo script tiene fisso TUTTO (architettura, maschera, preprocessing, clip,
soggetti) e cambia solo i pesi. Il confronto e' quindi pulito per costruzione.

Due varianti di randomizzazione, riportate entrambe perche' hanno debolezze
opposte:

  permuted  permuta le entrate di ogni tensore di pesi. Conserva ESATTAMENTE
            la distribuzione marginale dei pesi appresi (scala, sparsita',
            spettro marginale) e distrugge la struttura. E' il controllo piu'
            severo: non puo' essere liquidato come "rete mal scalata".

  fresh     re-inizializzazione con lo schema di init del modello, se lo si
            trova. Piu' convenzionale in letteratura, ma la scala puo'
            differire da quella post-training e alterare la propagazione.

Se le due varianti concordano, la conclusione e' robusta.

Uso:
    python randinit_extract.py --variant permuted --seed 0
    python randinit_extract.py --variant fresh    --seed 0
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

NUM_FRAMES = 16
FLAT_DIM = 77763


# --- identico a extract_latents.py / f0_probe.py ----------------------------

def encode_window(model, bold_raw, batch_size=8):
    bold_raw = np.ascontiguousarray(bold_raw, dtype=np.float32)
    assert bold_raw.shape == (NUM_FRAMES, FLAT_DIM), bold_raw.shape
    mu = bold_raw.mean(axis=0, keepdims=True)
    sd = bold_raw.std(axis=0, keepdims=True)
    bold = np.where(sd > 1e-6, (bold_raw - mu) / np.clip(sd, 1e-6, None), 0.0)
    sample = {
        "bold": torch.from_numpy(bold.astype(np.float16)),
        "mean": torch.from_numpy(mu.astype(np.float32).squeeze(0)),
        "std":  torch.from_numpy(sd.astype(np.float32).squeeze(0)),
        "tr":   1.0,
    }
    embeds = model.run_embedding(sample, tr=1.0, batch_size=batch_size)
    patch = embeds.patch_embeds.float().numpy()
    return (patch[0] if patch.shape[0] == 1 else patch).astype(np.float32)


# --- randomizzazione dei pesi ----------------------------------------------

def get_encoder(model):
    """run_embedding chiama self.model.encoder.forward_embedding(...)."""
    return model.model.encoder


def randomize_permuted(enc, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_par = n_el = 0
    for _, p in enc.named_parameters():
        if p.numel() < 2:
            continue
        flat = p.detach().cpu().reshape(-1)
        idx = torch.randperm(flat.numel(), generator=g)
        p.data.copy_(flat[idx].reshape(p.shape).to(p.device))
        n_par += 1
        n_el += p.numel()
    return f"permuted ({n_par} tensori, {n_el/1e6:.1f}M parametri)"


def randomize_fresh(enc, seed):
    torch.manual_seed(seed)
    # 1) metodo sul modulo
    for fn in ("initialize_weights", "init_weights"):
        if hasattr(enc, fn) and callable(getattr(enc, fn)):
            getattr(enc, fn)()
            return f"fresh (enc.{fn}())"
    # 2) funzione a livello di modulo, applicata ricorsivamente
    try:
        import cortex_mae.models_mae as mm
        for fn in ("_init_weights", "init_weights"):
            if hasattr(mm, fn) and callable(getattr(mm, fn)):
                enc.apply(getattr(mm, fn))
                return f"fresh (models_mae.{fn} via apply)"
    except Exception:
        pass
    # 3) fallback: reset_parameters standard di torch dove esiste
    n = 0
    for m in enc.modules():
        if hasattr(m, "reset_parameters") and callable(m.reset_parameters):
            m.reset_parameters()
            n += 1
    if n:
        return f"fresh (reset_parameters su {n} moduli)"
    return None


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    cache = os.environ.get("CS_CACHE", ".")
    ap.add_argument("--variant", choices=["permuted", "fresh"], default="permuted")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lag", type=int, default=4)
    ap.add_argument("--flat-dir", default=os.path.join(cache, "fmri_flat_1hz"))
    ap.add_argument("--ref-dir", default=None,
                    help="dir dei latenti addestrati; definisce quali clip estrarre")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--model", default="cortex_mae_flat")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    ref_dir = Path(args.ref_dir or os.path.join(cache, f"latents_lag{args.lag}"))
    out_dir = Path(args.out_dir or os.path.join(
        cache, f"latents_lag{args.lag}_{args.variant}{args.seed}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # quali (soggetto, clip) estrarre: esattamente quelli gia' presenti nei
    # latenti addestrati, cosi' i due insiemi sono identici per costruzione
    refs = sorted(ref_dir.glob("*.npz"))
    if not refs:
        sys.exit(f"nessun latente di riferimento in {ref_dir}")
    print(f"{len(refs)} coppie (soggetto, clip) da {ref_dir.name}")

    from cortex_mae import CortexMAE
    model = CortexMAE.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.set_device(device)
    enc = get_encoder(model)

    print(f"randomizzazione: {args.variant}  seed={args.seed}")
    if args.variant == "permuted":
        desc = randomize_permuted(enc, args.seed)
    else:
        desc = randomize_fresh(enc, args.seed)
        if desc is None:
            sys.exit("non trovo uno schema di init: usa --variant permuted")
    print(f"   {desc}")

    # prova che i pesi sono davvero cambiati
    with torch.no_grad():
        probe = encode_window(model, np.zeros((NUM_FRAMES, FLAT_DIM), np.float32),
                              args.batch_size)
    print(f"   norma del latente su input nullo: {np.linalg.norm(probe):.3f}")

    run_cache: dict[tuple, np.ndarray] = {}
    t_start = time.time()
    n_ok = 0
    for i, ref in enumerate(refs):
        z = np.load(ref, allow_pickle=True)
        sub = str(z["subject"]); run_id = str(z["run_id"])
        clip_id = str(z["clip_id"])
        t0, t1 = [int(v) for v in np.asarray(z["fmri_window"]).ravel()[:2]]

        out = out_dir / ref.name
        if out.exists():
            n_ok += 1
            continue

        key = (sub, run_id)
        if key not in run_cache:
            run_cache.clear()          # una run alla volta in RAM
            npy = Path(args.flat_dir) / f"{sub}_{run_id}.npy"
            run_cache[key] = np.load(npy).astype(np.float32)
        bold = run_cache[key][t0:t1]

        patch = encode_window(model, bold, args.batch_size)
        np.savez_compressed(
            out,
            patch_embeds=patch,
            subject=sub, clip_id=clip_id, run_id=run_id,
            film_id=str(z["film_id"]), content_label=str(z["content_label"]),
            start_s=int(z["start_s"]), lag=int(z["lag"]),
            fmri_window=np.array([t0, t1]),
            variant=args.variant, seed=args.seed,
        )
        n_ok += 1
        if n_ok % 50 == 0:
            el = time.time() - t_start
            print(f"   {n_ok}/{len(refs)}  [{el:.0f}s, "
                  f"stimato totale {el/n_ok*len(refs):.0f}s]", flush=True)

    print(f"\nfatto: {n_ok} latenti in {out_dir}")
    print(f"\nOra: python randinit_compare.py --trained {ref_dir} --random {out_dir}")


if __name__ == "__main__":
    main()
