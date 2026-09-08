# Auditing the latent space of a brain foundation model

Code for a validation-and-audit study of **CortexMAE-F**, a self-supervised
foundation model for fMRI, on CNeuroMod / Algonauts 2025 (4 subjects,
622 clip–subject latents).

**Written report: [`report.pdf`](report.pdf)** — start there. The code exists
to make the report verifiable.

---

## What this is

A stimulus-to-brain encoding model can be regularised by predicting, alongside
raw voxels, the latent of a frozen foundation model. That only makes sense if
the latent is driven by the stimulus, and reconstruction quality does not
establish this: roughly 80% of BOLD variance during film watching is shared
with rest, so a model can reconstruct the brain well while encoding almost
nothing about the film.

The first half of this work validates the latent as a target through
inter-subject clip identification. The second half follows up an anomaly that
surfaced during that validation: subtracting the per-subject mean latent
raises identification by 40%, while standardising lowers it. That asymmetry
indicates an additive offset rather than a scale problem. The subtracted
vector is the dominant component of the representation and nearly identical
across subjects, which suggests it encodes the brain's default,
stimulus-independent state — but the same observations are predicted by the
architectural anisotropy known in transformer representations since 2018.

The audit separates the two readings.

## What was found

**The latent is stimulus-locked.** Inter-subject clip identification reaches
26.6× chance, 39 SD above a permutation null, with accuracy scaling with
stimulus drive as an evoked response should.

**The mean direction is not architectural bias.** The encoder's response to a
null input explains 3% of the mean's variance, against 39–65% in
weight-randomised networks of the same architecture, and the trained mean
direction is orthogonal to any random network's. Removing the bias direction
produces no retrieval gain; removing the residual produces all of it.

**Pretraining attenuates anisotropy rather than creating it.** The
mean-to-residual ratio falls from 0.91 and 1.22 in untrained networks to 0.65
in the trained one. The dominant direction is already present at
initialisation and is stronger there; training reduces it and rotates it
toward a data-dependent direction.

**What pretraining buys is accessibility, not information.** Running the same
identification protocol on the input flat maps gives 10.3× chance, against
37.3× for the centered latent. The encoder makes existing information 3.6×
more linearly accessible without adding any.

**56% of a plausible cortical map was grid geometry.** Participation in the
mean direction correlated at −0.72 with distance from the flat-map border:
border patches contain few vertices, so their mean is averaged over less
heterogeneous tissue. This is an aggregation artefact, and any group
projecting flat-map latents onto cortex would hit it.

**The pre-specified anatomical test failed.** Participation was not higher in
the default mode network than in sensory networks; the contrast came out with
the opposite sign and was not significant under a spin test.

**The failure was ill-posed.** The mean is not a direction but a
low-dimensional subspace with a flat spectrum, whose components relate to
cortical anatomy in opposite ways and cancel under projection onto a single
axis. Decomposed, one component is grid geometry, one tracks visual evoked
response, and one aligns with the principal connectivity gradient (r = 0.652,
spin-test p = 0.001, replicating across all six subject splits). This last
result is post-hoc and is reported as exploratory.

## Repository layout

```
src/pipeline/     stimulus characterisation, clip selection, latent extraction
src/validation/   is the latent stimulus-locked, and how should it be preprocessed
src/audit/        is the mean direction learned structure or architectural bias
jobs/             SLURM submission scripts
```

Scripts at the top level of `validation/` and `audit/` correspond to results
in the report. Each `extra/` subdirectory holds secondary analyses — parameter
sweeps, alternative representations, infrastructure for the patch-to-cortex
chain — and has its own README.

Every script is single-purpose, takes command-line arguments, and carries a
docstring stating the question it answers and why its controls are there. The
docstrings are part of the argument, not decoration.

## Running this

**This code does not run without data, by design.** The fMRI is governed by
the CNeuroMod usage agreement and is not redistributed here, and neither are
model weights or atlases.

External dependencies to obtain separately:

| What | Where | Note |
|---|---|---|
| CortexMAE weights | [MedARC-AI/CortexMAE](https://github.com/MedARC-AI/CortexMAE) | not on PyPI; install from source, own licence |
| CNeuroMod / Algonauts 2025 | [algonauts.csail.mit.edu](https://algonauts.csail.mit.edu/) | usage agreement required |
| Schaefer2018 fsLR-32k parcellation | [ThomasYeoLab/CBIG](https://github.com/ThomasYeoLab/CBIG) | `Parcellations/HCP/fslr32k/cifti` |
| neuromaps annotations | fetched on first use | requires network access |

```bash
cp env.example.sh env.sh    # fill in your paths and account
source env.sh
pip install -r requirements.txt
```

Typical order: `pipeline/` to build the clip manifest and extract latents,
then `validation/`, then `audit/`. On a cluster without network access on
compute nodes, atlases and model weights must be cached from a login node
first.

## Notes and limitations

The numbers in the report were produced by this code as it stands.

Four subjects, one model, one dataset. The subspace decomposition is post-hoc:
it is a second analysis of the same data, motivated by the failure of the
primary test, and the report says so. Bonferroni correction and cross-subject
replication protect it from the simplest form of that objection, but it is
exploratory until replicated on a second model.

One earlier claim was withdrawn during this work: that raw voxels lack a
comparably stable mean. At 622 samples in 1.24M dimensions that is a sampling
confound, not evidence of absence.

## License

<!-- MIT is the usual choice for research code. Check CortexMAE's licence
first, since part of this code interacts with that model. -->
