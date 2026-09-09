# EEG-Vision ICLR paper draft

This directory contains a modular LaTeX manuscript titled *Is Vision
Pretraining a Free Lunch for EEG Decoding? Lossless Temporal Folding Across 12
Benchmarks*. It studies when off-the-shelf visual weights become a strong EEG
baseline through lossless geometry alignment.

## Structure

- `main.tex`: entrypoint and ICLR-style fallback setup.
- `results.tex`: headline numerical macros mirrored from the finalized YAML
  configs; those configs are the source of truth.
- `sections/00_abstract.tex` through `sections/08_conclusion.tex`: main paper.
- `sections/appendix.tex`: recipes, padded shapes, protocol, backbone-head
  details, and additional visualizations.
- `sections/appendix_detailed_results.tex`: Appendix C-style per-dataset
  comparisons with supervised architectures and EEG foundation models.
- `references.bib`: bibliography.
- `ICLR_SUBMISSION_PLAN.md`: practical plan for turning the draft into a
  defensible main-track submission.

## Compile

The machine currently has no TeX distribution. On a machine with LaTeX:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The source automatically uses `iclr2027_conference.sty` if that official style
file is placed in `paper/`; otherwise it uses a normal article layout for local
editing. Do not upload an invented or unofficial ICLR style file.

## Current result status

The adapter was corrected on 25 August 2026 from contiguous-chunk folding to
the intended phase-interleaved permutation. The finalized EfficientNet-B0
table now contains all 12 datasets, each with validation-selected checkpoints
and one final test evaluation per seed. All formal EfficientNet-B0 datasets use
the five-seed recipes recorded in their YAML configs; those configs are the
source of truth for means, standard deviations, and per-seed results. The headline
paper tables retain the published supervised baselines, BIOT, LaBraM, CBraMod,
and REVE references, plus finalized local B0, ConvNeXt-Tiny, and ViT-Small rows
where available. Binary tasks use BA,
PR-AUC, and ROC-AUC; multiclass tasks use BA, Cohen's $\kappa$, and weighted
F1. The
CBraMod TUAB and TUEV cells use the target-corpus-excluded controls; TUEV and
REVE--ISRUC are marked with comparability caveats. PhysioNet-MI uses `P=1`, so its
input mapping is unaffected. The ConvNeXt TUAB configuration remains pending
and is shown as `--`; other completed DINOv3 results are reported in the
cross-backbone appendix table.

Appendix C mirrors REVE's detailed-results organization: all 12 datasets have
individual three-metric tables containing the shared supervised architecture
suite, EEG foundation models, REVE where available, and the finalized local B0
result.

The local subject/trial split definitions reproduce the partitions reported by
CBraMod; the overlapping REVE benchmark values are included as an additional
published comparison. The current headline table is the completed unified
all-BF16 min-64 reproduction. Fold-factor selection chooses the smallest valid `P`
that reaches at least 64 folded rows; SEED-V's native 62 rows are treated as
close enough and left at `P=1`. CHB-MIT, SHU-MI, ISRUC, and TUEV have completed
their locked five-seed reruns.
SEED-V remains a documented limitation because its prespecified `P=1` result
is below the strongest EEG foundation baselines.

Before submission:

1. compile using the official ICLR template and check the page limit;
2. add any remaining backbone configurations only after validation protocols
   are frozen;
3. consider frozen-transfer controls as a separate future experiment.

## Result policy

The main table must contain only validation-selected checkpoints with one final
test evaluation per seed. Exploratory `test_each_epoch=true` peaks must never be
copied into the formal table.
