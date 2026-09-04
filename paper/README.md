# EEG-Vision ICLR paper draft

This directory contains a modular LaTeX manuscript titled *Is Vision
Pretraining a Free Lunch for EEG Decoding? Lossless Temporal Folding Across 11
Benchmarks*. It studies when off-the-shelf visual weights become a strong EEG
baseline through lossless geometry alignment.

## Structure

- `main.tex`: entrypoint and ICLR-style fallback setup.
- `results.tex`: headline numerical macros mirrored from
  `experiments/PHASE_FOLD_RESULTS.md`, the repository source of truth.
- `sections/00_abstract.tex` through `sections/08_conclusion.tex`: main paper.
- `sections/appendix.tex`: recipes, padded shapes, protocol, per-seed values,
  and additional visualizations.
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
table now contains all 11 datasets, each with validation-selected checkpoints
and one final test evaluation per seed. CHB-MIT, SHU-MI, ISRUC, and TUEV have
been replaced by their locked five-seed runs (42--46); datasets still awaiting
five-seed replacement retain the canonical 3407--3409 sweep. The headline
table also includes published BIOT, LaBraM-Base, and CBraMod references for all
11 datasets, plus REVE-Base on its eight overlapping tasks. The
CBraMod TUAB and TUEV cells use the target-corpus-excluded controls; TUEV and
REVE--ISRUC are marked with comparability caveats. PhysioNet-MI uses `P=1`, so its
input mapping is unaffected. ConvNeXt probes are exploratory and are not
included in the headline table.

Appendix C mirrors REVE's detailed-results organization: all 11 datasets have
individual two-metric tables containing the shared supervised architecture
suite, EEG foundation models, REVE where available, and the finalized local B0
result.

The local subject/trial split definitions reproduce the partitions reported by
CBraMod; the overlapping REVE benchmark values are included as an additional
published comparison. The current headline table is the completed unified
all-BF16 min-64 reproduction. Fold-factor selection chooses the smallest valid `P`
that reaches at least 64 folded rows; SEED-V's native 62 rows are treated as
close enough and left at `P=1`. CHB-MIT, SHU-MI, ISRUC, and TUEV have completed
their locked five-seed reruns.
SEED-V remains a documented limitation because this prespecified `P=1` result
underperforms the historical `P=8` recipe.

Before submission:

1. reproduce CBraMod under the identical split definitions and seeds;
2. rerun the formal comparison table from a prospectively frozen protocol;
3. add random-init and frozen-transfer controls;
4. compile using the official ICLR template and check the page limit.

## Result policy

The main table must contain only validation-selected checkpoints with one final
test evaluation per seed. Exploratory `test_each_epoch=true` peaks must never be
copied into the formal table.
