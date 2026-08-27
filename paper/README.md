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
- `sections/appendix.tex`: recipes, padded shapes, protocol, visualizations, and
  the submission-critical experiment list.
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
and one final test evaluation for seeds 3407, 3408, and 3409. PhysioNet-MI uses
`P=1`, so its input mapping is unaffected. ConvNeXt probes are exploratory and
are not included in the headline table.

Before submission:

1. reproduce CBraMod on identical splits and seeds;
2. rerun the formal comparison table from a prospectively frozen protocol;
3. add random-init and frozen-transfer controls;
4. compile using the official ICLR template and check the page limit.

## Result policy

The main table must contain only validation-selected checkpoints with one final
test evaluation per seed. Exploratory `test_each_epoch=true` peaks must never be
copied into the formal table.
