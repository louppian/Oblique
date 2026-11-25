This directory contains code derived from [CARZero](https://github.com/laihaoran/CARZero)
(Lai et al., "CARZero: Cross-Attention Alignment for Radiology Zero-Shot Classification",
CVPR 2024), licensed under the Apache License, Version 2.0 (see `LICENSE` in this directory).

## Modifications made in this repository (Bi-MCQ)

- Added `models/BiMCQ_model.py` (`BiMCQModel`) implementing the Bi-MCQ dual-fusion
  (I2T/T2I) architecture on top of CARZero's encoders and fusion modules.
- Registered the new module in `models/__init__.py`.
- Added `load_BiMCQ` / `bimcq_classification` and related helpers in `CARZero.py` for
  building and running the Bi-MCQ model.

These changes are additive; the original CARZero encoder, fusion-module, and builder code
is otherwise unmodified. Per Apache License 2.0 Section 4(b), this file serves as the
notice that files in this directory have been changed from the original CARZero source.
