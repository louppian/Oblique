# Analysis mapping

All paths are relative to the repository root. Each experiment writes one CSV and one PNG with the same basename under `results/analysis`.

| Code | Purpose | Result basename |
|---|---|---|
| `nih_all_directions.py` | Common learned residual matrix; exact NF-versus-all-positive maximum cosine; disjoint-positive-label disease separation; three conditions | `nih_all_directions` |
| `nih_difference_three_loss.py` | Fixed images, text encoder training with original difference alignment and text NF constraint | `nih_difference_three_loss` |
| `m_alignment_examples.py` | Images and labels along estimated noise-to-mean direction | `m_alignment_examples` |
| `noise_neighbors.py` | Noise direction and image neighbors | `noise_neighbors` |

`carzero_embedding_distribution.ipynb` is retained as the original CARZero loader/reference notebook.

## Preserved prior experiments

`positive_residual_epistemic`, `positive_nf_constraint`, `positive_nf_convergence`, `positive_three_loss`, and `nih_pos_neg_three_loss` are archived under `code/old/analysis`; matching CSV/PNG files are under `results/old/analysis`. Checkpoints and caches retain their existing paths.

## Current all-direction experiment

Official NIH train_val/test lists: train 73,510, validation 13,014, test 25,596. Existing patient validation split retained. Original CARZero encoder frozen; a shared 766-dimensional residual matrix is learned. Plane coordinates and original residual magnitude are preserved when reconstructing embeddings. Residual direction changes. Three matrix-training conditions: difference alignment, separation, sum. Separation is NF hardest-positive hinge plus hardest-disjoint-label-positive hinge, both at cosine limit zero. Every training candidate is searched each epoch. Results use converged matrices and validation-only thresholds. This architecture differs from the earlier text-encoder training experiment.

A converged objective and universal pairwise separation are reported separately; the CSV records maximum cosine, fraction of violating pairs, and fraction of anchors satisfying every pair.
