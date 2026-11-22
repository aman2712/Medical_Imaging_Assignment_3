# ENG2440 Assignment 3: Molecular Property Prediction with Graph Neural Networks

Student: Md Aman Khan (25011138)

This project predicts blood-brain barrier penetration (BBBP, MoleculeNet) from molecular
structure, comparing a Morgan-fingerprint logistic-regression baseline with a bond-aware
graph neural network under random and Bemis-Murcko scaffold splits.

## Setup

1. Use Python 3.13 and install the packages: `pip install -r requirements.txt`
2. No manual download is needed. The first run downloads the BBBP dataset through the
   PyTorch Geometric `MoleculeNet` loader into `data/MoleculeNet/`, and later runs use
   that cached copy. The first run therefore needs an internet connection.
3. If you want the data somewhere else, change `DATA_ROOT` in the Configuration cell of
   the notebook. This is the only path to change.

## Run order

1. Open `25011138_ENG2440_A3.ipynb`.
2. Run all cells from top to bottom.

On the first run the notebook does these steps itself:

- It downloads and caches the BBBP dataset in `data/`.
- It cleans the dataset, builds the random and scaffold splits, and trains the models.
- It saves trained models and result tables in `checkpoints/`.
- It reports the test results for both models under both split strategies, then the
  error and similarity analysis.

If `checkpoints/` already holds the saved models, training is skipped and they are loaded instead.

## Reproducibility

- Seed: 42 for the splits and the baseline; the GNN is repeated with seeds 42, 43 and 44.
- Hardware: Apple M4 Max, 36 GB memory. The GNN is trained on the CPU, not on `mps`:
  the scatter aggregation used by message passing is non-deterministic on the Apple GPU,
  so the same seed gives different results between runs. On the CPU the runs reproduce
  exactly, and at this model size the CPU is also faster.
- Time: about 4 minutes for a full run from scratch, almost all of it the six GNN
  trainings (two split strategies by three seeds). With `checkpoints/` already
  populated the whole notebook runs in about 35 seconds.
- Results may differ slightly on other hardware.

## Files

- `25011138_ENG2440_A3.ipynb`: the executed notebook
- `scripts/`: helper code used by the notebook
- `requirements.txt`: package versions
- `ai_use_declaration.txt`: AI use declaration
- `Md_Aman_Khan_25011138_ENG2440_A3.pdf`: the submitted report
- `report_source.tex`: the LaTeX source of the report, kept for reference
  (its figures live in `report_assets/`, which is not uploaded)

The downloaded dataset (`data/`) and `checkpoints/` are not uploaded.
