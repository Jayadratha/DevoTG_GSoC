# DevoTG: Developmental Temporal Graph Networks

A framework for analyzing *C. elegans* neural development using Temporal Graph Neural Networks (TGNs) and discrete temporal graph analysis. DevoTG integrates cell lineage data and multi-timepoint connectome reconstructions to model how a nervous system wires itself from birth to adulthood.

**Research branch** (`research`): contains all paper figures, baseline comparison scripts, and reproducibility tools for the accompanying manuscript.

---

## Overview

DevoTG consists of two complementary components:

| Component | Data | Task | Result |
|-----------|------|------|--------|
| **TGN (CTDG)** | Cell lineage CSV (642 divisions, 1,203 cells) | Cell division event prediction | Test AUC **0.839 ± 0.007** (5 seeds) |
| **DTDG analysis** | Witvliet et al. 2021 (8 timepoints, 225 neurons) | Connectome topology, connection stability | 3-class stability taxonomy |

The TGN outperforms a static GNN baseline with identical architecture by **26 AUC points** (0.839 vs 0.577), demonstrating that temporal memory is the critical inductive bias for developmental prediction.

---

## Repository Structure

```
DevoTG_GSoC/
├── devotg/                        # Core package
│   ├── data/                      # Loaders: lineage CSV, Witvliet Excel, CTDG/DTDG builders
│   ├── models/                    # TGN: TGNMemory + TransformerConv + LinkPredictor
│   ├── visualization/             # Plotly animations, static plots, heatmaps
│   ├── analysis/                  # Network topology, centrality, stability classification
│   └── utils/                     # Threshold calculators
│
├── scripts/
│   ├── train_tgn.py               # Train TGN on cell lineage data
│   ├── run_connectome_analysis.py # DTDG analysis of Witvliet connectome
│   ├── run_baselines.py           # Random / Degree / Static GNN baselines (single run)
│   ├── run_multi_seed_eval.py     # Multi-seed evaluation (mean ± std, fully reproducible)
│   └── generate_paper_figures.py  # All 5 publication figures (300 DPI)
│
├── notebooks/                     # Numbered Jupyter notebooks (01–04)
├── data/
│   ├── cell_lineage_datasets/     # cells_birth_and_pos.csv
│   ├── connectome_datasets/       # Witvliet et al. Excel files (downloaded)
│   └── processed_datasets/        # DTDG CSVs and PKL files
├── outputs/
│   ├── models/                    # Training history, performance JSON, baseline results
│   ├── connectome_analysis/       # Network stats, interactive HTML plots, animations
│   ├── lineage_analysis/          # Cell division plots, lineage animations
│   └── paper_figures/             # Publication figures (research branch)
├── config.yaml                    # Hyperparameters and paths
└── CLAUDE.md                      # Architecture and navigation guide
```

---

## Installation

```bash
git clone https://github.com/Jayadratha/DevoTG_GSoC.git
cd DevoTG_GSoC

conda env create -f environment.yml
conda activate devotg
pip install -e .
```

**Requirements:** Python 3.12, PyTorch 2.5.1, PyTorch Geometric 2.6.1, CUDA 12.1 (optional but recommended).

---

## Quick Start

### Train the TGN
```bash
python scripts/train_tgn.py --save_model --verbose
```

### Run connectome analysis
```bash
python scripts/run_connectome_analysis.py --skip-download
```

### Reproduce paper results (5-seed evaluation)
```bash
python scripts/run_multi_seed_eval.py --n_seeds 5 --epochs 20
```
Results are saved to `outputs/models/multi_seed_results.json`. All seeds are fully deterministic.

### Generate paper figures
```bash
python scripts/generate_paper_figures.py
```
Produces 5 figures in `outputs/paper_figures/` (300 DPI PNG):
- `fig1_sample_devotg.png` — temporal evolution as staggered 3D layers
- `fig2_stability_comparison.png` — connection stability vs Witvliet et al.
- `fig3_ava_avb_ave_circuit.png` — AVA/AVB/AVE command interneurons
- `fig4_spatiotemporal.png` — lineage tree + spatial birth positions
- `fig5_tgn_performance.png` — training curves + baseline comparison

---

## Results

### Cell Division Prediction (TGN vs Baselines)

| Model | Test AUC | Test AP |
|-------|----------|---------|
| Random | 0.539 ± 0.000 | 0.553 ± 0.000 |
| Degree heuristic | 0.463 ± 0.012 | 0.488 ± 0.005 |
| Static GNN (no memory) | 0.577 ± 0.080 | 0.563 ± 0.061 |
| **DevoTG TGN** | **0.839 ± 0.007** | **0.763 ± 0.006** |

Mean ± std over 5 seeds. Test metrics at best-validation-AUC checkpoint.

### Connectome DTDG Analysis (Witvliet et al. 2021, 8 timepoints)

| Metric | Value |
|--------|-------|
| Neurons | 225 (sensory 64, inter 43, motor 42, muscle 32, modulatory 31, glia 10) |
| Edge growth | 858 (birth) → 2,496 (adult) |
| Stable connections (≥6/8 timepoints) | 650 (15.1%) |
| Developmental connections (2–5 timepoints) | 1,207 (28.1%) |
| Variable connections (1 timepoint) | 2,440 (56.8%) |

---

## Model Architecture

```
TGNMemory (GRU, 100-D)
    ↓  IdentityMessage + LastAggregator
TransformerConv (2 heads, 100-D output, dropout 0.1)
    ↓
LinkPredictor (100 → 100 → 1)
```

Total: **132,501** trainable parameters. Trained with Adam (lr=0.001), BCEWithLogitsLoss, batch size 200, 20 epochs. Temporal split: 70/15/15 train/val/test.

---

## Data Sources

- **Connectome**: Witvliet et al. (2021). "Connectomes across development reveal principles of brain maturation." *Nature* 596, 257–261. Data via [ConnectomeToolbox](https://github.com/openworm/ConnectomeToolbox).
- **Cell lineage**: WormAtlas canonical *C. elegans* lineage, `data/cell_lineage_datasets/cells_birth_and_pos.csv`.

---

## Citation

```bibtex
@software{devotg2025,
  title   = {DevoTG: Developmental Temporal Graph Networks},
  author  = {Gayen, Jayadratha and Alicea, Bradly},
  year    = {2025},
  url     = {https://github.com/Jayadratha/DevoTG_GSoC}
}
```

---

## Acknowledgments

GSoC 2025 project under the DevoLearn / OpenWorm organization. Mentors: Bradly Alicea, Mehul Arora, Sarrah Bastawala, Jesse Parent. Data provided by Witvliet et al. (2021) and the WormAtlas community.
