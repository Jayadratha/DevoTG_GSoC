# DevoTG - Developmental Temporal Graph Networks

GSoC 2025 project under DevoLearn. Analyzes C. elegans cell division and connectome development using temporal graph neural networks.

## Quick Reference

```bash
# Environment
conda activate devotg

# Run all pipelines (in order)
python scripts/run_cell_lineage_analysis.py --verbose
python scripts/train_tgn.py --save_model --verbose
python scripts/run_connectome_analysis.py --skip-download
```

## Architecture

```
devotg/
├── data/                  # Data loading & graph construction
│   ├── dataset_loader.py          # Cell division CSV loading (DatasetLoader)
│   ├── connectome_loader.py       # Witvliet et al. Excel download & processing (ConnectomeDatasetLoader)
│   └── temporal_graph_builder.py  # CTDG construction (build_cell_ctdg, pad_feature to 172-D)
├── models/
│   └── tgn_model.py               # TGN with TGNMemory + TransformerConv + LinkPredictor
├── visualization/
│   ├── cell_visualizer.py         # Cell division plots (CellDivisionVisualizer)
│   ├── connectome_visualizer.py   # Network metric plots (ConnectomeVisualizer)
│   ├── neural_animator.py         # Plotly connectome animations (NeuralNetworkAnimator)
│   └── lineage_animator.py        # Lineage tree animations (LineageAnimator)
├── analysis/
│   ├── statistics.py              # Temporal, spatial, lineage, correlation analysis
│   └── network_analysis.py        # Topology metrics, centrality, stability (ConnectomeNetworkAnalyzer)
└── utils/
    └── thresholds.py              # Automatic threshold calculation (1sigma/2sigma/percentile)
```

## Data Flow

**Cell lineage:** CSV -> DatasetLoader -> TemporalGraphBuilder -> CTDG (TemporalData) -> TGNModel
**Connectome:** Excel (8 timepoints) -> ConnectomeDatasetLoader -> DTDG CSVs -> NetworkAnalyzer -> Visualizations

Key concepts:
- **CTDG**: Continuous-Time Dynamic Graph — temporal events with exact timestamps
- **DTDG**: Discrete-Time Dynamic Graph — 8 fixed developmental snapshots
- **Feature dim**: 172-D vectors for nodes and edges (padded via `pad_feature()`)

## Scripts

| Script | What it does | Key flags |
|--------|-------------|-----------|
| `run_cell_lineage_analysis.py` | Stats, thresholds, plots, reports | `--threshold_method`, `--verbose` |
| `train_tgn.py` | Train TGN, evaluate AUC/AP, save weights | `--epochs`, `--save_model`, `--device` |
| `run_connectome_analysis.py` | Download data, network analysis, animations | `--skip-download`, `--skip-analysis`, `--skip-visualization` |

## Outputs

```
outputs/
├── lineage_analysis/        # Cell division plots (PNG), interactive HTML, animations (MP4)
├── models/                  # training_history.png, trained_tgn_model.pth, training_summary.json
└── connectome_analysis/     # Network graphs, heatmaps, dashboards, neural animations (HTML/MP4)
```

## Configuration

All parameters in `config.yaml`:
- **model**: memory_dim=100, embedding_dim=100, learning_rate=0.001, dropout=0.1
- **training**: epochs=20, batch_size=200, val_ratio=0.15, test_ratio=0.15
- **connectome**: stability_threshold=7, top_k_nodes=10

## Data Sources

- **Cell lineage**: `data/cell_lineage_datasets/cells_birth_and_pos.csv`
- **Connectome**: 8 Excel files in `data/connectome_datasets/` from Witvliet et al. 2021
- **Processed**: DTDG CSVs and PKL files in `data/processed_datasets/`

## Environment

- Python 3.12, PyTorch 2.5.1+cu121, PyG 2.6.1
- Key deps: networkx, plotly, matplotlib, scipy, scikit-learn, pandas
- Conda env name: `devotg`
- CUDA 12.1 (compatible with driver 12.4+)

## Where to Look

### "I want to..."

| Goal | Read | Write/Edit |
|------|------|------------|
| **Load or change input data format** | `devotg/data/dataset_loader.py` | Same file — modify `load_csv()` or `validate_dataset()` |
| **Add a new data source** | `devotg/data/connectome_loader.py` for reference | New loader in `devotg/data/`, register in `devotg/data/__init__.py` |
| **Change how graphs are built** | `devotg/data/temporal_graph_builder.py` | Same file — `build_cell_ctdg()`, `pad_feature()` |
| **Modify the TGN model** | `devotg/models/tgn_model.py` | Same file — `TGNModel`, `GraphAttentionEmbedding`, `LinkPredictor` |
| **Tune hyperparameters** | `config.yaml` | Same file — model/training/visualization sections |
| **Change training loop** | `scripts/train_tgn.py` | Same file — epochs, loss, evaluation logic |
| **Fix a cell division plot** | `devotg/visualization/cell_visualizer.py` | Same file — `CellDivisionVisualizer` methods |
| **Fix a connectome plot** | `devotg/visualization/connectome_visualizer.py` | Same file — `ConnectomeVisualizer` methods |
| **Edit neural animations** | `devotg/visualization/neural_animator.py` | Same file — `NeuralNetworkAnimator`, frame logic |
| **Edit lineage animations** | `devotg/visualization/lineage_animator.py` | Same file — `LineageAnimator` |
| **Change network metrics** | `devotg/analysis/network_analysis.py` | Same file — `ConnectomeNetworkAnalyzer` |
| **Change statistical analysis** | `devotg/analysis/statistics.py` | Same file — `temporal_analysis()`, `spatial_analysis()` |
| **Adjust size/time thresholds** | `devotg/utils/thresholds.py` | Same file — `ThresholdCalculator`, sigma methods |
| **See generated figures** | `outputs/lineage_analysis/visualizations/` | Re-run `scripts/run_cell_lineage_analysis.py` |
| **See training results** | `outputs/models/training_summary.json`, `training_history.png` | Re-run `scripts/train_tgn.py` |
| **See connectome outputs** | `outputs/connectome_analysis/visualizations/` | Re-run `scripts/run_connectome_analysis.py` |
| **Check raw cell data** | `data/cell_lineage_datasets/cells_birth_and_pos.csv` | Don't — source data |
| **Check raw connectome data** | `data/connectome_datasets/*.xlsx` | Don't — downloaded from Witvliet et al. |
| **Check processed graphs** | `data/processed_datasets/dtdg_*.csv`, `*.pkl` | Regenerated by `run_connectome_analysis.py` |
| **Debug a pipeline run** | `logs/` — `analysis_pipeline.log`, `connectome_analysis.log`, `tgn_training.log` | Logs are auto-generated |
| **Explore interactively** | `notebooks/` — numbered 01-04 in order | Edit notebooks directly |
| **Add a new analysis pipeline** | `scripts/` for existing patterns | New script in `scripts/`, new module in `devotg/analysis/` |
| **Change dependencies** | `environment.yml` (conda), `requirements.txt` (pip) | Same files |

### Key entry points by pipeline

- **Cell lineage end-to-end**: Start at `scripts/run_cell_lineage_analysis.py` — it calls `devotg/data/dataset_loader.py` -> `devotg/utils/thresholds.py` -> `devotg/analysis/statistics.py` -> `devotg/visualization/cell_visualizer.py`
- **TGN training end-to-end**: Start at `scripts/train_tgn.py` — it calls `devotg/data/temporal_graph_builder.py` -> `devotg/models/tgn_model.py`
- **Connectome end-to-end**: Start at `scripts/run_connectome_analysis.py` — it calls `devotg/data/connectome_loader.py` -> `devotg/analysis/network_analysis.py` -> `devotg/visualization/connectome_visualizer.py` + `neural_animator.py`

## Notes

- No formal test suite — validation through dataset checks and notebook runs
- Logs written to `logs/` (analysis_pipeline.log, connectome_analysis.log, tgn_training.log)
- `run_connectome_analysis.py` does NOT accept `--verbose`
- PyG packages (pyg-lib, torch-scatter, torch-sparse, etc.) require special index: `https://data.pyg.org/whl/torch-2.5.1+cu121.html`
