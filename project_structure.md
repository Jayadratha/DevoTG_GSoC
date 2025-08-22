# DevoTG Project Structure

This document provides a complete overview of the DevoTG repository structure and organization.

## 📁 Repository Structure

```
DevoTG/
├── README.md                          # Main project documentation
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package installation script
├── .gitignore                        # Git ignore rules
├── config.yaml                       # Configuration file
├── PROJECT_STRUCTURE.md              # This file
│
├── devotg/                           # Main Python package
│   ├── __init__.py                   # Package initialization
│   │
│   ├── data/                         # Data processing modules
│   │   ├── __init__.py
│   │   ├── dataset_loader.py         # CSV data loading and validation
│   │   ├── temporal_graph_builder.py # CTDG construction for TGN
│   │   └── preprocessing.py          # Data preprocessing utilities
│   │
│   ├── models/                       # Neural network models
│   │   ├── __init__.py
│   │   ├── tgn_model.py             # Complete TGN implementation
│   │   ├── attention.py             # Attention mechanisms
│   │   └── link_predictor.py        # Link prediction components
│   │
│   ├── visualization/               # Visualization components
│   │   ├── __init__.py
│   │   ├── cell_visualizer.py       # Main visualization class
│   │   ├── lineage_animator.py      # Animated lineage trees
│   │   ├── plotly_utils.py          # Plotly visualization helpers
│   │   └── matplotlib_utils.py      # Matplotlib utilities
│   │
│   ├── analysis/                    # Statistical analysis
│   │   ├── __init__.py
│   │   ├── statistics.py            # Comprehensive statistical analysis
│   │   ├── patterns.py              # Pattern detection algorithms
│   │   └── metrics.py               # Evaluation metrics
│   │
│   └── utils/                       # Utility functions
│       ├── __init__.py
│       ├── thresholds.py            # Threshold calculations (1/2-sigma)
│       ├── io_utils.py              # Input/output utilities
│       └── constants.py             # Project constants
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_data_analysis.ipynb       # Dataset exploration and statistics
│   ├── 02_visualization.ipynb       # Interactive visualizations
│   └── 03_tgn_training.ipynb        # TGN model training and evaluation
│
├── scripts/                          # Utility scripts
│   ├── run_analysis.py              # Complete analysis pipeline
│   ├── train_tgn.py                 # TGN training script
│   └── generate_visualizations.py   # Batch visualization generation
│
├── data/                             # Data directory
│   ├── raw/                         # Raw data files
│   │   └── cells_birth_and_pos.csv  # Main dataset
│   ├── processed/                   # Processed data files
│   └── sample/                      # Sample/example data
│
├── outputs/                          # Generated outputs
│   ├── visualizations/              # Generated plots and animations
│   │   ├── static_plot_generation.png
│   │   ├── interactive_cell_division_plot.html
│   │   ├── lineage_animation_ABpl.html
│   │   └── division_patterns_analysis.png
│   ├── statistics/                  # Analysis results
│   │   ├── comprehensive_analysis_report.json
│   │   └── detailed_analysis_results.json
│   ├── dataset_processing/          # Processed datasets
│   │   ├── processed_cell_data.csv
│   │   ├── calculated_thresholds.json
│   │   └── visualization_ready_data.csv
│   ├── models/                      # Saved model weights
│   │   ├── trained_tgn_model.pth
│   │   ├── training_history.png
│   │   └── performance_summary.json
│   └── reports/                     # Generated reports
│
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── test_data/                   # Data processing tests
│   │   ├── test_dataset_loader.py
│   │   └── test_temporal_graph_builder.py
│   ├── test_models/                 # Model tests
│   │   └── test_tgn_model.py
│   ├── test_visualization/          # Visualization tests
│   │   ├── test_cell_visualizer.py
│   │   └── test_lineage_animator.py
│   └── test_utils/                  # Utility tests
│       └── test_thresholds.py
│
├── docs/                            # Documentation
│   ├── api/                         # API documentation
│   ├── examples/                    # Usage examples
│   ├── tutorials/                   # Step-by-step tutorials
│   └── bibliography.md             # Research references
│
└── config/                          # Configuration files
    ├── default_config.yaml          # Default configuration
    ├── development.yaml             # Development settings
    └── production.yaml              # Production settings
```

## 📚 Module Overview

### 🔬 Data Processing (`devotg.data`)

**Purpose**: Handle loading, validation, and conversion of cell division datasets.

- **`DatasetLoader`**: Robust CSV loading with validation and cleaning
- **`TemporalGraphBuilder`**: Convert cell lineage data to temporal graphs
- **`build_cell_ctdg()`**: Main function for creating CTDG from CSV data

**Key Features**:
- Automatic data validation and cleaning
- Support for various CSV formats and encodings
- Generation mapping and lineage tree construction
- Feature padding and normalization for neural networks

### 🧠 Models (`devotg.models`)

**Purpose**: Temporal graph neural network implementations for cell division analysis.

- **`TGNModel`**: Complete TGN wrapper with training and evaluation
- **`GraphAttentionEmbedding`**: Attention-based graph embedding layer
- **`LinkPredictor`**: Neural network for predicting cell division events

**Key Features**:
- Memory-augmented temporal graph networks
- Multi-head attention mechanisms
- Configurable architectures and hyperparameters
- Built-in training loops and evaluation metrics

### 🎨 Visualization (`devotg.visualization`)

**Purpose**: Create interactive and static visualizations of cell division patterns.

- **`CellDivisionVisualizer`**: Main visualization class with multiple rendering options
- **`LineageAnimator`**: Animated lineage tree exploration and path tracing

**Key Features**:
- Static 3D matplotlib plots with customizable coloring
- Interactive Plotly visualizations with time sliders
- Animated lineage trees showing developmental paths
- Export capabilities (PNG, HTML, MP4, GIF)

### 📊 Analysis (`devotg.analysis`)

**Purpose**: Comprehensive statistical analysis and pattern detection.

- **`StatisticalAnalyzer`**: Complete analysis pipeline with multiple metrics
- **`generate_comprehensive_report()`**: Automated report generation

**Key Features**:
- Temporal, spatial, and lineage pattern analysis
- Correlation analysis and feature importance
- Automated threshold calculation (1σ, 2σ, percentile methods)
- JSON and CSV export of results

### 🔧 Utils (`devotg.utils`)

**Purpose**: Helper functions and utilities for data processing and analysis.

- **`ThresholdCalculator`**: Statistical threshold computation with multiple methods
- **IO utilities**: File handling and data export functions
- **Constants**: Project-wide configuration and parameters

## 📓 Notebooks

### `01_data_analysis.ipynb`
- Dataset loading and validation
- Exploratory data analysis with statistics and plots
- Threshold calculation for visualization categorization
- Data export and preprocessing

### `02_visualization.ipynb`
- Interactive 3D visualizations with Plotly
- Static publication-quality plots with Matplotlib
- Animated lineage tree exploration
- Batch generation and export of all visualizations

### `03_tgn_training.ipynb`
- Temporal graph construction from cell division data
- TGN model configuration and training
- Performance evaluation and analysis
- Model saving and export

## 🖥️ Scripts

### `run_analysis.py`
Complete analysis pipeline that can be run from command line:
```bash
python scripts/run_analysis.py --csv_path data/raw/cells.csv --output_dir results/
```

### `train_tgn.py`
TGN training script with configurable parameters:
```bash
python scripts/train_tgn.py --epochs 50 --batch_size 128 --save_model
```

### `generate_visualizations.py`
Batch visualization generation for multiple datasets:
```bash
python scripts/generate_visualizations.py --input_dir data/ --output_dir viz/
```

## 🧪 Testing

### Test Structure
- **Unit tests**: Individual function and class testing
- **Integration tests**: Module interaction testing  
- **End-to-end tests**: Complete pipeline testing
- **Performance tests**: Speed and memory benchmarking

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test module
python -m pytest tests/test_visualization/

# Run with coverage
python -m pytest tests/ --cov=devotg --cov-report=html
```

## 📁 Output Structure

### Generated Files
- **Visualizations**: PNG, HTML, MP4/GIF animations
- **Statistics**: JSON reports with comprehensive analysis
- **Processed Data**: CSV files with calculated features
- **Models**: PyTorch model weights and training history
- **Reports**: Automated analysis summaries

### File Naming Convention
- Static plots: `static_plot_{color_scheme}.png`
- Interactive plots: `interactive_plot_{color_scheme}.html`
- Animations: `{animation_type}_{target_cell}.html`
- Data exports: `{processing_type}_data_{timestamp}.csv`

## ⚙️ Configuration

### `config.yaml`
Central configuration file with:
- Model hyperparameters
- Data processing settings
- Visualization parameters
- Output preferences
- Logging configuration

### Environment Variables
- `DEVOTG_DATA_PATH`: Default data directory
- `DEVOTG_OUTPUT_PATH`: Default output directory
- `DEVOTG_CONFIG_PATH`: Configuration file path

## 🚀 Quick Start

1. **Installation**:
   ```bash
   git clone https://github.com/yourusername/DevoTG.git
   cd DevoTG
   pip install -e .
   ```

2. **Basic Usage**:
   ```python
   from devotg import CellDivisionVisualizer, TGNModel
   
   # Load and visualize data
   visualizer = CellDivisionVisualizer("data/cells.csv")
   fig = visualizer.create_interactive_plot()
   fig.show()
   
   # Train TGN model
   model = TGNModel.from_csv("data/cells.csv")
   history = model.train(epochs=20)
   ```

3. **Command Line**:
   ```bash
   # Run complete analysis
   devotg-analyze --csv data/cells.csv --output results/
   
   # Train model
   devotg-train --epochs 50 --save-model
   
   # Generate visualizations
   devotg-visualize --input data/ --interactive --animated
   ```

## 📖 Documentation

- **API Reference**: Detailed module and function documentation
- **Tutorials**: Step-by-step guides for common tasks
- **Examples**: Code snippets and use cases
- **Research Background**: Scientific context and methodology

## 🤝 Contributing

- **Code Style**: PEP 8 compliance with Black formatting
- **Testing**: All new features must include tests
- **Documentation**: Docstrings and README updates required
- **Pull Requests**: Feature branches with descriptive commits

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*This structure preserves all original functionality while providing a professional, maintainable, and extensible framework for C. elegans developmental analysis.*