```
├── .git
├── .gitignore
├── LICENCE.md
├── README.md
├── config.yaml
├── data
│   ├── cell_lineage_datasets
│   │   └── cells_birth_and_pos.csv
│   ├── connectome_datasets
│   │   ├── witvliet_2020_1.xlsx
│   │   ├── witvliet_2020_2.xlsx
│   │   ├── witvliet_2020_3.xlsx
│   │   ├── witvliet_2020_4.xlsx
│   │   ├── witvliet_2020_5.xlsx
│   │   ├── witvliet_2020_6.xlsx
│   │   ├── witvliet_2020_7.xlsx
│   │   └── witvliet_2020_8.xlsx
│   ├── processed_datasets
│   │   ├── calculated_thresholds.json
│   │   ├── dtdg_celegans_development.json
│   │   ├── dtdg_celegans_development.pkl
│   │   ├── dtdg_edges_temporal.csv
│   │   ├── dtdg_nodes.csv
│   │   ├── dtdg_summary_statistics.csv
│   │   ├── processed_cell_data.csv
│   │   └── visualization_ready_data.csv
│   └── sample_data.csv
├── devotg
│   ├── __init__.py
│   ├── __pycache__
│   ├── analysis
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── network_analysis.py
│   │   └── statistics.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── connectome_loader.py
│   │   ├── dataset_loader.py
│   │   └── temporal_graph_builder.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   └── tgn_model.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── make_json_safe.py
│   │   └── thresholds.py
│   └── visualization
│       ├── __init__.py
│       ├── __pycache__
│       ├── cell_visualizer.py
│       ├── connectome_visualizer.py
│       ├── lineage_animator.py
│       └── neural_animator.py
├── environment.yml
├── logs
│   ├── analysis_pipeline.log
│   ├── connectome_analysis.log
│   └── tgn_training.log
├── notebooks
│   ├── 01_cell_lineage_data_analysis.ipynb
│   ├── 02_cell_lineage_visualization.ipynb
│   ├── 03_tgn_training.ipynb
│   └── 04_connectome_development_analysis.ipynb
├── outputs
│   ├── connectome_analysis
│   │   ├── analysis_summary.json
│   │   ├── statistics
│   │   │   ├── comprehensive_network_report.json
│   │   │   ├── developmental_connections.csv
│   │   │   ├── network_growth_analysis.csv
│   │   │   ├── network_summary.json
│   │   │   ├── node_importance_over_time.csv
│   │   │   ├── stable_connections.csv
│   │   │   └── variable_connections.csv
│   │   └── visualizations
│   │       ├── centrality_analysis.png
│   │       ├── connection_types_evolution.png
│   │       ├── interactive_network_explorer.html
│   │       ├── network_analysis_dashboard.html
│   │       ├── network_development_animation.mp4
│   │       ├── network_graph_directed_weighted.png
│   │       ├── network_graph_hubs.png
│   │       ├── network_growth_metrics.png
│   │       ├── neural_development_animation.html
│   │       ├── neural_network_development_animation.html
│   │       └── node_importance_heatmap.html
│   ├── lineage_analysis
│   │   ├── analysis_summary.json
│   │   ├── dataset_processing
│   │   │   ├── calculated_thresholds.json
│   │   │   └── visualization_ready_data.csv
│   │   ├── statistics
│   │   │   ├── comprehensive_analysis_report.json
│   │   │   ├── comprehensive_network_report.json
│   │   │   ├── detailed_analysis_results.json
│   │   │   ├── developmental_connections.csv
│   │   │   ├── network_growth_analysis.csv
│   │   │   ├── network_summary.json
│   │   │   ├── node_importance_over_time.csv
│   │   │   ├── stable_connections.csv
│   │   │   └── variable_connections.csv
│   │   └── visualizations
│   │       ├── cell_division_animation.mp4
│   │       ├── division_patterns_analysis.png
│   │       ├── generation_analysis.png
│   │       ├── interactive_cell_division_plot.html
│   │       ├── interactive_plot_generation.html
│   │       ├── interactive_plot_size.html
│   │       ├── interactive_plot_time.html
│   │       ├── lineage_animation_ABpl.html
│   │       ├── static_plot_generation.png
│   │       ├── static_plot_size.png
│   │       ├── static_plot_time.png
│   │       ├── time_progression_analysis.png
│   │       └── visualization_summary.json
│   └── models
│       ├── data_characteristics_analysis.png
│       ├── performance_summary.json
│       ├── temporal_pattern_analysis.png
│       ├── trained_tgn_model.pth
│       ├── training_history.png
│       └── training_summary.json
├── requirements.txt
├── scripts
│   ├── run_cell_lineage_analysis.py
│   ├── run_connectome_analysis.py
│   └── train_tgn.py
└── project_structure.md
```
