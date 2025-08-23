"""
DevoTG: Developmental Temporal Graph Networks

A comprehensive framework for analyzing C. elegans cell division patterns and 
connectome development using temporal graph neural networks and interactive visualizations.

Author: Jayadratha Gayen
Version: 0.2.0
"""

__version__ = "0.2.0"
__author__ = "Jayadratha Gayen"
__email__ = "jayadratha.gayen@research.iiit.ac.in"
__description__ = "Developmental Temporal Graph Networks for C. elegans Analysis"

# Core imports for easy access
from .data import (
    DatasetLoader,
    TemporalGraphBuilder,
    build_cell_ctdg,
    ConnectomeDatasetLoader,
    load_connectome_datasets
)

from .visualization import (
    CellDivisionVisualizer,
    LineageAnimator,
    ConnectomeVisualizer,
    InteractiveConnectomeVisualizer,
    NeuralNetworkAnimator,
    create_comprehensive_visualizations,
    create_neural_network_animation
)

from .models import (
    TGNModel,
    GraphAttentionEmbedding,
    LinkPredictor
)

from .utils import (
    ThresholdCalculator,
    calculate_automatic_thresholds    
)

from .analysis import (
    StatisticalAnalyzer,
    generate_comprehensive_report,
    ConnectomeNetworkAnalyzer,
    analyze_connectome_network
)

# Package metadata
__all__ = [
    # Data modules
    'DatasetLoader',
    'TemporalGraphBuilder', 
    'build_cell_ctdg',
    'ConnectomeDatasetLoader',
    'load_connectome_datasets',
    
    # Visualization modules
    'CellDivisionVisualizer',
    'LineageAnimator',
    'ConnectomeVisualizer',
    'InteractiveConnectomeVisualizer',
    'NeuralNetworkAnimator',
    'create_comprehensive_visualizations',
    'create_neural_network_animation',
    
    # Model modules
    'TGNModel',
    'GraphAttentionEmbedding',
    'LinkPredictor',
    
    # Utility modules
    'ThresholdCalculator',
    'calculate_automatic_thresholds',
    
    # Analysis modules
    'StatisticalAnalyzer',
    'generate_comprehensive_report',
    'ConnectomeNetworkAnalyzer',
    'analyze_connectome_network'
]

def get_version():
    """Get the current version of DevoTG."""
    return __version__

def get_info():
    """Get package information."""
    return {
        'name': 'devotg',
        'version': __version__,
        'description': __description__,
        'author': __author__,
        'email': __email__
    }
