"""
DevoTG: Developmental Temporal Graph Networks

A comprehensive framework for analyzing C. elegans cell division patterns using 
temporal graph neural networks and interactive visualizations.

Author: DevoTG Team
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "DevoTG Team"
__email__ = "devotg@example.com"
__description__ = "Developmental Temporal Graph Networks for C. elegans Analysis"

# Core imports for easy access
from .data import (
    DatasetLoader,
    TemporalGraphBuilder,
    build_cell_ctdg
)

from .visualization import (
    CellDivisionVisualizer,
    LineageAnimator
)

# from .models import (
#     TGNModel,
#     GraphAttentionEmbedding,
#     LinkPredictor
# )

from .utils import (
    ThresholdCalculator,
    calculate_automatic_thresholds
)

from .analysis import (
    StatisticalAnalyzer,
    generate_comprehensive_report
)

# Package metadata
__all__ = [
    # Data modules
    'DatasetLoader',
    'TemporalGraphBuilder', 
    'build_cell_ctdg',
    
    # Visualization modules
    'CellDivisionVisualizer',
    'LineageAnimator',
    
    # Model modules
    'TGNModel',
    'GraphAttentionEmbedding',
    'LinkPredictor',
#
    # Utility modules
    'ThresholdCalculator',
    'calculate_automatic_thresholds',
    
    # Analysis modules
    'StatisticalAnalyzer',
    'generate_comprehensive_report'
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
