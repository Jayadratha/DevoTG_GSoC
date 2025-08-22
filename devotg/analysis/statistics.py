"""
Statistical Analysis Utilities

This module contains statistical analysis functions for cell division data.
Preserves all original functionality from the notebooks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')


def basic_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate basic statistics for the cell division dataset.
    
    Args:
        df: DataFrame with cell division data
        
    Returns:
        Dictionary containing basic statistics
    """
    stats_dict = {}
    
    # Basic info
    stats_dict['dataset_info'] = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2)
    }
    
    # Missing values
    stats_dict['missing_values'] = df.isnull().sum().to_dict()
    
    # Numeric columns statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats_dict['numeric_summary'] = df[numeric_cols].describe().to_dict()
    
    # Cell-specific statistics
    if 'Parent Cell' in df.columns:
        parent_counts = df['Parent Cell'].value_counts()
        stats_dict['parent_cell_stats'] = {
            'unique_parents': df['Parent Cell'].nunique(),
            'total_divisions': len(df),
            'avg_divisions_per_parent': len(df) / df['Parent Cell'].nunique(),
            'max_divisions_single_parent': parent_counts.iloc[0] if len(parent_counts) > 0 else 0,
            'most_active_parents': parent_counts.head(10).to_dict()
        }
    
    # Daughter cell statistics
    if 'Daughter 1' in df.columns and 'Daughter 2' in df.columns:
        all_daughters = pd.concat([df['Daughter 1'], df['Daughter 2']]).dropna()
        daughter_counts = all_daughters.value_counts()
        
        stats_dict['daughter_cell_stats'] = {
            'unique_daughters': all_daughters.nunique(),
            'total_daughter_occurrences': len(all_daughters),
            'most_frequent_daughters': daughter_counts.head(10).to_dict()
        }
        
        # Cells that are both parents and daughters (lineage connections)
        parent_set = set(df['Parent Cell'].unique())
        daughter_set = set(all_daughters.unique())
        both_roles = parent_set.intersection(daughter_set)
        
        stats_dict['lineage_stats'] = {
            'cells_both_parent_daughter': len(both_roles),
            'only_parents': len(parent_set - daughter_set),
            'only_daughters': len(daughter_set - parent_set),
            'example_both_roles': list(both_roles)[:10]
        }
    
    return stats_dict


def temporal_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze temporal patterns in cell divisions.
    
    Args:
        df: DataFrame with birth time data
        
    Returns:
        Dictionary with temporal analysis results
    """
    if 'Birth Time' not in df.columns:
        raise ValueError("DataFrame must contain 'Birth Time' column")
    
    birth_times = df['Birth Time'].dropna()
    
    temporal_stats = {
        'basic_stats': {
            'min_time': birth_times.min(),
            'max_time': birth_times.max(),
            'time_span': birth_times.max() - birth_times.min(),
            'mean_time': birth_times.mean(),
            'median_time': birth_times.median(),
            'std_time': birth_times.std(),
            'unique_time_points': birth_times.nunique()
        }
    }
    
    # Time distribution analysis
    # Binning analysis
    n_bins = min(50, birth_times.nunique())
    counts, bin_edges = np.histogram(birth_times, bins=n_bins)
    temporal_stats['distribution'] = {
        'histogram_counts': counts.tolist(),
        'bin_edges': bin_edges.tolist(),
        'peak_time_bin': bin_edges[np.argmax(counts)],
        'peak_divisions_count': int(np.max(counts))
    }
    
    # Time intervals between divisions
    sorted_times = birth_times.sort_values()
    time_intervals = sorted_times.diff().dropna()
    temporal_stats['intervals'] = {
        'mean_interval': time_intervals.mean(),
        'median_interval': time_intervals.median(),
        'std_interval': time_intervals.std(),
        'min_interval': time_intervals.min(),
        'max_interval': time_intervals.max()
    }
    
    # Divisiveness over time (divisions per unit time)
    time_bins = pd.cut(birth_times, bins=20)
    divisions_per_bin = time_bins.value_counts().sort_index()
    temporal_stats['activity_over_time'] = {
        'peak_activity_period': str(divisions_per_bin.idxmax()),
        'peak_activity_count': int(divisions_per_bin.max()),
        'lowest_activity_period': str(divisions_per_bin.idxmin()),
        'lowest_activity_count': int(divisions_per_bin.min()),
        'activity_distribution': divisions_per_bin.tolist()
    }
    
    return temporal_stats


def spatial_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze spatial distribution of cell divisions.
    
    Args:
        df: DataFrame with spatial coordinate data
        
    Returns:
        Dictionary with spatial analysis results
    """
    coord_cols = ['parent_x', 'parent_y', 'parent_z']
    missing_cols = [col for col in coord_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame must contain coordinate columns: {missing_cols}")
    
    spatial_stats = {}
    
    # Basic coordinate statistics
    for coord in coord_cols:
        data = df[coord].dropna()
        spatial_stats[coord] = {
            'min': data.min(),
            'max': data.max(),
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'range': data.max() - data.min(),
            'q25': data.quantile(0.25),
            'q75': data.quantile(0.75)
        }
    
    # Calculate cell sizes (distance from origin)
    df_temp = df.copy()
    df_temp['cell_size'] = np.sqrt(
        df_temp['parent_x']**2 + 
        df_temp['parent_y']**2 + 
        df_temp['parent_z']**2
    )
    
    cell_sizes = df_temp['cell_size'].dropna()
    spatial_stats['cell_size'] = {
        'min': cell_sizes.min(),
        'max': cell_sizes.max(),
        'mean': cell_sizes.mean(),
        'median': cell_sizes.median(),
        'std': cell_sizes.std(),
        'q25': cell_sizes.quantile(0.25),
        'q75': cell_sizes.quantile(0.75)
    }
    
    # Spatial clustering analysis
    # Calculate pairwise distances (sample if too large)
    if len(df) > 1000:
        sample_df = df.sample(n=1000, random_state=42)
    else:
        sample_df = df
    
    coords = sample_df[coord_cols].values
    from scipy.spatial.distance import pdist
    distances = pdist(coords)
    
    spatial_stats['clustering'] = {
        'mean_pairwise_distance': distances.mean(),
        'median_pairwise_distance': np.median(distances),
        'std_pairwise_distance': distances.std(),
        'min_pairwise_distance': distances.min(),
        'max_pairwise_distance': distances.max()
    }
    
    # Center of mass
    spatial_stats['center_of_mass'] = {
        'x': df['parent_x'].mean(),
        'y': df['parent_y'].mean(),
        'z': df['parent_z'].mean()
    }
    
    return spatial_stats


def lineage_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze cell lineage patterns.
    
    Args:
        df: DataFrame with parent-daughter relationships
        
    Returns:
        Dictionary with lineage analysis results
    """
    required_cols = ['Parent Cell', 'Daughter 1', 'Daughter 2']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame must contain columns: {missing_cols}")
    
    lineage_stats = {}
    
    # Basic lineage statistics
    parent_counts = df['Parent Cell'].value_counts()
    all_daughters = pd.concat([df['Daughter 1'], df['Daughter 2']]).dropna()
    daughter_counts = all_daughters.value_counts()
    
    # Find cells that appear as both parents and daughters
    parents_set = set(df['Parent Cell'].unique())
    daughters_set = set(all_daughters.unique())
    both_roles = parents_set.intersection(daughters_set)
    root_cells = parents_set - daughters_set  # Only parents, never daughters
    leaf_cells = daughters_set - parents_set  # Only daughters, never parents
    
    lineage_stats['basic_lineage'] = {
        'total_unique_parents': len(parents_set),
        'total_unique_daughters': len(daughters_set),
        'cells_both_roles': len(both_roles),
        'root_cells': len(root_cells),
        'leaf_cells': len(leaf_cells),
        'total_unique_cells': len(parents_set.union(daughters_set))
    }
    
    # Generation analysis (simplified)
    # Build a simple lineage tree to estimate generations
    lineage_tree = {}
    for _, row in df.iterrows():
        parent = row['Parent Cell']
        daughters = [row['Daughter 1'], row['Daughter 2']]
        lineage_tree[parent] = daughters
    
    # Estimate generations by finding paths from roots
    def estimate_generation(cell, tree, visited=None, depth=0):
        if visited is None:
            visited = set()
        if cell in visited or depth > 20:  # Prevent infinite loops
            return depth
        visited.add(cell)
        
        if cell not in tree:
            return depth
        
        max_child_depth = depth
        for child in tree[cell]:
            if child:  # Skip empty strings
                child_depth = estimate_generation(child, tree, visited.copy(), depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    generation_depths = {}
    for root in root_cells:
        generation_depths[root] = estimate_generation(root, lineage_tree)
    
    lineage_stats['generation_analysis'] = {
        'max_generation_depth': max(generation_depths.values()) if generation_depths else 0,
        'avg_generation_depth': np.mean(list(generation_depths.values())) if generation_depths else 0,
        'root_generation_depths': generation_depths
    }
    
    # Division frequency analysis
    lineage_stats['division_patterns'] = {
        'most_prolific_parents': parent_counts.head(10).to_dict(),
        'single_division_parents': (parent_counts == 1).sum(),
        'multiple_division_parents': (parent_counts > 1).sum(),
        'max_divisions_single_parent': parent_counts.max()
    }
    
    return lineage_stats


def correlation_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze correlations between different variables.
    
    Args:
        df: DataFrame with cell division data
        
    Returns:
        Dictionary with correlation analysis results
    """
    # Calculate cell size if not present
    df_temp = df.copy()
    if 'cell_size' not in df_temp.columns:
        df_temp['cell_size'] = np.sqrt(
            df_temp['parent_x']**2 + 
            df_temp['parent_y']**2 + 
            df_temp['parent_z']**2
        )
    
    # Select numeric columns for correlation
    numeric_cols = ['Birth Time', 'parent_x', 'parent_y', 'parent_z', 'cell_size']
    available_cols = [col for col in numeric_cols if col in df_temp.columns]
    
    if len(available_cols) < 2:
        return {'error': 'Insufficient numeric columns for correlation analysis'}
    
    corr_matrix = df_temp[available_cols].corr()
    
    correlation_stats = {
        'correlation_matrix': corr_matrix.to_dict(),
        'strong_correlations': [],
        'weak_correlations': [],
        'significant_pairs': []
    }
    
    # Find strong and weak correlations
    for i, col1 in enumerate(available_cols):
        for j, col2 in enumerate(available_cols):
            if i < j:  # Avoid duplicates and self-correlation
                corr_val = corr_matrix.loc[col1, col2]
                pair = (col1, col2, corr_val)
                
                if abs(corr_val) > 0.7:
                    correlation_stats['strong_correlations'].append(pair)
                elif abs(corr_val) < 0.3:
                    correlation_stats['weak_correlations'].append(pair)
                
                if abs(corr_val) > 0.5:
                    correlation_stats['significant_pairs'].append(pair)
    
    return correlation_stats


def generate_comprehensive_report(df: pd.DataFrame, save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive statistical analysis report.
    
    Args:
        df: DataFrame with cell division data
        save_path: Optional path to save the report as JSON
        
    Returns:
        Complete analysis report dictionary
    """
    print("Generating comprehensive statistical analysis report...")
    
    report = {
        'metadata': {
            'analysis_timestamp': str(pd.Timestamp.now()),
            'dataset_shape': df.shape,
            'dataset_columns': list(df.columns)
        }
    }
    
    try:
        report['basic_statistics'] = basic_dataset_statistics(df)
        print("✓ Basic statistics completed")
    except Exception as e:
        report['basic_statistics'] = {'error': str(e)}
        print(f"✗ Basic statistics failed: {e}")
    
    try:
        report['temporal_analysis'] = temporal_analysis(df)
        print("✓ Temporal analysis completed")
    except Exception as e:
        report['temporal_analysis'] = {'error': str(e)}
        print(f"✗ Temporal analysis failed: {e}")
    
    try:
        report['spatial_analysis'] = spatial_analysis(df)
        print("✓ Spatial analysis completed")
    except Exception as e:
        report['spatial_analysis'] = {'error': str(e)}
        print(f"✗ Spatial analysis failed: {e}")
    
    try:
        report['lineage_analysis'] = lineage_analysis(df)
        print("✓ Lineage analysis completed")
    except Exception as e:
        report['lineage_analysis'] = {'error': str(e)}
        print(f"✗ Lineage analysis failed: {e}")
    
    try:
        report['correlation_analysis'] = correlation_analysis(df)
        print("✓ Correlation analysis completed")
    except Exception as e:
        report['correlation_analysis'] = {'error': str(e)}
        print(f"✗ Correlation analysis failed: {e}")
    
    # Save report if path provided
    if save_path:
        import json
        from pathlib import Path
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved to: {save_path}")
    
    return report


def create_summary_plots(df: pd.DataFrame, save_dir: Optional[str] = None) -> List[plt.Figure]:
    """
    Create summary plots for the dataset.
    
    Args:
        df: DataFrame with cell division data
        save_dir: Optional directory to save plots
        
    Returns:
        List of matplotlib figures
    """
    figures = []
    
    # Calculate cell size if not present
    df_temp = df.copy()
    if 'cell_size' not in df_temp.columns:
        df_temp['cell_size'] = np.sqrt(
            df_temp['parent_x']**2 + 
            df_temp['parent_y']**2 + 
            df_temp['parent_z']**2
        )
    
    # Figure 1: Basic distributions
    fig1, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Birth time distribution
    if 'Birth Time' in df.columns:
        axes[0, 0].hist(df['Birth Time'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Birth Time')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Birth Times')
    
    # Cell size distribution
    axes[0, 1].hist(df_temp['cell_size'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_xlabel('Cell Size (Distance from Origin)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Cell Sizes')
    
    # Parent cell frequency
    if 'Parent Cell' in df.columns:
        parent_counts = df['Parent Cell'].value_counts().head(15)
        axes[1, 0].barh(range(len(parent_counts)), parent_counts.values)
        axes[1, 0].set_yticks(range(len(parent_counts)))
        axes[1, 0].set_yticklabels(parent_counts.index)
        axes[1, 0].set_xlabel('Number of Divisions')
        axes[1, 0].set_title('Most Active Parent Cells')
    
    # Coordinate ranges
    coord_cols = ['parent_x', 'parent_y', 'parent_z']
    if all(col in df.columns for col in coord_cols):
        coord_ranges = [df[col].max() - df[col].min() for col in coord_cols]
        axes[1, 1].bar(coord_cols, coord_ranges, alpha=0.7, color=['red', 'green', 'blue'])
        axes[1, 1].set_ylabel('Coordinate Range')
        axes[1, 1].set_title('Spatial Coordinate Ranges')
    
    plt.tight_layout()
    figures.append(fig1)
    
    # Figure 2: Correlation heatmap
    numeric_cols = ['Birth Time', 'parent_x', 'parent_y', 'parent_z', 'cell_size']
    available_cols = [col for col in numeric_cols if col in df_temp.columns]
    
    if len(available_cols) > 2:
        fig2, ax = plt.subplots(1, 1, figsize=(10, 8))
        corr_matrix = df_temp[available_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation Matrix of Numeric Variables')
        plt.tight_layout()
        figures.append(fig2)
    
    # Save figures if directory provided
    if save_dir:
        from pathlib import Path
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for i, fig in enumerate(figures, 1):
            fig.savefig(save_dir / f'summary_plot_{i}.png', dpi=300, bbox_inches='tight')
        print(f"Summary plots saved to: {save_dir}")
    
    return figures


class StatisticalAnalyzer:
    """
    Class for comprehensive statistical analysis of cell division data.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the analyzer with a dataset.
        
        Args:
            df: DataFrame with cell division data
        """
        self.df = df
        self.results = {}
        
    def run_all_analyses(self) -> Dict[str, Any]:
        """Run all available analyses."""
        print("Running comprehensive statistical analysis...")
        
        analyses = [
            ('basic_statistics', basic_dataset_statistics),
            ('temporal_analysis', temporal_analysis),
            ('spatial_analysis', spatial_analysis),
            ('lineage_analysis', lineage_analysis),
            ('correlation_analysis', correlation_analysis)
        ]
        
        for name, func in analyses:
            try:
                self.results[name] = func(self.df)
                print(f"✓ {name.replace('_', ' ').title()} completed")
            except Exception as e:
                self.results[name] = {'error': str(e)}
                print(f"✗ {name.replace('_', ' ').title()} failed: {e}")
        
        return self.results
    
    def get_summary(self) -> str:
        """Get a text summary of the analysis results."""
        if not self.results:
            return "No analysis results available. Run analyses first."
        
        summary_lines = ["STATISTICAL ANALYSIS SUMMARY", "=" * 50]
        
        # Basic info
        if 'basic_statistics' in self.results and 'dataset_info' in self.results['basic_statistics']:
            info = self.results['basic_statistics']['dataset_info']
            summary_lines.extend([
                f"Dataset: {info['total_rows']} rows × {info['total_columns']} columns",
                f"Memory usage: {info['memory_usage_mb']:.2f} MB",
                ""
            ])
        
        # Add summaries from each analysis
        for analysis_name, analysis_results in self.results.items():
            if 'error' not in analysis_results:
                summary_lines.append(f"{analysis_name.replace('_', ' ').title()}:")
                # Add key findings (simplified)
                summary_lines.append("  ✓ Analysis completed successfully")
                summary_lines.append("")
        
        return "\n".join(summary_lines)
    
    def save_results(self, file_path: str):
        """Save analysis results to JSON file."""
        import json
        from pathlib import Path
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"Analysis results saved to: {file_path}")