"""
Threshold Calculation Utilities

This module contains utilities for calculating thresholds for data analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def calculate_thresholds_sigma(data_series: pd.Series, n_sigma: int = 1) -> Tuple[float, float, float, float]:
    """
    Calculate n-sigma thresholds for a given pandas Series.

    Args:
        data_series (pd.Series): The data series to analyze.
        n_sigma (int): Number of standard deviations to use (default: 1)

    Returns:
        tuple: (mean, std_dev, lower_threshold, upper_threshold)
    """
    mean = data_series.mean()
    std_dev = data_series.std()
    lower_threshold = mean - n_sigma * std_dev
    upper_threshold = mean + n_sigma * std_dev
    return mean, std_dev, lower_threshold, upper_threshold


def calculate_thresholds_percentile(data_series: pd.Series, 
                                  lower_percentile: float = 25, 
                                  upper_percentile: float = 75) -> Tuple[float, float]:
    """
    Calculate percentile-based thresholds for a given pandas Series.

    Args:
        data_series (pd.Series): The data series to analyze.
        lower_percentile (float): Lower percentile threshold (0-100)
        upper_percentile (float): Upper percentile threshold (0-100)

    Returns:
        tuple: (lower_threshold, upper_threshold)
    """
    lower_threshold = np.percentile(data_series, lower_percentile)
    upper_threshold = np.percentile(data_series, upper_percentile)
    return lower_threshold, upper_threshold


def calculate_automatic_thresholds(df: pd.DataFrame, 
                                 method: str = "1sigma",
                                 percentile_values: Tuple[float, float] = (25, 75)) -> dict:
    """
    Automatically calculate thresholds for cell size and birth time.
    
    Args:
        df: DataFrame with cell division data
        method: Method for threshold calculation ("1sigma", "2sigma", "percentile")
        percentile_values: Percentile values if using percentile method
        
    Returns:
        Dictionary with calculated thresholds
    """
    # Ensure required columns exist
    required_cols = ['Birth Time', 'parent_x', 'parent_y', 'parent_z']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Calculate cell size if not already done
    if 'cell_size' not in df.columns:
        df = df.copy()  # Don't modify original DataFrame
        df['cell_size'] = np.sqrt(df['parent_x']**2 + df['parent_y']**2 + df['parent_z']**2)
    
    results = {}
    
    if method in ["1sigma", "2sigma"]:
        n_sigma = 1 if method == "1sigma" else 2
        
        # Birth Time Thresholds
        time_mean, time_std, time_lower, time_upper = calculate_thresholds_sigma(
            df['Birth Time'], n_sigma)
        results.update({
            'birth_time_mean': time_mean,
            'birth_time_std': time_std,
            'birth_time_threshold_low': time_lower,
            'birth_time_threshold_high': time_upper
        })
        
        # Cell Size Thresholds
        size_mean, size_std, size_lower, size_upper = calculate_thresholds_sigma(
            df['cell_size'], n_sigma)
        results.update({
            'cell_size_mean': size_mean,
            'cell_size_std': size_std,
            'size_threshold_small': size_lower,
            'size_threshold_large': size_upper
        })
        
    elif method == "percentile":
        lower_perc, upper_perc = percentile_values
        
        # Birth Time Thresholds
        time_lower, time_upper = calculate_thresholds_percentile(
            df['Birth Time'], lower_perc, upper_perc)
        results.update({
            'birth_time_threshold_low': time_lower,
            'birth_time_threshold_high': time_upper
        })
        
        # Cell Size Thresholds
        size_lower, size_upper = calculate_thresholds_percentile(
            df['cell_size'], lower_perc, upper_perc)
        results.update({
            'size_threshold_small': size_lower,
            'size_threshold_large': size_upper
        })
        
    else:
        raise ValueError(f"Unknown method: {method}. Use '1sigma', '2sigma', or 'percentile'")
    
    return results


def print_threshold_summary(df: pd.DataFrame, thresholds: dict):
    """
    Print a summary of calculated thresholds.
    
    Args:
        df: DataFrame with the data
        thresholds: Dictionary of calculated thresholds
    """
    print("=" * 60)
    print("THRESHOLD CALCULATION SUMMARY")
    print("=" * 60)
    
    # Birth Time Summary
    if 'birth_time_mean' in thresholds:
        print("\nBirth Time Analysis:")
        print(f"  Mean: {thresholds['birth_time_mean']:.2f}")
        print(f"  Std Dev: {thresholds['birth_time_std']:.2f}")
        print(f"  Lower Threshold: {thresholds['birth_time_threshold_low']:.2f}")
        print(f"  Upper Threshold: {thresholds['birth_time_threshold_high']:.2f}")
        
        # Count samples in each category
        early_count = sum(df['Birth Time'] < thresholds['birth_time_threshold_low'])
        late_count = sum(df['Birth Time'] > thresholds['birth_time_threshold_high'])
        mid_count = len(df) - early_count - late_count
        
        print(f"  Early divisions: {early_count} ({early_count/len(df)*100:.1f}%)")
        print(f"  Mid divisions: {mid_count} ({mid_count/len(df)*100:.1f}%)")
        print(f"  Late divisions: {late_count} ({late_count/len(df)*100:.1f}%)")
    else:
        print("\nBirth Time Analysis (Percentile-based):")
        print(f"  Lower Threshold: {thresholds['birth_time_threshold_low']:.2f}")
        print(f"  Upper Threshold: {thresholds['birth_time_threshold_high']:.2f}")
    
    # Cell Size Summary
    if 'cell_size' not in df.columns:
        df_temp = df.copy()
        df_temp['cell_size'] = np.sqrt(df_temp['parent_x']**2 + df_temp['parent_y']**2 + df_temp['parent_z']**2)
    else:
        df_temp = df
        
    if 'cell_size_mean' in thresholds:
        print("\nCell Size Analysis:")
        print(f"  Mean: {thresholds['cell_size_mean']:.2f}")
        print(f"  Std Dev: {thresholds['cell_size_std']:.2f}")
        print(f"  Small Threshold: {thresholds['size_threshold_small']:.2f}")
        print(f"  Large Threshold: {thresholds['size_threshold_large']:.2f}")
        
        # Count samples in each category
        small_count = sum(df_temp['cell_size'] < thresholds['size_threshold_small'])
        large_count = sum(df_temp['cell_size'] > thresholds['size_threshold_large'])
        medium_count = len(df_temp) - small_count - large_count
        
        print(f"  Small cells: {small_count} ({small_count/len(df_temp)*100:.1f}%)")
        print(f"  Medium cells: {medium_count} ({medium_count/len(df_temp)*100:.1f}%)")
        print(f"  Large cells: {large_count} ({large_count/len(df_temp)*100:.1f}%)")
    else:
        print("\nCell Size Analysis (Percentile-based):")
        print(f"  Small Threshold: {thresholds['size_threshold_small']:.2f}")
        print(f"  Large Threshold: {thresholds['size_threshold_large']:.2f}")
    
    print("=" * 60)


class ThresholdCalculator:
    """
    Class for managing threshold calculations with different methods.
    """
    
    def __init__(self, method: str = "1sigma", percentile_values: Tuple[float, float] = (25, 75)):
        """
        Initialize the threshold calculator.
        
        Args:
            method: Method for threshold calculation
            percentile_values: Percentile values for percentile method
        """
        self.method = method
        self.percentile_values = percentile_values
        self.thresholds = {}
        
    def calculate_thresholds(self, df: pd.DataFrame) -> dict:
        """
        Calculate thresholds for the given DataFrame.
        
        Args:
            df: DataFrame with cell division data
            
        Returns:
            Dictionary with calculated thresholds
        """
        self.thresholds = calculate_automatic_thresholds(
            df, self.method, self.percentile_values)
        return self.thresholds
    
    def get_thresholds(self) -> dict:
        """Get the calculated thresholds."""
        return self.thresholds
    
    def apply_thresholds_to_visualizer(self, visualizer):
        """
        Apply calculated thresholds to a CellDivisionVisualizer instance.
        
        Args:
            visualizer: CellDivisionVisualizer instance
        """
        if not self.thresholds:
            raise ValueError("No thresholds calculated. Run calculate_thresholds first.")
            
        visualizer.size_threshold_small = self.thresholds['size_threshold_small']
        visualizer.size_threshold_large = self.thresholds['size_threshold_large']
        visualizer.birth_time_threshold_low = self.thresholds['birth_time_threshold_low']
        visualizer.birth_time_threshold_high = self.thresholds['birth_time_threshold_high']
        
        # Recalculate cell properties with new thresholds
        visualizer.calculate_cell_properties()
        
        print("Thresholds applied to visualizer and properties recalculated.")
    
    def print_summary(self, df: pd.DataFrame):
        """
        Print threshold summary.
        
        Args:
            df: DataFrame with the data
        """
        if not self.thresholds:
            raise ValueError("No thresholds calculated. Run calculate_thresholds first.")
            
        print_threshold_summary(df, self.thresholds)


def validate_thresholds(df: pd.DataFrame, thresholds: dict, 
                       min_samples_per_category: int = 5) -> bool:
    """
    Validate that thresholds create reasonable category distributions.
    
    Args:
        df: DataFrame with the data
        thresholds: Dictionary of thresholds
        min_samples_per_category: Minimum samples required per category
        
    Returns:
        True if thresholds are valid, False otherwise
    """
    # Calculate cell size if needed
    if 'cell_size' not in df.columns:
        df_temp = df.copy()
        df_temp['cell_size'] = np.sqrt(df_temp['parent_x']**2 + df_temp['parent_y']**2 + df_temp['parent_z']**2)
    else:
        df_temp = df
    
    # Check birth time categories
    early_count = sum(df['Birth Time'] < thresholds['birth_time_threshold_low'])
    late_count = sum(df['Birth Time'] > thresholds['birth_time_threshold_high'])
    mid_count = len(df) - early_count - late_count
    
    if min(early_count, mid_count, late_count) < min_samples_per_category:
        print("Warning: Birth time categories have insufficient samples.")
        print(f"Early: {early_count}, Mid: {mid_count}, Late: {late_count}")
        return False
    
    # Check size categories
    small_count = sum(df_temp['cell_size'] < thresholds['size_threshold_small'])
    large_count = sum(df_temp['cell_size'] > thresholds['size_threshold_large'])
    medium_count = len(df_temp) - small_count - large_count
    
    if min(small_count, medium_count, large_count) < min_samples_per_category:
        print("Warning: Size categories have insufficient samples.")
        print(f"Small: {small_count}, Medium: {medium_count}, Large: {large_count}")
        return False
    
    return True