"""
Dataset Loading Utilities

This module contains utilities for loading and preprocessing cell division datasets.
Preserves all original functionality while providing additional utilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple


class DatasetLoader:
    """
    Main class for loading and preprocessing cell division datasets.
    """
    
    def __init__(self, default_csv_path: Optional[str] = None):
        """
        Initialize the dataset loader.
        
        Args:
            default_csv_path: Default path to CSV file
        """
        self.default_csv_path = default_csv_path
        self.df = None
        self.metadata = {}
        
    def load_csv(self, csv_path: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load cell division data from CSV file.
        
        Args:
            csv_path: Path to CSV file (uses default if None)
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            Loaded DataFrame
        """
        if csv_path is None:
            csv_path = self.default_csv_path
            
        if csv_path is None:
            raise ValueError("No CSV path provided and no default path set")
            
        # Check if file exists
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        # Load with appropriate settings
        default_kwargs = {
            'sep': None,
            'engine': 'python'
        }
        default_kwargs.update(kwargs)
        
        try:
            self.df = pd.read_csv(csv_path, **default_kwargs)
            
            # Clean column names
            self.df.columns = self.df.columns.str.strip()
            
            # Store metadata
            self.metadata = {
                'csv_path': str(csv_path),
                'original_shape': self.df.shape,
                'columns': list(self.df.columns),
                'load_time': pd.Timestamp.now()
            }
            
            print(f"Loaded dataset: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return self.df
            
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")
    
    def validate_dataset(self, df: Optional[pd.DataFrame] = None) -> bool:
        """
        Validate that the dataset has required columns and data.
        
        Args:
            df: DataFrame to validate (uses loaded data if None)
            
        Returns:
            True if valid, False otherwise
        """
        if df is None:
            df = self.df
            
        if df is None:
            raise ValueError("No dataset loaded or provided")
            
        required_columns = [
            'Parent Cell', 'parent_x', 'parent_y', 'parent_z',
            'Daughter 1', 'Daughter 2', 'Birth Time'
        ]
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return False
            
        # Check for missing values in critical columns
        critical_columns = ['Parent Cell', 'Birth Time']
        for col in critical_columns:
            if df[col].isnull().any():
                print(f"Warning: Missing values found in critical column '{col}'")
                
        # Check data types
        numeric_columns = ['parent_x', 'parent_y', 'parent_z', 'Birth Time']
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    print(f"Converted column '{col}' to numeric")
                except Exception:
                    print(f"Warning: Could not convert column '{col}' to numeric")
                    
        # Check for reasonable data ranges
        if df['Birth Time'].min() < 0:
            print("Warning: Negative birth times found")
            
        # Check coordinate ranges
        coord_cols = ['parent_x', 'parent_y', 'parent_z']
        for col in coord_cols:
            if df[col].min() == df[col].max():
                print(f"Warning: No variation in coordinate column '{col}'")
                
        return True
    
    def get_dataset_summary(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Get comprehensive summary of the dataset.
        
        Args:
            df: DataFrame to summarize (uses loaded data if None)
            
        Returns:
            Dictionary with summary statistics
        """
        if df is None:
            df = self.df
            
        if df is None:
            raise ValueError("No dataset loaded or provided")
            
        summary = {
            'basic_info': {
                'shape': df.shape,
                'columns': list(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum()
            },
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric_stats'] = df[numeric_cols].describe().to_dict()
            
        # Cell-specific statistics
        if 'Parent Cell' in df.columns:
            summary['parent_cell_stats'] = {
                'unique_parents': df['Parent Cell'].nunique(),
                'total_divisions': len(df),
                'most_frequent_parent': df['Parent Cell'].value_counts().head(5).to_dict()
            }
            
        if 'Daughter 1' in df.columns and 'Daughter 2' in df.columns:
            all_daughters = pd.concat([df['Daughter 1'], df['Daughter 2']]).dropna()
            summary['daughter_cell_stats'] = {
                'unique_daughters': all_daughters.nunique(),
                'most_frequent_daughters': all_daughters.value_counts().head(5).to_dict()
            }
            
        # Time range
        if 'Birth Time' in df.columns:
            summary['time_range'] = {
                'min_time': df['Birth Time'].min(),
                'max_time': df['Birth Time'].max(),
                'time_span': df['Birth Time'].max() - df['Birth Time'].min(),
                'unique_time_points': df['Birth Time'].nunique()
            }
            
        # Coordinate ranges
        coord_cols = ['parent_x', 'parent_y', 'parent_z']
        coord_summary = {}
        for col in coord_cols:
            if col in df.columns:
                coord_summary[col] = {
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'range': df[col].max() - df[col].min(),
                    'std': df[col].std()
                }
        if coord_summary:
            summary['coordinate_stats'] = coord_summary
            
        return summary
    
    def print_dataset_info(self, df: Optional[pd.DataFrame] = None):
        """
        Print comprehensive dataset information.
        
        Args:
            df: DataFrame to analyze (uses loaded data if None)
        """
        if df is None:
            df = self.df
            
        if df is None:
            raise ValueError("No dataset loaded or provided")
            
        print("=" * 60)
        print("DATASET INFORMATION")
        print("=" * 60)
        
        print("\nBasic Info:")
        print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\nColumns:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col} ({df[col].dtype})")
            
        print("\nMissing Values:")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            print("  No missing values found")
        else:
            for col, count in missing[missing > 0].items():
                pct = count / len(df) * 100
                print(f"  {col}: {count} ({pct:.1f}%)")
                
        # Cell statistics
        if 'Parent Cell' in df.columns:
            print("\nCell Division Statistics:")
            print(f"  Unique parent cells: {df['Parent Cell'].nunique()}")
            print(f"  Total divisions: {len(df)}")
            
            parent_counts = df['Parent Cell'].value_counts()
            print("  Most active parent cells:")
            for cell, count in parent_counts.head(5).items():
                print(f"    {cell}: {count} divisions")
                
        # Time statistics
        if 'Birth Time' in df.columns:
            print("\nTemporal Statistics:")
            print(f"  Time range: {df['Birth Time'].min():.1f} - {df['Birth Time'].max():.1f}")
            print(f"  Time span: {df['Birth Time'].max() - df['Birth Time'].min():.1f}")
            print(f"  Unique time points: {df['Birth Time'].nunique()}")
            
        # Spatial statistics
        coord_cols = ['parent_x', 'parent_y', 'parent_z']
        if all(col in df.columns for col in coord_cols):
            print("\nSpatial Statistics:")
            for col in coord_cols:
                print(f"  {col}: {df[col].min():.1f} to {df[col].max():.1f} "
                      f"(range: {df[col].max() - df[col].min():.1f})")
                      
        print("=" * 60)
    
    def clean_dataset(self, df: Optional[pd.DataFrame] = None, 
                     fill_missing: bool = True, 
                     remove_duplicates: bool = True) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and duplicates.
        
        Args:
            df: DataFrame to clean (uses loaded data if None)
            fill_missing: Whether to fill missing values
            remove_duplicates: Whether to remove duplicate rows
            
        Returns:
            Cleaned DataFrame
        """
        if df is None:
            df = self.df
            
        if df is None:
            raise ValueError("No dataset loaded or provided")
            
        df_clean = df.copy()
        original_shape = df_clean.shape
        
        # Handle missing values
        if fill_missing:
            # Fill missing string columns with empty strings
            string_cols = df_clean.select_dtypes(include=['object']).columns
            for col in string_cols:
                if col in ['Daughter 1', 'Daughter 2']:
                    df_clean[col] = df_clean[col].fillna('')
                    
            # Fill missing numeric columns with appropriate values
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_clean[col].isnull().any():
                    if col == 'Birth Time':
                        # For birth time, use forward fill or median
                        df_clean[col] = df_clean[col].fillna(method='ffill').fillna(df_clean[col].median())
                    else:
                        # For coordinates, use median
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                        
        # Remove duplicates
        if remove_duplicates:
            df_clean = df_clean.drop_duplicates()
            
        # Clean column names
        df_clean.columns = df_clean.columns.str.strip()
        
        # Convert data types
        numeric_cols = ['parent_x', 'parent_y', 'parent_z', 'Birth Time']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
        print(f"Dataset cleaned: {original_shape} → {df_clean.shape}")
        
        return df_clean
    
    def export_processed_data(self, df: pd.DataFrame, 
                             output_path: str, 
                             include_metadata: bool = True):
        """
        Export processed data to CSV with optional metadata.
        
        Args:
            df: DataFrame to export
            output_path: Path for output file
            include_metadata: Whether to include metadata file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export main data
        df.to_csv(output_path, index=False)
        print(f"Data exported to: {output_path}")
        
        # Export metadata if requested
        if include_metadata and self.metadata:
            metadata_path = output_path.with_suffix('.metadata.json')
            import json
            
            # Add processing metadata
            export_metadata = self.metadata.copy()
            export_metadata.update({
                'export_time': str(pd.Timestamp.now()),
                'export_shape': df.shape,
                'export_path': str(output_path)
            })
            
            with open(metadata_path, 'w') as f:
                json.dump(export_metadata, f, indent=2, default=str)
            print(f"Metadata exported to: {metadata_path}")


def load_sample_data() -> pd.DataFrame:
    """
    Load sample data for testing purposes.
    
    Returns:
        Sample DataFrame with cell division data
    """
    # Create sample data that matches the expected format
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'Parent Cell': [f'P{i}' for i in range(n_samples)],
        'parent_x': np.random.uniform(150, 550, n_samples),
        'parent_y': np.random.uniform(150, 350, n_samples),
        'parent_z': np.random.uniform(10, 20, n_samples),
        'Daughter 1': [f'D{i}_1' for i in range(n_samples)],
        'Daughter 2': [f'D{i}_2' for i in range(n_samples)],
        'Birth Time': np.sort(np.random.uniform(0, 600, n_samples))
    }
    
    return pd.DataFrame(data)


def quick_load_and_validate(csv_path: str) -> Tuple[pd.DataFrame, bool]:
    """
    Quick function to load and validate a dataset.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Tuple of (DataFrame, is_valid)
    """
    loader = DatasetLoader()
    
    try:
        df = loader.load_csv(csv_path)
        is_valid = loader.validate_dataset(df)
        
        if is_valid:
            print("Dataset loaded and validated successfully!")
        else:
            print("Dataset loaded but validation failed.")
            
        return df, is_valid
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, False