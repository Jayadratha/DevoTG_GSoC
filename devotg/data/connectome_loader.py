#!/usr/bin/env python3
"""
Connectome Dataset Loader

Module for downloading and processing Witvliet et al. C. elegans connectome datasets
from the ConnectomeToolbox repository.
"""

import os
import requests
import pandas as pd
import json
import pickle
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ConnectomeDatasetLoader:
    """
    Loads and processes C. elegans connectome datasets from Witvliet et al. 2021.
    
    This class handles downloading Excel files from the ConnectomeToolbox GitHub repository,
    processing the connectome data, and converting it into temporal graph formats.
    """
    
    def __init__(self, data_dir: str = "connectome_datasets"):
        """
        Initialize the ConnectomeDatasetLoader.
        
        Args:
            data_dir (str): Directory to store downloaded datasets
        """
        self.data_dir = data_dir
        self.github_base_url = "https://raw.githubusercontent.com/openworm/ConnectomeToolbox/main/cect/data/"
        
        # Excel files to download
        self.excel_files = [
            "witvliet_2020_1.xlsx",
            "witvliet_2020_2.xlsx", 
            "witvliet_2020_3.xlsx",
            "witvliet_2020_4.xlsx",
            "witvliet_2020_5.xlsx",
            "witvliet_2020_6.xlsx",
            "witvliet_2020_7.xlsx",
            "witvliet_2020_8.xlsx"
        ]
        
        # Developmental time points from Witvliet et al. paper
        self.time_points = {
            'Dataset1': {'stage': 'L1', 'age_hours': 0, 'description': 'Birth'},
            'Dataset2': {'stage': 'L1', 'age_hours': 5, 'description': 'L1 5hrs'},
            'Dataset3': {'stage': 'L1', 'age_hours': 8, 'description': 'L1 8hrs'},
            'Dataset4': {'stage': 'L1', 'age_hours': 16, 'description': 'L1 16hrs'},
            'Dataset5': {'stage': 'L2', 'age_hours': 23, 'description': 'L2 23hrs'},
            'Dataset6': {'stage': 'L3', 'age_hours': 27, 'description': 'L3 27hrs'},
            'Dataset7': {'stage': 'Adult', 'age_hours': 45, 'description': 'Adult 45hrs'},
            'Dataset8': {'stage': 'Adult', 'age_hours': 45, 'description': 'Adult 45hrs'}
        }
        
        # Neuron classification prefixes
        self.prefix_map = [
            ('CEPsh', 'glia'),      # Specific glia, must come before 'CEP'
            ('BWM', 'muscle'),      # General muscle group
            ('IL1', 'motor'),       # Specific motor neuron group
            ('IL2', 'sensory'),     # Specific sensory neuron group
            ('ADA', 'inter'),       ('ADE', 'modulatory'),  ('ADF', 'sensory'),
            ('ADL', 'sensory'),     ('AFD', 'sensory'),     ('AIA', 'inter'),
            ('AIB', 'inter'),       ('AIM', 'modulatory'),  ('AIN', 'inter'),
            ('AIY', 'inter'),       ('AIZ', 'inter'),       ('ALA', 'modulatory'),
            ('ALM', 'sensory'),     ('ALN', 'sensory'),     ('AQR', 'sensory'),
            ('ASE', 'sensory'),     ('ASG', 'sensory'),     ('ASH', 'sensory'),
            ('ASI', 'sensory'),     ('ASJ', 'sensory'),     ('ASK', 'sensory'),
            ('AUA', 'sensory'),     ('AVA', 'inter'),       ('AVB', 'inter'),
            ('AVD', 'inter'),       ('AVE', 'inter'),       ('AVF', 'modulatory'),
            ('AVH', 'modulatory'),  ('AVJ', 'modulatory'),  ('AVK', 'modulatory'),
            ('AVL', 'modulatory'),  ('AVM', 'sensory'),     ('AWA', 'sensory'),
            ('AWB', 'sensory'),     ('AWC', 'sensory'),     ('BAG', 'sensory'),
            ('BDU', 'inter'),       ('CEP', 'modulatory'),  # General 'CEP' group
            ('DVA', 'modulatory'),  ('DVC', 'inter'),       ('FLP', 'sensory'),
            ('GLR', 'glia'),        ('HSN', 'modulatory'),  ('OLL', 'sensory'),
            ('OLQ', 'sensory'),     ('PLN', 'sensory'),     ('PVC', 'inter'),
            ('PVN', 'modulatory'),  ('PVP', 'inter'),       ('PVQ', 'modulatory'),
            ('PVR', 'inter'),       ('PVT', 'inter'),       ('RIA', 'inter'),
            ('RIB', 'inter'),       ('RIC', 'modulatory'),  ('RID', 'modulatory'),
            ('RIF', 'inter'),       ('RIG', 'inter'),       ('RIH', 'inter'),
            ('RIM', 'inter'),       ('RIP', 'inter'),       ('RIR', 'inter'),
            ('RIS', 'modulatory'),  ('RIV', 'motor'),       ('RMD', 'motor'),
            ('RME', 'motor'),       ('RMF', 'motor'),       ('RMG', 'modulatory'),
            ('RMH', 'motor'),       ('SAA', 'sensory'),     ('SDQ', 'sensory'),
            ('SIA', 'motor'),       ('SIB', 'motor'),       ('SMB', 'motor'),
            ('SMD', 'motor'),       ('URA', 'motor'),       ('URB', 'sensory'),
            ('URX', 'sensory'),     ('URY', 'sensory')
        ]
        
        self.datasets = {}
        self.dtdg_data = None
        
    def download_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Download all Witvliet connectome datasets.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of loaded datasets
        """
        # Create directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        successful_downloads = []
        
        for excel_file in self.excel_files:
            try:
                url = self.github_base_url + excel_file
                logger.info(f"Downloading {excel_file}...")
                response = requests.get(url)
                
                if response.status_code == 200:
                    file_path = os.path.join(self.data_dir, excel_file)
                    # Save the file
                    with open(file_path, 'wb') as f:
                        f.write(response.content)

                    # Read the data
                    df = pd.read_excel(file_path)
                    dataset_num = excel_file.split('_')[2].split('.')[0]  # Extract number
                    self.datasets[f'Dataset{dataset_num}'] = df
                    successful_downloads.append(excel_file)
                    logger.info(f"✓ {excel_file}: {df.shape[0]} connections")
                else:
                    logger.error(f"✗ Failed to download {excel_file}: {response.status_code}")
            except Exception as e:
                logger.error(f"✗ Error with {excel_file}: {e}")
                
        logger.info(f"Successfully downloaded {len(successful_downloads)} datasets")
        return self.datasets

    def load_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load datasets from local Excel files into memory."""
        for excel_file in self.excel_files:
            file_path = os.path.join(self.data_dir, excel_file)
            if os.path.exists(file_path):
                df = pd.read_excel(file_path)
                dataset_num = excel_file.split('_')[2].split('.')[0]
                self.datasets[f'Dataset{dataset_num}'] = df
                logger.info(f"Loaded {excel_file}: {df.shape[0]} connections")
            else:
                logger.warning(f"File not found locally: {excel_file}")
        return self.datasets  
 
    def analyze_datasets(self) -> None:
        """Analyze the structure and content of downloaded datasets."""
        if not self.datasets:
            logger.warning("No datasets loaded. Call download_datasets() first.")
            return
            
        logger.info("Detailed analysis of the datasets:")
        logger.info("=" * 50)
        
        # Check structure of each dataset
        for dataset_name, df in self.datasets.items():
            logger.info(f"=== {dataset_name} ===")
            logger.info(f"Shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            logger.info(f"Synapse types: {df['type'].value_counts().to_dict()}")
            logger.info(f"Synapse count range: {df['synapses'].min()} - {df['synapses'].max()}")
            logger.info(f"Mean synapses per connection: {df['synapses'].mean():.2f}")
            logger.info("")
            
        # Get union of all neurons
        all_neurons = set()
        for df in self.datasets.values():
            all_neurons.update(df['pre'].unique())
            all_neurons.update(df['post'].unique())
            
        logger.info(f"Total unique neurons across all datasets: {len(all_neurons)}")
        
        # Connection count progression
        logger.info("\nConnection count progression:")
        for dataset_name, df in self.datasets.items():
            time_info = self.time_points[dataset_name]
            logger.info(f"{dataset_name} ({time_info['stage']} - {time_info['age_hours']}h): {len(df)} connections")
    
    def classify_neurons(self, all_neurons: List[str]) -> List[str]:
        """
        Classify neurons based on name prefixes.
        
        Args:
            all_neurons (List[str]): List of neuron names
            
        Returns:
            List[str]: List of neuron types corresponding to input neurons
        """
        neuron_types = []
        unclassified_nodes = []
        
        for node in all_neurons:
            node_type = 'unknown'  # Set a default type
            
            # Clean up the node name for consistent matching
            processed_node = node.replace(' ', '')
            
            # Find the first matching prefix in our ordered map
            for prefix, n_type in self.prefix_map:
                if processed_node.startswith(prefix):
                    node_type = n_type
                    break
                    
            neuron_types.append(node_type)
            if node_type == 'unknown':
                unclassified_nodes.append(node)
                
        if unclassified_nodes:
            logger.warning(f"{len(unclassified_nodes)} nodes could not be classified")
            logger.debug(f"Sample unclassified nodes: {unclassified_nodes[:10]}")
            
        # Log classification results
        type_counts = pd.Series(neuron_types).value_counts()
        logger.info("\nNode types distribution:")
        for node_type, count in type_counts.items():
            logger.info(f"{node_type}: {count}")
            
        return neuron_types
    
    def create_dtdg_dataset(self) -> Dict:
        """
        Create Discrete Time Dynamic Graph (DTDG) dataset.
        
        Returns:
            Dict: DTDG dataset structure
        """
        if not self.datasets:
            raise ValueError("No datasets loaded. Call download_datasets() first.")
            
        logger.info("Creating DTDG (Discrete Time Dynamic Graph) dataset...")
        logger.info("=" * 60)
        
        # Get all unique neurons
        all_neurons = set()
        for df in self.datasets.values():
            all_neurons.update(df['pre'].unique())
            all_neurons.update(df['post'].unique())
            
        all_nodes = sorted(list(all_neurons))
        node_to_id = {node: i for i, node in enumerate(all_nodes)}
        id_to_node = {i: node for i, node in enumerate(all_nodes)}
        
        # Classify neurons
        neuron_types = self.classify_neurons(all_nodes)
        
        # Create DTDG structure
        dtdg_data = {
            'metadata': {
                'description': 'C. elegans Developmental Connectome - Discrete Time Dynamic Graph',
                'source': 'Witvliet et al. 2021 - Connectomes across development reveal principles of brain maturation',
                'organism': 'Caenorhabditis elegans',
                'data_type': 'connectome',
                'num_nodes': len(all_nodes),
                'num_timepoints': len(self.datasets),
                'temporal_resolution': 'discrete',
                'node_labels': all_nodes,
                'time_points': self.time_points
            },
            'nodes': {
                'id': list(range(len(all_nodes))),
                'name': all_nodes,
                'type': neuron_types
            },
            'edges': [],
            'temporal_data': {}
        }
        
        # Process each timepoint
        for dataset_name, df in self.datasets.items():
            dataset_num = int(dataset_name.replace('Dataset', ''))
            time_info = self.time_points[dataset_name]
            
            # Create edge list for this timepoint
            edges_at_time = []
            for _, row in df.iterrows():
                pre_id = node_to_id[row['pre']]
                post_id = node_to_id[row['post']]
                weight = row['synapses']
                edge_type = row['type']
                
                edges_at_time.append({
                    'source': pre_id,
                    'target': post_id,
                    'weight': weight,
                    'type': edge_type
                })
                
            dtdg_data['temporal_data'][dataset_num] = {
                'time': time_info['age_hours'],
                'stage': time_info['stage'],
                'description': time_info['description'],
                'edges': edges_at_time,
                'num_edges': len(edges_at_time),
                'num_chemical': sum(1 for e in edges_at_time if e['type'] == 'chemical'),
                'num_electrical': sum(1 for e in edges_at_time if e['type'] == 'electrical'),
                'total_weight': sum(e['weight'] for e in edges_at_time)
            }
            
        logger.info("Temporal data summary:")
        for dataset_num, data in dtdg_data['temporal_data'].items():
            logger.info(f"T{dataset_num}: {data['time']}h ({data['stage']}) - "
                       f"{data['num_edges']} edges, {data['total_weight']} total synapses")
                       
        self.dtdg_data = dtdg_data
        logger.info("DTDG dataset created successfully!")
        logger.info(f"Structure: {len(dtdg_data['nodes']['id'])} nodes, "
                   f"{len(dtdg_data['temporal_data'])} timepoints")
        
        return dtdg_data
    
    def save_dtdg_dataset(self, output_dir: str = "dataset_processing") -> None:
        """
        Save DTDG dataset in multiple formats.
        
        Args:
            output_dir (str): Directory to save processed datasets
        """
        if self.dtdg_data is None:
            raise ValueError("No DTDG data available. Call create_dtdg_dataset() first.")
            
        logger.info("Saving DTDG dataset in multiple formats...")
        
        # Create directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save as JSON (human-readable)
        json_path = os.path.join(output_dir, 'dtdg_celegans_development.json')
        with open(json_path, 'w') as f:
            json.dump(self.dtdg_data, f, indent=2)
        logger.info(f"✓ Saved as JSON: {json_path}")
        
        # Save as pickle
        pkl_path = os.path.join(output_dir, 'dtdg_celegans_development.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.dtdg_data, f)
        logger.info(f"✓ Saved as pickle: {pkl_path}")
        
        # Create CSV format for easy analysis
        # Nodes CSV
        nodes_df = pd.DataFrame({
            'node_id': self.dtdg_data['nodes']['id'],
            'node_name': self.dtdg_data['nodes']['name'],
            'node_type': self.dtdg_data['nodes']['type']
        })
        nodes_csv_path = os.path.join(output_dir, 'dtdg_nodes.csv')
        nodes_df.to_csv(nodes_csv_path, index=False)
        logger.info(f"✓ Saved nodes: {nodes_csv_path}")
        
        # Edges CSV with temporal information
        edges_temporal = []
        for timepoint, data in self.dtdg_data['temporal_data'].items():
            for edge in data['edges']:
                edges_temporal.append({
                    'timepoint': timepoint,
                    'time_hours': data['time'],
                    'stage': data['stage'],
                    'source_id': edge['source'],
                    'target_id': edge['target'],
                    'source_name': self.dtdg_data['nodes']['name'][edge['source']],
                    'target_name': self.dtdg_data['nodes']['name'][edge['target']],
                    'weight': edge['weight'],
                    'type': edge['type']
                })
                
        edges_df = pd.DataFrame(edges_temporal)
        edges_csv_path = os.path.join(output_dir, 'dtdg_edges_temporal.csv')
        edges_df.to_csv(edges_csv_path, index=False)
        logger.info(f"✓ Saved temporal edges: {edges_csv_path}")
        
        # Create summary statistics
        summary_stats = []
        for timepoint, data in self.dtdg_data['temporal_data'].items():
            summary_stats.append({
                'timepoint': timepoint,
                'time_hours': data['time'],
                'stage': data['stage'],
                'description': data['description'],
                'num_edges': data['num_edges'],
                'num_chemical': data['num_chemical'],
                'num_electrical': data['num_electrical'],
                'total_synapses': data['total_weight'],
                'avg_synapses_per_edge': data['total_weight'] / data['num_edges'] if data['num_edges'] > 0 else 0
            })
            
        summary_df = pd.DataFrame(summary_stats)
        summary_csv_path = os.path.join(output_dir, 'dtdg_summary_statistics.csv')
        summary_df.to_csv(summary_csv_path, index=False)
        logger.info(f"✓ Saved summary statistics: {summary_csv_path}")
        
        logger.info("\nDataset summary:")
        logger.info(f"- Total nodes: {len(self.dtdg_data['nodes']['id'])}")
        logger.info(f"- Total timepoints: {len(self.dtdg_data['temporal_data'])}")
        logger.info(f"- Total edges across all timepoints: {len(edges_temporal)}")
        logger.info(f"- Temporal span: {min(data['time'] for data in self.dtdg_data['temporal_data'].values())} - "
                   f"{max(data['time'] for data in self.dtdg_data['temporal_data'].values())} hours")
        logger.info(f"- Developmental stages: {set(data['stage'] for data in self.dtdg_data['temporal_data'].values())}")


def load_connectome_datasets(data_dir: str = "data/connectome_datasets", 
                           output_dir: str = "output/dataset_processing") -> ConnectomeDatasetLoader:
    """
    Convenience function to load and process connectome datasets.
    
    Args:
        data_dir (str): Directory for raw dataset files
        output_dir (str): Directory for processed datasets
        
    Returns:
        ConnectomeDatasetLoader: Configured loader with processed data
    """
    loader = ConnectomeDatasetLoader(data_dir)
    # download datasets if not already present in data_dir
    logger.info(f"Checking for datasets in {data_dir}...")
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        logger.info("No datasets found. Downloading...")
        loader.download_datasets()
    else:
        logger.info("Datasets already present. Skipping download.")
        loader.load_datasets()
        
    # Load datasets
    if not loader.datasets:
        raise RuntimeError("Failed to download or load any datasets")   
    
    
        
    # Analyze datasets
    loader.analyze_datasets()
    
    # Create DTDG
    dtdg_data = loader.create_dtdg_dataset()
    
    # Save processed data
    loader.save_dtdg_dataset(output_dir)
    
    return loader
