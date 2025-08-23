#!/usr/bin/env python3
"""
Network Analysis Module

Comprehensive analysis tools for temporal connectome networks including
growth analysis, connection stability, and network topology metrics.
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import logging
# from devotg.utils.make_json_safe import make_json_safe


logger = logging.getLogger(__name__)


class ConnectomeNetworkAnalyzer:
    """
    Analyzes temporal connectome networks for growth patterns, stability, and topology.
    """
    
    def __init__(self, edges_df: pd.DataFrame, nodes_df: pd.DataFrame):
        """
        Initialize the network analyzer.
        
        Args:
            edges_df (pd.DataFrame): Temporal edge data
            nodes_df (pd.DataFrame): Node information data
        """
        self.edges_df = edges_df.copy()
        self.nodes_df = nodes_df.copy()
        self.timepoints = sorted(edges_df['timepoint'].unique())
        
    def analyze_network_growth(self) -> pd.DataFrame:
        """
        Analyze network growth metrics over time.
        
        Returns:
            pd.DataFrame: Network growth analysis results
        """
        logger.info("Analyzing network growth over time...")
        
        growth_analysis = []
        all_nodes = set(self.nodes_df['node_name'].unique())
        
        for timepoint in self.timepoints:
            # Get edges for current timepoint
            current_edges = self.edges_df[self.edges_df['timepoint'] == timepoint]
            
            # Create NetworkX graph
            G = nx.from_pandas_edgelist(
                current_edges,
                source='source_name',
                target='target_name',
                edge_attr='weight',
                create_using=nx.DiGraph()
            )
            
            # Calculate network metrics
            edges = current_edges.to_dict('records')
            nodes_with_connections = set()
            for edge in edges:
                nodes_with_connections.add(edge['source_name'])
                nodes_with_connections.add(edge['target_name'])
                
            # Count degree distribution
            in_degrees = {}
            out_degrees = {}
            for edge in edges:
                out_degrees[edge['source_name']] = out_degrees.get(edge['source_name'], 0) + 1
                in_degrees[edge['target_name']] = in_degrees.get(edge['target_name'], 0) + 1
                
            # Connection density
            max_possible_edges = len(all_nodes) * (len(all_nodes) - 1)  # directed graph
            density = len(edges) / max_possible_edges if max_possible_edges > 0 else 0
            
            # Get summary statistics
            current_summary = current_edges.groupby('timepoint').agg({
                'weight': ['sum', 'mean', 'max'],
                'type': 'count'
            }).iloc[0]
            
            num_chemical = len(current_edges[current_edges['type'] == 'chemical'])
            num_electrical = len(current_edges[current_edges['type'] == 'electrical'])
            
            growth_analysis.append({
                'timepoint': timepoint,
                'time_hours': current_edges['time_hours'].iloc[0] if len(current_edges) > 0 else 0,
                'stage': current_edges['stage'].iloc[0] if len(current_edges) > 0 else 'Unknown',
                'num_edges': len(edges),
                'num_chemical': num_chemical,
                'num_electrical': num_electrical,
                'total_synapses': current_summary[('weight', 'sum')],
                'connected_nodes': len(nodes_with_connections),
                'connection_density': density,
                'avg_in_degree': np.mean(list(in_degrees.values())) if in_degrees else 0,
                'avg_out_degree': np.mean(list(out_degrees.values())) if out_degrees else 0,
                'max_edge_weight': current_summary[('weight', 'max')],
                'avg_edge_weight': current_summary[('weight', 'mean')]
            })
            
        growth_df = pd.DataFrame(growth_analysis)
        logger.info("Network growth analysis complete")
        return growth_df
    
    def analyze_connection_stability(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Analyze connection stability across development.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Stable, developmental, and variable connections
        """
        logger.info("Analyzing connection stability across development...")
        
        # Track which connections appear in which timepoints
        all_connections = set()
        connection_timepoints = {}
        
        for timepoint in self.timepoints:
            current_edges = self.edges_df[self.edges_df['timepoint'] == timepoint]
            for _, edge in current_edges.iterrows():
                connection_key = (edge['source_name'], edge['target_name'], edge['type'])
                all_connections.add(connection_key)
                if connection_key not in connection_timepoints:
                    connection_timepoints[connection_key] = []
                connection_timepoints[connection_key].append(timepoint)
                
        # Classify connections
        stable_connections = []
        variable_connections = []
        developmental_connections = []
        
        for connection, timepoints in connection_timepoints.items():
            source_name, target_name, conn_type = connection
            
            # Get node IDs
            source_id = self.nodes_df[self.nodes_df['node_name'] == source_name]['node_id'].iloc[0] if len(
                self.nodes_df[self.nodes_df['node_name'] == source_name]) > 0 else None
            target_id = self.nodes_df[self.nodes_df['node_name'] == target_name]['node_id'].iloc[0] if len(
                self.nodes_df[self.nodes_df['node_name'] == target_name]) > 0 else None
                
            connection_data = {
                'source_id': source_id,
                'target_id': target_id,
                'source_name': source_name,
                'target_name': target_name,
                'type': conn_type,
                'timepoints_present': len(timepoints),
                'first_appearance': min(timepoints),
                'last_appearance': max(timepoints),
                'timepoint_list': timepoints
            }
            
            if len(timepoints) >= 7:  # Present in most timepoints
                stable_connections.append(connection_data)
            elif max(timepoints) > min(timepoints) + 2:  # Developmental pattern
                developmental_connections.append(connection_data)
            else:  # Variable
                variable_connections.append(connection_data)
                
        stable_df = pd.DataFrame(stable_connections)
        developmental_df = pd.DataFrame(developmental_connections)
        variable_df = pd.DataFrame(variable_connections)
        
        logger.info("Connection stability classification:")
        logger.info(f"- Stable connections (present in ≥7 timepoints): {len(stable_connections)}")
        logger.info(f"- Developmental connections (dynamic pattern): {len(developmental_connections)}")
        logger.info(f"- Variable connections (inconsistent): {len(variable_connections)}")
        logger.info(f"- Total unique connections: {len(all_connections)}")
        
        return stable_df, developmental_df, variable_df
    
    def calculate_network_metrics(self, timepoint: int) -> Dict:
        """
        Calculate comprehensive network metrics for a specific timepoint.
        
        Args:
            timepoint (int): Timepoint to analyze
            
        Returns:
            Dict: Dictionary of network metrics
        """
        current_edges = self.edges_df[self.edges_df['timepoint'] == timepoint]
        
        if len(current_edges) == 0:
            return {}
            
        # Create NetworkX graph
        G = nx.from_pandas_edgelist(
            current_edges,
            source='source_name',
            target='target_name',
            edge_attr='weight',
            create_using=nx.DiGraph()
        )
        
        metrics = {
            'timepoint': timepoint,
            'num_nodes': len(G.nodes()),
            'num_edges': len(G.edges()),
            'density': nx.density(G)
        }
        
        if len(G.nodes()) > 1:
            # Centrality measures
            try:
                metrics['betweenness_centrality'] = nx.betweenness_centrality(G)
                metrics['degree_centrality'] = nx.degree_centrality(G)
                metrics['in_degree_centrality'] = nx.in_degree_centrality(G)
                metrics['out_degree_centrality'] = nx.out_degree_centrality(G)
                
                # Convert to undirected for some metrics
                G_undirected = G.to_undirected()
                metrics['closeness_centrality'] = nx.closeness_centrality(G_undirected)
                metrics['clustering_coefficient'] = nx.average_clustering(G_undirected)
                
                # Path lengths
                if nx.is_weakly_connected(G):
                    metrics['avg_shortest_path_length'] = nx.average_shortest_path_length(G_undirected)
                else:
                        # Calculate for largest connected component
                    largest_cc = max(nx.weakly_connected_components(G), key=len)
                    G_largest = G.subgraph(largest_cc).to_undirected()
                    metrics['avg_shortest_path_length'] = nx.average_shortest_path_length(G_largest)
                        
                # Weighted degree calculations
                weighted_in_degree = {}
                weighted_out_degree = {}
                for node in G.nodes():
                    weighted_in_degree[node] = sum([G[u][node]['weight'] for u in G.predecessors(node)])
                    weighted_out_degree[node] = sum([G[node][v]['weight'] for v in G.successors(node)])
                    
                metrics['weighted_in_degree'] = weighted_in_degree
                metrics['weighted_out_degree'] = weighted_out_degree
                
            except Exception as e:
                logger.warning(f"Could not calculate some metrics for timepoint {timepoint}: {e}")
                
        return metrics
    
    def get_top_nodes_by_centrality(self, timepoint: int, measure: str = 'degree', top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top nodes by centrality measure for a specific timepoint.
        
        Args:
            timepoint (int): Timepoint to analyze
            measure (str): Centrality measure ('degree', 'betweenness', 'closeness')
            top_k (int): Number of top nodes to return
            
        Returns:
            List[Tuple[str, float]]: List of (node_name, centrality_score) tuples
        """
        metrics = self.calculate_network_metrics(timepoint)
        
        if f'{measure}_centrality' not in metrics:
            return []
            
        centrality_dict = metrics[f'{measure}_centrality']
        top_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return top_nodes
    
    def analyze_node_importance_over_time(self) -> pd.DataFrame:
        """
        Analyze node importance evolution over all timepoints.
        
        Returns:
            pd.DataFrame: Node importance data over time
        """
        logger.info("Analyzing node importance over time...")
        
        importance_data = []
        all_nodes = self.nodes_df['node_name'].tolist()
        
        for tp in self.timepoints:
            metrics = self.calculate_network_metrics(tp)
            
            if not metrics:
                continue
                
            # Get centrality measures
            betweenness = metrics.get('betweenness_centrality', {})
            degree_cent = metrics.get('degree_centrality', {})
            closeness = metrics.get('closeness_centrality', {})
            weighted_in_degree = metrics.get('weighted_in_degree', {})
            weighted_out_degree = metrics.get('weighted_out_degree', {})
            
            # Compile importance data
            for node in all_nodes:
                node_type = self.nodes_df[self.nodes_df['node_name'] == node]['node_type'].iloc[0] if len(
                    self.nodes_df[self.nodes_df['node_name'] == node]) > 0 else 'unknown'
                    
                importance_data.append({
                    'timepoint': tp,
                    'node': node,
                    'betweenness': betweenness.get(node, 0),
                    'degree_centrality': degree_cent.get(node, 0),
                    'closeness': closeness.get(node, 0),
                    'weighted_in_degree': weighted_in_degree.get(node, 0),
                    'weighted_out_degree': weighted_out_degree.get(node, 0),
                    'total_weighted_degree': weighted_in_degree.get(node, 0) + weighted_out_degree.get(node, 0),
                    'node_type': node_type
                })
                
        importance_df = pd.DataFrame(importance_data)
        logger.info("Node importance analysis complete")
        return importance_df
    
    def calculate_network_efficiency(self, timepoint: int) -> Dict[str, float]:
        """
        Calculate network efficiency metrics.
        
        Args:
            timepoint (int): Timepoint to analyze
            
        Returns:
            Dict[str, float]: Network efficiency metrics
        """
        current_edges = self.edges_df[self.edges_df['timepoint'] == timepoint]
        
        if len(current_edges) == 0:
            return {}
            
        G = nx.from_pandas_edgelist(
            current_edges,
            source='source_name',
            target='target_name',
            edge_attr='weight',
            create_using=nx.DiGraph()
        )
        
        # Convert to undirected for efficiency calculations
        G_undirected = G.to_undirected()
        
        efficiency_metrics = {}
        
        try:
            # Global efficiency
            efficiency_metrics['global_efficiency'] = nx.global_efficiency(G_undirected)
            
            # Local efficiency
            efficiency_metrics['local_efficiency'] = nx.local_efficiency(G_undirected)
            
            # Network diameter (if connected)
            if nx.is_connected(G_undirected):
                efficiency_metrics['diameter'] = nx.diameter(G_undirected)
                efficiency_metrics['radius'] = nx.radius(G_undirected)
            else:
                # Use largest connected component
                largest_cc = max(nx.connected_components(G_undirected), key=len)
                G_largest = G_undirected.subgraph(largest_cc)
                efficiency_metrics['diameter'] = nx.diameter(G_largest)
                efficiency_metrics['radius'] = nx.radius(G_largest)
                
        except Exception as e:
            logger.warning(f"Could not calculate efficiency metrics for timepoint {timepoint}: {e}")
            
        return efficiency_metrics
    
    def detect_network_communities(self, timepoint: int, method: str = 'louvain') -> Dict:
        """
        Detect communities in the network using various algorithms.
        
        Args:
            timepoint (int): Timepoint to analyze
            method (str): Community detection method ('louvain', 'girvan_newman', 'label_propagation')
            
        Returns:
            Dict: Community detection results
        """
        current_edges = self.edges_df[self.edges_df['timepoint'] == timepoint]
        
        if len(current_edges) == 0:
            return {}
            
        G = nx.from_pandas_edgelist(
            current_edges,
            source='source_name',
            target='target_name',
            edge_attr='weight',
            create_using=nx.Graph()  # Use undirected for community detection
        )
        
        communities = {}
        
        try:
            if method == 'label_propagation':
                # Label propagation algorithm
                community_generator = nx.algorithms.community.label_propagation_communities(G)
                communities['communities'] = list(community_generator)
                communities['num_communities'] = len(communities['communities'])
                
            elif method == 'girvan_newman':
                # Girvan-Newman algorithm (computationally expensive)
                community_generator = nx.algorithms.community.girvan_newman(G)
                communities['communities'] = next(community_generator)  # Get first level
                communities['num_communities'] = len(communities['communities'])
                
        except Exception as e:
            logger.warning(f"Could not detect communities for timepoint {timepoint}: {e}")
            
        return communities
 
    
    def generate_comprehensive_network_report(self, save_path: Optional[str] = None) -> Dict:
        """
        Generate a comprehensive network analysis report.
        
        Args:
            save_path (str, optional): Path to save the report
            
        Returns:
            Dict: Comprehensive analysis results
        """
        logger.info("Generating comprehensive network report...")
        
        report = {
            'metadata': {
                'total_timepoints': len(self.timepoints),
                'timepoint_range': (min(self.timepoints), max(self.timepoints)),
                'total_nodes': len(self.nodes_df),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        # Network growth analysis
        report['growth_analysis'] = self.analyze_network_growth()
        
        # Connection stability analysis
        stable_df, dev_df, var_df = self.analyze_connection_stability()
        report['stability_analysis'] = {
            'stable_connections': stable_df,
            'developmental_connections': dev_df,
            'variable_connections': var_df,
            'summary': {
                'num_stable': len(stable_df),
                'num_developmental': len(dev_df),
                'num_variable': len(var_df)
            }
        }
        
        # Node importance analysis
        report['importance_analysis'] = self.analyze_node_importance_over_time()
        
        # Timepoint-specific metrics
        report['timepoint_metrics'] = {}
        for tp in self.timepoints:
            metrics = self.calculate_network_metrics(tp)
            efficiency = self.calculate_network_efficiency(tp)
            
            report['timepoint_metrics'][tp] = {
                'basic_metrics': metrics,
                'efficiency_metrics': efficiency
            }
            
        # Summary statistics
        growth_df = report['growth_analysis']
        report['summary_statistics'] = {
            'max_edges': growth_df['num_edges'].max(),
            'max_synapses': growth_df['total_synapses'].max(),
            'final_density': growth_df['connection_density'].iloc[-1] if len(growth_df) > 0 else 0,
            'growth_rate_edges': (growth_df['num_edges'].iloc[-1] - growth_df['num_edges'].iloc[0]) / len(growth_df) if len(growth_df) > 1 else 0,
            'growth_rate_synapses': (growth_df['total_synapses'].iloc[-1] - growth_df['total_synapses'].iloc[0]) / len(growth_df) if len(growth_df) > 1 else 0
        }
        
        # Save report if path provided
        if save_path:
            def make_json_safe(obj):
                """Recursively convert numpy/pandas types to JSON-safe types."""
                if isinstance(obj, dict):
                    return {str(k): make_json_safe(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_safe(v) for v in obj]
                elif isinstance(obj, (np.integer,)):
                    return int(obj)
                elif isinstance(obj, (np.floating,)):
                    return float(obj)
                elif isinstance(obj, (np.bool_)):
                    return bool(obj)
                elif isinstance(obj, (pd.Timestamp,)):
                    return obj.isoformat()
                else:
                    return obj

            # Convert DataFrames to dict for JSON serialization
            report_serializable = {}
            for key, value in report.items():
                if isinstance(value, pd.DataFrame):
                    report_serializable[key] = value.to_dict('records')
                elif isinstance(value, dict):
                    report_serializable[key] = {}
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, pd.DataFrame):
                            report_serializable[key][subkey] = subvalue.to_dict('records')
                        else:
                            report_serializable[key][subkey] = subvalue
                else:
                    report_serializable[key] = value

            # ✅ Ensure all keys/values are JSON safe
            report_serializable = make_json_safe(report_serializable) 

            import json
            with open(save_path, 'w') as f:
                json.dump(report_serializable, f, indent=2, default=str)
            logger.info(f"Report saved to {save_path}")
            
        logger.info("Comprehensive network report generated")
        return report


def analyze_connectome_network(edges_csv_path: str, 
                             nodes_csv_path: str,
                             output_dir: str = "statistics") -> ConnectomeNetworkAnalyzer:
    """
    Convenience function to perform comprehensive connectome network analysis.
    
    Args:
        edges_csv_path (str): Path to temporal edges CSV file
        nodes_csv_path (str): Path to nodes CSV file
        output_dir (str): Directory to save analysis results
        
    Returns:
        ConnectomeNetworkAnalyzer: Analyzer with completed analysis
    """
    # Load data
    edges_df = pd.read_csv(edges_csv_path)
    nodes_df = pd.read_csv(nodes_csv_path)
    
    # Create analyzer
    analyzer = ConnectomeNetworkAnalyzer(edges_df, nodes_df)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform analyses and save results
    growth_df = analyzer.analyze_network_growth()
    growth_df.to_csv(os.path.join(output_dir, 'network_growth_analysis.csv'), index=False)
    
    stable_df, dev_df, var_df = analyzer.analyze_connection_stability()
    stable_df.to_csv(os.path.join(output_dir, 'stable_connections.csv'), index=False)
    dev_df.to_csv(os.path.join(output_dir, 'developmental_connections.csv'), index=False)
    var_df.to_csv(os.path.join(output_dir, 'variable_connections.csv'), index=False)
    
    importance_df = analyzer.analyze_node_importance_over_time()
    importance_df.to_csv(os.path.join(output_dir, 'node_importance_over_time.csv'), index=False)
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_network_report(
        os.path.join(output_dir, 'comprehensive_network_report.json')
    )
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")
    return analyzer
                    