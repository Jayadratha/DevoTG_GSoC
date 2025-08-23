#!/usr/bin/env python3
"""
Connectome Visualization Module

Comprehensive visualization tools for temporal connectome networks including
static plots, interactive visualizations, and animations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import logging
from typing import Dict, Optional, Any
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)


class ConnectomeVisualizer:
    """
    Creates comprehensive visualizations for temporal connectome data.
    """
    
    def __init__(self, edges_df: pd.DataFrame, nodes_df: pd.DataFrame, summary_df: pd.DataFrame):
        """
        Initialize the connectome visualizer.
        
        Args:
            edges_df (pd.DataFrame): Temporal edge data
            nodes_df (pd.DataFrame): Node information data
            summary_df (pd.DataFrame): Summary statistics data
        """
        self.edges_df = edges_df.copy()
        self.nodes_df = nodes_df.copy()
        self.summary_df = summary_df.copy()
        self.timepoints = sorted(edges_df['timepoint'].unique())
        
        # Color map for neuron types
        self.color_map = {
            'inter': '#87CEEB',        # Sky Blue
            'modulatory': '#FFD700',   # Gold
            'sensory': '#98FB98',      # Pale Green
            'motor': '#FFA07A',        # Light Salmon
            'muscle': '#F08080',       # Light Coral
            'glia': '#DDA0DD',         # Plum
            'unknown': '#D3D3D3'       # Light Gray
        }
        
        # Set style
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        
    def create_growth_metrics_plot(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive network growth metrics visualization.
        
        Args:
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        logger.info("Creating network growth metrics plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Global Network Growth Metrics Over C. elegans Development', fontsize=16)
        
        # Plot 1: Number of Edges (Chemical vs. Electrical)
        sns.lineplot(ax=axes[0, 0], x='time_hours', y='num_chemical', 
                    data=self.summary_df, marker='o', label='Chemical')
        sns.lineplot(ax=axes[0, 0], x='time_hours', y='num_electrical', 
                    data=self.summary_df, marker='o', label='Electrical')
        axes[0, 0].set_title('Growth of Synapse Types')
        axes[0, 0].set_xlabel('Time (hours)')
        axes[0, 0].set_ylabel('Number of Connections')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # Plot 2: Total Synaptic Weight
        sns.lineplot(ax=axes[0, 1], x='time_hours', y='total_synapses', 
                    data=self.summary_df, marker='o', color='purple')
        axes[0, 1].set_title('Total Synaptic Weight (Strength)')
        axes[0, 1].set_xlabel('Time (hours)')
        axes[0, 1].set_ylabel('Total Synapses')
        axes[0, 1].grid(True)
        
        # Calculate additional metrics for plots 3 and 4
        growth_df = self._calculate_growth_metrics()
        
        # Plot 3: Network Density
        sns.lineplot(ax=axes[1, 0], x='time_hours', y='connection_density', 
                    data=growth_df, marker='o', color='green')
        axes[1, 0].set_title('Network Connection Density')
        axes[1, 0].set_xlabel('Time (hours)')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].grid(True)
        
        # Plot 4: Average Degree
        sns.lineplot(ax=axes[1, 1], x='time_hours', y='avg_in_degree', 
                    data=growth_df, marker='o', label='Avg. In-Degree')
        sns.lineplot(ax=axes[1, 1], x='time_hours', y='avg_out_degree', 
                    data=growth_df, marker='o', label='Avg. Out-Degree')
        axes[1, 1].set_title('Average Node Degree')
        axes[1, 1].set_xlabel('Time (hours)')
        axes[1, 1].set_ylabel('Average Degree')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Growth metrics plot saved to {save_path}")
            
        return fig
    
    def create_connection_types_plot(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create connection types evolution plot.
        
        Args:
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        logger.info("Creating connection types plot...")
        
        plt.figure(figsize=(12, 7))
        plt.stackplot(self.summary_df['time_hours'], 
                     self.summary_df['num_chemical'], 
                     self.summary_df['num_electrical'],
                     labels=['Chemical Synapses', 'Electrical Synapses'],
                     colors=['#4c72b0', '#dd8452'])
        plt.xlabel('Time (hours)')
        plt.ylabel('Number of Connections')
        plt.title('Evolution of Connection Types Over Development', fontsize=16)
        plt.legend(loc='upper left')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Connection types plot saved to {save_path}")
            
        return plt.gcf()
    
    def create_centrality_analysis_plot(self, timepoint: int = None, 
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Create centrality analysis plot for a specific timepoint.
        
        Args:
            timepoint (int, optional): Timepoint to analyze (default: last timepoint)
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        if timepoint is None:
            timepoint = max(self.timepoints)
            
        logger.info(f"Creating centrality analysis plot for timepoint {timepoint}...")
        
        # Get edges for the specified timepoint
        current_edges = self.edges_df[self.edges_df['timepoint'] == timepoint]
        G = nx.from_pandas_edgelist(current_edges, 'source_name', 'target_name', 
                                  create_using=nx.DiGraph())
        
        # Calculate degree centrality
        degree_centrality = nx.degree_centrality(G)
        centrality_df = pd.DataFrame(list(degree_centrality.items()), 
                                   columns=['neuron', 'degree_centrality']).sort_values('degree_centrality', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='degree_centrality', y='neuron', data=centrality_df.head(10), palette='viridis')
        plt.title(f'Top 10 Most Central Neurons at Timepoint {timepoint} (Degree Centrality)', fontsize=16)
        plt.xlabel('Degree Centrality')
        plt.ylabel('Neuron')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Centrality analysis plot saved to {save_path}")
            
        return plt.gcf()
    
    def create_network_graph_plot(self, timepoint: int = None, top_n_nodes: int = 50,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create static directed and weighted network graph visualization.
        
        Args:
            timepoint (int, optional): Timepoint to visualize (default: last timepoint)
            top_n_nodes (int): Number of most connected nodes to include
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        if timepoint is None:
            timepoint = max(self.timepoints)
            
        logger.info(f"Creating network graph for timepoint {timepoint}...")
        
        # Filter for the specified timepoint
        current_edges = self.edges_df[self.edges_df['timepoint'] == timepoint]
        
        # Create directed graph
        G = nx.from_pandas_edgelist(
            current_edges,
            source='source_name',
            target='target_name',
            edge_attr='weight',
            create_using=nx.DiGraph()
        )
        
        # Select top connected nodes
        degree_centrality = nx.degree_centrality(G)
        top_nodes = pd.Series(degree_centrality).nlargest(top_n_nodes).index.tolist()
        G_subgraph = G.subgraph(top_nodes)
        
        # Prepare node colors and sizes
        node_info_map = self.nodes_df.set_index('node_name')['node_type'].to_dict()
        node_colors = [self.color_map.get(node_info_map.get(node, 'unknown'), '#D3D3D3') 
                      for node in G_subgraph.nodes()]
        
        # Get edge weights and scale them
        weights = [G_subgraph[u][v]['weight'] for u, v in G_subgraph.edges()]
        max_weight = max(weights) if weights else 1
        scaled_weights = [0.1 + (w / max_weight * 10) for w in weights]
        
        # Create the plot
        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(G_subgraph, seed=42, k=0.8)
        
        # Draw nodes and edges
        nx.draw_networkx_nodes(G_subgraph, pos, node_size=500, node_color=node_colors, alpha=0.9)
        nx.draw_networkx_edges(
            G_subgraph,
            pos,
            width=scaled_weights,
            alpha=0.6,
            edge_color="black",
            arrowsize=12,
            arrowstyle='->'
        )
        nx.draw_networkx_labels(G_subgraph, pos, font_size=8)
        
        # Create legend
        legend_elements = [Patch(facecolor=color, edgecolor='black', label=label.capitalize())
                          for label, color in self.color_map.items()]
        plt.legend(handles=legend_elements, loc='upper right', title="Cell Types", fontsize=12)
        
        plt.title(f'Connectome of Top {top_n_nodes} Hubs (Timepoint {timepoint}) - Directed & Weighted', 
                 fontsize=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Network graph saved to {save_path}")
            
        return plt.gcf()
    
    def create_animated_network_growth(self, output_path: str = "network_animation.mp4") -> None:
        """
        Create animated visualization of network growth over time.
        
        Args:
            output_path (str): Path to save the animation
        """
        logger.info("Creating network growth animation...")
        
        fig, ax = plt.subplots(figsize=(15, 15))
        
        # Create base graph with all nodes
        all_edges = self.edges_df[['source_name', 'target_name']].drop_duplicates()
        G_base = nx.DiGraph()
        for _, edge in all_edges.iterrows():
            G_base.add_edge(edge['source_name'], edge['target_name'])
            
        # Add isolated nodes
        for node in self.nodes_df['node_name']:
            G_base.add_node(node)
            
        # Calculate consistent layout
        logger.info("Calculating layout...")
        pos = nx.spring_layout(G_base, seed=42, k=0.5)
        
        # Prepare node colors and types
        node_type_dict = self.nodes_df.set_index('node_name')['node_type'].to_dict()
        base_node_colors = [self.color_map.get(node_type_dict.get(node, 'unknown'), '#D3D3D3') 
                           for node in G_base.nodes()]
        
        def update(timepoint):
            ax.clear()
            
            # Get current edges
            current_edges = self.edges_df[self.edges_df['timepoint'] == timepoint]
            
            # Create graph for current timepoint
            G_t = nx.from_pandas_edgelist(
                current_edges,
                source='source_name',
                target='target_name',
                edge_attr='weight',
                create_using=nx.DiGraph()
            )
            
            # Draw all nodes faintly
            nx.draw_networkx_nodes(G_base, pos, node_size=50, node_color=base_node_colors, 
                                 alpha=0.2, ax=ax)
            
            # Draw active edges
            if G_t.edges():
                weights = [G_t[u][v]['weight'] for u, v in G_t.edges()]
                max_weight = max(weights) if weights else 1
                scaled_weights = [0.5 + (w / max_weight * 10) for w in weights]
                
                nx.draw_networkx_edges(
                    G_t, pos, width=scaled_weights, alpha=0.7, edge_color='gray',
                    arrowsize=10, arrowstyle='->', ax=ax
                )
            
            # Highlight active nodes
            active_nodes = list(G_t.nodes())
            active_node_colors = [self.color_map.get(node_type_dict.get(node, 'unknown'), '#D3D3D3') 
                                for node in active_nodes]
            nx.draw_networkx_nodes(G_t, pos, node_size=100, node_color=active_node_colors, 
                                 alpha=1.0, ax=ax)
            
            # Set title
            time_info = self.summary_df[self.summary_df['timepoint'] == timepoint].iloc[0]
            ax.set_title(
                f"Timepoint {timepoint}: {time_info['description']} ({time_info['time_hours']} hours)\n"
                f"{time_info['num_edges']} Directed Edges | {time_info['total_synapses']} Weighted Synapses",
                fontsize=16
            )
            ax.set_facecolor('white')
            fig.set_facecolor('white')
        
        # Create extended frame list for delays
        extended_frames = []
        for tp in self.timepoints:
            extended_frames.extend([tp] * 2)  # 2 seconds per timepoint
        
        # Add 5 extra frames at the end
        final_timepoint = self.timepoints[-1]
        extended_frames.extend([final_timepoint] * 5)
        
        # Create animation
        ani = animation.FuncAnimation(
            fig, update, frames=extended_frames, interval=1000, repeat_delay=3000
        )
        
        # Save animation
        logger.info(f"Saving animation to {output_path}...")
        ani.save(output_path, writer='ffmpeg', fps=1, dpi=150)
        plt.close()
        logger.info(f"Animation saved to {output_path}")
    
    def _calculate_growth_metrics(self) -> pd.DataFrame:
        """Calculate additional growth metrics for visualization."""
        growth_analysis = []
        all_nodes = set(self.nodes_df['node_name'].unique())
        
        for timepoint in self.timepoints:
            current_edges = self.edges_df[self.edges_df['timepoint'] == timepoint]
            
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
            
            # Get time info
            time_info = current_edges[['time_hours', 'stage']].iloc[0] if len(current_edges) > 0 else {'time_hours': 0, 'stage': 'Unknown'}
            
            growth_analysis.append({
                'timepoint': timepoint,
                'time_hours': time_info['time_hours'],
                'stage': time_info['stage'],
                'num_edges': len(edges),
                'connected_nodes': len(nodes_with_connections),
                'connection_density': density,
                'avg_in_degree': np.mean(list(in_degrees.values())) if in_degrees else 0,
                'avg_out_degree': np.mean(list(out_degrees.values())) if out_degrees else 0,
            })
            
        return pd.DataFrame(growth_analysis)


class InteractiveConnectomeVisualizer:
    """
    Creates interactive visualizations using Plotly for connectome analysis.
    """
    
    def __init__(self, edges_df: pd.DataFrame, nodes_df: pd.DataFrame, summary_df: pd.DataFrame):
        """
        Initialize the interactive visualizer.
        
        Args:
            edges_df (pd.DataFrame): Temporal edge data
            nodes_df (pd.DataFrame): Node information data
            summary_df (pd.DataFrame): Summary statistics data
        """
        self.edges_df = edges_df.copy()
        self.nodes_df = nodes_df.copy()
        self.summary_df = summary_df.copy()
        self.timepoints = sorted(edges_df['timepoint'].unique())
        
        # Color map for neuron types
        self.color_map = {
            'inter': '#87CEEB',
            'modulatory': '#FFD700',
            'sensory': '#98FB98',
            'motor': '#FFA07A',
            'muscle': '#F08080',
            'glia': '#DDA0DD',
            'unknown': '#D3D3D3'
        }
        
    def create_network_analysis_dashboard(self) -> go.Figure:
        """
        Create an interactive dashboard with network analysis metrics.
        
        Returns:
            go.Figure: Interactive dashboard figure
        """
        logger.info("Creating network analysis dashboard...")
        
        # Calculate network metrics
        metrics_data = []
        
        for tp in self.timepoints:
            current_edges = self.edges_df[self.edges_df['timepoint'] == tp]
            
            if len(current_edges) == 0:
                continue
                
            # Create networkx graph
            G = nx.from_pandas_edgelist(
                current_edges,
                source='source_name',
                target='target_name',
                edge_attr='weight',
                create_using=nx.DiGraph()
            )
            
            # Calculate metrics
            density = nx.density(G) if len(G.nodes()) > 0 else 0
            
            try:
                avg_path_length = nx.average_shortest_path_length(G.to_undirected())
            except Exception:
                avg_path_length = 0
                
            try:
                clustering = nx.average_clustering(G.to_undirected())
            except Exception:
                clustering = 0
                
            # Centrality measures
            betweenness = nx.betweenness_centrality(G)
            degree_centrality = nx.degree_centrality(G)
            
            metrics_data.append({
                'timepoint': tp,
                'density': density,
                'avg_path_length': avg_path_length,
                'clustering': clustering,
                'max_betweenness': max(betweenness.values()) if betweenness else 0,
                'max_degree_centrality': max(degree_centrality.values()) if degree_centrality else 0,
                'num_nodes': len(G.nodes()),
                'num_edges': len(G.edges())
            })
            
        metrics_df = pd.DataFrame(metrics_data)
        
        # Create dashboard
        dashboard = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Network Density Over Time', 'Clustering Coefficient',
                           'Centrality Measures', 'Network Growth'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Network density
        dashboard.add_trace(
            go.Scatter(x=metrics_df['timepoint'], y=metrics_df['density'],
                      mode='lines+markers', name='Density',
                      line=dict(color='#FF6B6B', width=3)),
            row=1, col=1
        )
        
        # Clustering coefficient
        dashboard.add_trace(
            go.Scatter(x=metrics_df['timepoint'], y=metrics_df['clustering'],
                      mode='lines+markers', name='Clustering',
                      line=dict(color='#4ECDC4', width=3)),
            row=1, col=2
        )
        
        # Centrality measures
        dashboard.add_trace(
            go.Scatter(x=metrics_df['timepoint'], y=metrics_df['max_betweenness'],
                      mode='lines+markers', name='Max Betweenness',
                      line=dict(color='#45B7D1', width=3)),
            row=2, col=1
        )
        
        dashboard.add_trace(
            go.Scatter(x=metrics_df['timepoint'], y=metrics_df['max_degree_centrality'],
                      mode='lines+markers', name='Max Degree Centrality',
                      line=dict(color='#96CEB4', width=3)),
            row=2, col=1
        )
        
        # Network growth
        dashboard.add_trace(
            go.Scatter(x=metrics_df['timepoint'], y=metrics_df['num_nodes'],
                      mode='lines+markers', name='Nodes',
                      line=dict(color='#FFEAA7', width=3)),
            row=2, col=2
        )
        
        dashboard.add_trace(
            go.Scatter(x=metrics_df['timepoint'], y=metrics_df['num_edges'],
                      mode='lines+markers', name='Edges',
                      line=dict(color='#DDA0DD', width=3)),
            row=2, col=2
        )
        
        # Update layout
        dashboard.update_layout(
            title="<b>Neural Network Analysis Dashboard</b>",
            height=700,
            showlegend=True,
            plot_bgcolor='rgba(240,240,240,0.95)'
        )
        
        # Update axes labels
        dashboard.update_xaxes(title_text="Timepoint")
        dashboard.update_yaxes(title_text="Density", row=1, col=1)
        dashboard.update_yaxes(title_text="Clustering Coefficient", row=1, col=2)
        dashboard.update_yaxes(title_text="Centrality Score", row=2, col=1)
        dashboard.update_yaxes(title_text="Count", row=2, col=2)
        
        return dashboard
    
    def create_node_importance_heatmap(self) -> go.Figure:
        """
        Create a heatmap showing node importance over time.
        
        Returns:
            go.Figure: Interactive heatmap figure
        """
        logger.info("Creating node importance heatmap...")
        
        # Calculate importance metrics
        importance_data = []
        all_nodes = self.nodes_df['node_name'].tolist()
        
        for tp in self.timepoints:
            current_edges = self.edges_df[self.edges_df['timepoint'] == tp]
            
            # Create graph
            G = nx.from_pandas_edgelist(
                current_edges,
                source='source_name',
                target='target_name',
                edge_attr='weight',
                create_using=nx.DiGraph()
            )
            
            # Calculate centrality measures
            if len(G.nodes()) > 1:
                try:
                    betweenness = nx.betweenness_centrality(G)
                    degree_cent = nx.degree_centrality(G)
                    closeness = nx.closeness_centrality(G)
                    
                    # Calculate weighted degree
                    weighted_degree = {}
                    for node in all_nodes:
                        if node in G.nodes():
                            in_weight = sum([G[u][node]['weight'] for u in G.predecessors(node)])
                            out_weight = sum([G[node][v]['weight'] for v in G.successors(node)])
                            weighted_degree[node] = in_weight + out_weight
                        else:
                            weighted_degree[node] = 0
                            
                except Exception as e:
                    logger.warning(f"Could not calculate centrality for timepoint {tp}: {e}")
                    betweenness = {node: 0 for node in all_nodes}
                    degree_cent = {node: 0 for node in all_nodes}
                    closeness = {node: 0 for node in all_nodes}
                    weighted_degree = {node: 0 for node in all_nodes}
            else:
                betweenness = {node: 0 for node in all_nodes}
                degree_cent = {node: 0 for node in all_nodes}
                closeness = {node: 0 for node in all_nodes}
                weighted_degree = {node: 0 for node in all_nodes}
            
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
                    'weighted_degree': weighted_degree.get(node, 0),
                    'node_type': node_type
                })
                
        importance_df = pd.DataFrame(importance_data)
        
        # Create heatmaps for different importance measures
        heatmap_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Betweenness Centrality', 'Degree Centrality',
                           'Closeness Centrality', 'Weighted Degree'),
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
                   [{"type": "heatmap"}, {"type": "heatmap"}]]
        )
        
        # Pivot data for heatmaps
        measures = ['betweenness', 'degree_centrality', 'closeness', 'weighted_degree']
        positions = [(1,1), (1,2), (2,1), (2,2)]
        
        for measure, (row, col) in zip(measures, positions):
            pivot_data = importance_df.pivot(index='node', columns='timepoint', values=measure)
            
            heatmap_fig.add_trace(
                go.Heatmap(
                    z=pivot_data.values,
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    colorscale='Viridis',
                    showscale=(row==1 and col==2),
                    hoverongaps=False,
                    hovertemplate='<b>%{y}</b><br>Timepoint: %{x}<br>Score: %{z:.3f}<extra></extra>'
                ),
                row=row, col=col
            )
            
        heatmap_fig.update_layout(
            title="<b>Node Importance Over Time</b><br><sub>Higher values indicate greater importance in the network</sub>",
            height=800,
            plot_bgcolor='white'
        )
        
        # Update axes
        for row in [1, 2]:
            for col in [1, 2]:
                heatmap_fig.update_xaxes(title_text="Timepoint", row=row, col=col)
                heatmap_fig.update_yaxes(title_text="Neuron", row=row, col=col)
                
        return heatmap_fig

    def create_interactive_network_explorer(self) -> go.Figure:
        """
        Create an interactive network explorer with filtering capabilities and multiple layouts.
        
        Returns:
            go.Figure: Interactive network explorer figure
        """
        logger.info("Creating interactive network explorer...")
        
        final_timepoint = max(self.timepoints)
        final_edges = self.edges_df[self.edges_df['timepoint'] == final_timepoint]
        
        G = nx.from_pandas_edgelist(
            final_edges,
            source='source_name',
            target='target_name',
            edge_attr=['weight', 'type'],
            create_using=nx.DiGraph()
        )
        
        nodes_df_copy = self.nodes_df.copy()
        if nodes_df_copy.index.name != 'node_name':
            nodes_df_copy.set_index('node_name', inplace=True)
        node_info_map = nodes_df_copy.to_dict('index')
        
        logger.info("Calculating network layouts...")
        layouts = {
            'Spring': nx.spring_layout(G, seed=42, k=0.5, iterations=50),
            'Kamada-Kawai': nx.kamada_kawai_layout(G),
            'Circular': nx.circular_layout(G),
            'Shell': nx.shell_layout(G),
        }
        pos = layouts['Spring']
        
        edge_x, edge_y, edge_hovertext = [], [], []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            edge_hovertext.append(
                f"Source: {edge[0]}<br>Target: {edge[1]}<br>Weight: {edge[2].get('weight', 'N/A')}"
            )
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1, color='#888'),
            hoverinfo='text',
            text=edge_hovertext,
            name='Connections'
        )
        
        node_x, node_y, node_text, node_hovertext, node_color, node_customdata = [], [], [], [], [], []
        
        for node in G.nodes():
            x, y = pos[node]
            info = node_info_map.get(node, {})
            node_type = info.get('node_type', 'unknown')
            degree = G.degree(node)
            
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_color.append(degree)
            node_customdata.append(node_type)
            node_hovertext.append(
                f"<b>{node}</b> ({node_type.capitalize()})<br>"
                f"Degree: {degree}<br>"
                f"In-Degree: {G.in_degree(node)}<br>"
                f"Out-Degree: {G.out_degree(node)}"
            )
            
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            text=node_text,
            hovertext=node_hovertext,
            hoverinfo='text',
            customdata=node_customdata,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                reversescale=True,
                color=node_color,
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Degree',
                    xanchor='left'
                ),
                line_width=2
            ),
            name='Neurons'
        )
        
        node_types = sorted(list(set(node_customdata)))
        
        filter_buttons = [
            dict(
                label="Show All",
                method="update",
                args=[{
                    'x': [None, node_x],
                    'y': [None, node_y],
                    'text': [None, node_text],
                    'hovertext': [None, node_hovertext],
                    'marker.color': [None, node_color],
                    'customdata': [None, node_customdata]
                }]
            )
        ]
        
        for node_type in node_types:
            filt_x, filt_y, filt_text, filt_hover, filt_color, filt_custom = [], [], [], [], [], []
            
            for i, full_type in enumerate(node_customdata):
                if full_type == node_type:
                    filt_x.append(node_x[i])
                    filt_y.append(node_y[i])
                    filt_text.append(node_text[i])
                    filt_hover.append(node_hovertext[i])
                    filt_color.append(node_color[i])
                    filt_custom.append(node_customdata[i])
            
            filter_buttons.append(
                dict(
                    label=node_type.capitalize(),
                    method="update",
                    args=[{
                        'x': [None, filt_x],
                        'y': [None, filt_y],
                        'text': [None, filt_text],
                        'hovertext': [None, filt_hover],
                        'marker.color': [None, filt_color],
                        'customdata': [None, filt_custom]
                    }]
                )
            )
        
        layout_buttons = []
        for layout_name, layout_pos in layouts.items():
            new_edge_x, new_edge_y = [], []
            for edge in G.edges():
                new_edge_x.extend([layout_pos[edge[0]][0], layout_pos[edge[1]][0], None])
                new_edge_y.extend([layout_pos[edge[0]][1], layout_pos[edge[1]][1], None])
            
            new_node_x = [layout_pos[node][0] for node in G.nodes()]
            new_node_y = [layout_pos[node][1] for node in G.nodes()]
            
            layout_buttons.append(
                dict(label=layout_name,
                     method="update",
                     args=[{'x': [new_edge_x, new_node_x],
                            'y': [new_edge_y, new_node_y]}]
                )
            )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        
        fig.update_layout(
            title='<b>Interactive Neural Network Explorer</b>',
            height=700,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#f0f0f0',
            updatemenus=[
                dict(
                    type="dropdown",
                    buttons=filter_buttons,
                    direction="down",
                    x=0.01, y=1, xanchor="left", yanchor="top",
                    active=0,
                ),
                dict(
                    type="dropdown",
                    buttons=layout_buttons,
                    direction="down",
                    x=0.17, y=1, xanchor="left", yanchor="top",
                    active=0,
                )
            ]
        )
        
        logger.info("Interactive network explorer created with filtering and layout options")
        
        return fig

def create_comprehensive_visualizations(edges_csv_path: str,
                                      nodes_csv_path: str,
                                      summary_csv_path: str,
                                      output_dir: str = "visualizations") -> Dict[str, Any]:
    """
    Create comprehensive visualizations for connectome data.
    
    Args:
        edges_csv_path (str): Path to edges CSV file
        nodes_csv_path (str): Path to nodes CSV file
        summary_csv_path (str): Path to summary statistics CSV file
        output_dir (str): Directory to save visualizations
        
    Returns:
        Dict[str, Any]: Dictionary containing created visualizations
    """
    # Load data
    edges_df = pd.read_csv(edges_csv_path)
    nodes_df = pd.read_csv(nodes_csv_path)
    summary_df = pd.read_csv(summary_csv_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create static visualizer
    static_viz = ConnectomeVisualizer(edges_df, nodes_df, summary_df)
    
    # Create interactive visualizer
    interactive_viz = InteractiveConnectomeVisualizer(edges_df, nodes_df, summary_df)
    
    visualizations = {}
    
    # Generate static visualizations
    logger.info("Generating static visualizations...")
    
    # Growth metrics plot
    growth_fig = static_viz.create_growth_metrics_plot(
        os.path.join(output_dir, 'network_growth_metrics.png')
    )
    visualizations['growth_metrics'] = growth_fig
    
    # Connection types plot
    types_fig = static_viz.create_connection_types_plot(
        os.path.join(output_dir, 'connection_types_evolution.png')
    )
    visualizations['connection_types'] = types_fig
    
    # Centrality analysis plot
    centrality_fig = static_viz.create_centrality_analysis_plot(
        save_path=os.path.join(output_dir, 'centrality_analysis.png')
    )
    visualizations['centrality_analysis'] = centrality_fig
    
    # Network graph plot
    network_fig = static_viz.create_network_graph_plot(
        save_path=os.path.join(output_dir, 'network_graph_directed_weighted.png')
    )
    visualizations['network_graph'] = network_fig
    
    # Network animation
    static_viz.create_animated_network_growth(
        os.path.join(output_dir, 'network_development_animation.mp4')
    )
    
    # Generate interactive visualizations
    logger.info("Generating interactive visualizations...")
    
    # Analysis dashboard
    dashboard = interactive_viz.create_network_analysis_dashboard()
    dashboard.write_html(os.path.join(output_dir, 'network_analysis_dashboard.html'))
    visualizations['dashboard'] = dashboard
    
    # Node importance heatmap
    heatmap = interactive_viz.create_node_importance_heatmap()
    heatmap.write_html(os.path.join(output_dir, 'node_importance_heatmap.html'))
    visualizations['heatmap'] = heatmap
    
    # Interactive network explorer
    explorer = interactive_viz.create_interactive_network_explorer()
    explorer.write_html(os.path.join(output_dir, 'interactive_network_explorer.html'))
    visualizations['explorer'] = explorer
    
    logger.info(f"All visualizations saved to {output_dir}")
    return visualizations