#!/usr/bin/env python3
"""
Neural Network Animator Module

Advanced interactive animation tools for temporal connectome networks using Plotly.
Creates sophisticated animated visualizations with hover information and directional edges.
"""

import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import logging
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class NeuralNetworkAnimator:
    """
    Creates advanced interactive animations of temporal neural networks.
    """
    
    def __init__(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, summary_df: pd.DataFrame):
        """
        Initialize the neural network animator.
        
        Args:
            nodes_df (pd.DataFrame): Node information data
            edges_df (pd.DataFrame): Temporal edge data  
            summary_df (pd.DataFrame): Summary statistics data
        """
        self.nodes_df = nodes_df.copy()
        self.edges_df = edges_df.copy()
        self.summary_df = summary_df.copy()
        
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
        
        # Edge visualization parameters
        self.MAX_EDGE_WIDTH = 6.0
        self.ELEC_GLYPH_THRESHOLD = 2
        
        self.setup_network_layout()
        
    def setup_network_layout(self):
        """Create consistent node positions via NetworkX spring layout."""
        logger.info("Setting up network layout...")
        
        G = nx.DiGraph()
        for _, n in self.nodes_df.iterrows():
            G.add_node(n['node_name'], type=n['node_type'])
            
        # Use combined set of edges from all timepoints for stable layout
        unique_edges = self.edges_df[['source_name', 'target_name']].drop_duplicates()
        for _, e in unique_edges.iterrows():
            G.add_edge(e['source_name'], e['target_name'])
            
        self.pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
        
        # Scale positions for visibility
        scale = 500
        for k in list(self.pos.keys()):
            x, y = self.pos[k]
            self.pos[k] = (x * scale, y * scale)
            
        logger.info(f"Layout for {len(self.pos)} nodes ready")
    
    def _scale_edge_width(self, weight: float, global_max_weight: float) -> float:
        """
        Logarithmic normalization of edge width to avoid runaway thickness.
        
        Args:
            weight (float): Edge weight
            global_max_weight (float): Maximum weight across all edges
            
        Returns:
            float: Scaled edge width
        """
        if global_max_weight <= 0:
            return 1.0
        width = 1.0 + (np.log1p(float(weight)) / np.log1p(float(global_max_weight))) * (self.MAX_EDGE_WIDTH - 1.0)
        return float(max(1.0, min(width, self.MAX_EDGE_WIDTH)))
    
    def create_animation_data(self) -> Tuple[List[Dict], List[Dict], List[int]]:
        """
        Prepare node frames and per-edge lists for animation.
        
        Returns:
            Tuple[List[Dict], List[Dict], List[int]]: Node frames, edge frames, and timepoints
        """
        logger.info("Preparing animation data...")
        timepoints = sorted(self.edges_df['timepoint'].unique())
        
        # Global max weight for normalization
        global_max_weight = max(1.0, float(self.edges_df['weight'].max()))
        
        node_frames = []
        edge_frames = []
        
        for tp in timepoints:
            current_edges = self.edges_df[self.edges_df['timepoint'] == tp].reset_index(drop=True)
            tp_summary = self.summary_df[self.summary_df['timepoint'] == tp].iloc[0]
            
            active_nodes = set(current_edges['source_name'].tolist() + current_edges['target_name'].tolist())
            
            # Calculate unique neighbors per node
            node_neighbors = {}
            for node in self.nodes_df['node_name']:
                outn = set(current_edges[current_edges['source_name'] == node]['target_name'].tolist())
                inn = set(current_edges[current_edges['target_name'] == node]['source_name'].tolist())
                node_neighbors[node] = len(outn.union(inn))
                
            # Build node frame
            nx_list, ny_list, ncolors, nsizes, ntext, nopacity = [], [], [], [], [], []
            n_names, n_types = [], []
            
            for _, nr in self.nodes_df.iterrows():
                name = nr['node_name']
                n_type = nr['node_type']
                if name not in self.pos:
                    continue
                    
                x, y = self.pos[name]
                nx_list.append(x)
                ny_list.append(y)
                ncolors.append(self.color_map.get(n_type, self.color_map['unknown']))
                n_names.append(name)
                n_types.append(n_type)
                
                degree = node_neighbors.get(name, 0)
                base_size = 10
                size = base_size + min(degree * 2.0, 30)  # Cap size
                nsizes.append(size)
                nopacity.append(1.0 if name in active_nodes else 0.25)
                
                # Create hover text with detailed information
                chem_conn = current_edges[
                    ((current_edges['source_name'] == name) | (current_edges['target_name'] == name))
                    & (current_edges['type'] == 'chemical')
                ]
                elec_conn = current_edges[
                    ((current_edges['source_name'] == name) | (current_edges['target_name'] == name))
                    & (current_edges['type'] == 'electrical')
                ]
                syn_sum = current_edges[
                    (current_edges['source_name'] == name) | (current_edges['target_name'] == name)
                ]['weight'].sum()
                
                txt = (f"<b>{name}</b> (ID: {nr.get('node_id', 'N/A')})<br>"
                       f"Type: {n_type.capitalize()}<br>"
                       f"Unique Neighbours: {degree}<br>"
                       f"Chemical: {len(chem_conn)}<br>"
                       f"Electrical: {len(elec_conn)}<br>"
                       f"Total Synapses: {syn_sum}<br>"
                       f"Stage: {tp_summary['stage']}<br>"
                       f"Status: {'Active' if name in active_nodes else 'Inactive'}")
                ntext.append(txt)
                
            node_frames.append({
                'x': nx_list, 'y': ny_list, 'marker_color': ncolors,
                'marker_size': nsizes, 'marker_opacity': nopacity,
                'text': ntext, 'customdata': list(zip(n_names, n_types))
            })
            
            # Process edges by type
            chem_edges = current_edges[current_edges['type'] == 'chemical']
            elec_edges = current_edges[current_edges['type'] == 'electrical']
            
            chem_edge_set = set(zip(chem_edges['source_name'], chem_edges['target_name']))
            elec_edge_set = set(zip(elec_edges['source_name'], elec_edges['target_name']))
            
            # Process chemical edges
            chem_list = []
            for _, e in chem_edges.iterrows():
                s = self.pos.get(e['source_name'])
                t = self.pos.get(e['target_name'])
                if s is None or t is None:
                    continue
                    
                w = self._scale_edge_width(e['weight'], global_max_weight)
                s_name, t_name = e['source_name'], e['target_name']
                is_bidirectional = (t_name, s_name) in chem_edge_set
                
                # Create hover text with bidirectional weight information
                if is_bidirectional:
                    reverse_edge = chem_edges[
                        (chem_edges['source_name'] == t_name) & (chem_edges['target_name'] == s_name)
                    ]
                    reverse_weight = reverse_edge['weight'].iloc[0] if not reverse_edge.empty else 'N/A'
                    
                    hover = (f"<b>Bidirectional Chemical</b><br>"
                             f"Nodes: {s_name} & {t_name}<br>"
                             f"Synapses ({s_name} → {t_name}): {e['weight']}<br>"
                             f"Synapses ({t_name} → {s_name}): {reverse_weight}<br>"
                             f"Stage: {e['stage']}<br>Time: {e['time_hours']}h")
                else:
                    hover = (f"<b>Chemical</b><br>Source: {s_name} (ID:{e['source_id']})<br>"
                             f"Target: {t_name} (ID:{e['target_id']})<br>Synapses: {e['weight']}<br>"
                             f"Stage: {e['stage']}<br>Time: {e['time_hours']}h")
                
                num_points = 20
                x_points = np.linspace(s[0], t[0], num_points)
                y_points = np.linspace(s[1], t[1], num_points)
                hover_texts_list = [hover] * num_points
                
                chem_list.append({
                    'x': x_points, 'y': y_points,
                    'width': w, 'hover_texts': hover_texts_list,
                    'arrow_start': s, 'arrow_end': t,
                    'source_name': e['source_name'], 'target_name': e['target_name']
                })
            
            # Process electrical edges
            elec_list = []
            for _, e in elec_edges.iterrows():
                s = self.pos.get(e['source_name'])
                t = self.pos.get(e['target_name'])
                if s is None or t is None:
                    continue
                    
                w = self._scale_edge_width(e['weight'], global_max_weight)
                s_name, t_name = e['source_name'], e['target_name']
                is_bidirectional = (t_name, s_name) in elec_edge_set
                
                # Create hover text with bidirectional weight information
                if is_bidirectional:
                    reverse_edge = elec_edges[
                        (elec_edges['source_name'] == t_name) & (elec_edges['target_name'] == s_name)
                    ]
                    reverse_weight = reverse_edge['weight'].iloc[0] if not reverse_edge.empty else 'N/A'
                    
                    hover = (f"<b>Bidirectional Electrical</b><br>Nodes: {s_name} & {t_name}<br>"
                             f"Synapses ({s_name}→{t_name}): {e['weight']}<br>"
                             f"Synapses ({t_name}→{s_name}): {reverse_weight}<br>"
                             f"Stage: {e['stage']}<br>Time: {e['time_hours']}h")
                else:
                    hover = (f"<b>Electrical</b><br>A: {s_name} (ID:{e['source_id']})<br>"
                             f"B: {t_name} (ID:{e['target_id']})<br>Synapses: {e['weight']}<br>"
                             f"Stage: {e['stage']}<br>Time: {e['time_hours']}h")
                
                num_points = 20
                x_points = np.linspace(s[0], t[0], num_points)
                y_points = np.linspace(s[1], t[1], num_points)
                hover_texts_list = [hover] * num_points
                
                elec_list.append({
                    'x': x_points, 'y': y_points,
                    'width': w, 'hover_texts': hover_texts_list,
                    'arrow_start': s, 'arrow_end': t,
                    'source_name': e['source_name'], 'target_name': e['target_name']
                })
            
            edge_frames.append({
                'chemical_edges': chem_list,
                'electrical_edges': elec_list,
                'timepoint': tp,
                'stage': tp_summary['stage'],
                'description': tp_summary['description'],
                'time_hours': tp_summary['time_hours'],
                'num_edges': tp_summary['num_edges'],
                'num_chemical': tp_summary['num_chemical'],
                'num_electrical': tp_summary['num_electrical'],
                'total_synapses': tp_summary['total_synapses'],
                'avg_synapses_per_edge': tp_summary['avg_synapses_per_edge']
            })
        
        logger.info(f"Prepared animation data for {len(timepoints)} timepoints")
        return node_frames, edge_frames, timepoints
    
    def create_interactive_plot(self) -> go.Figure:
        """
        Construct initial interactive plot with placeholders for animation.
        
        Returns:
            go.Figure: Interactive plot figure
        """
        logger.info("Creating initial interactive plot...")
        node_frames, edge_frames, timepoints = self.create_animation_data()
        fig = go.Figure()
        
        max_chem = max(len(f['chemical_edges']) for f in edge_frames)
        max_elec = max(len(f['electrical_edges']) for f in edge_frames)
        
        first_node = node_frames[0]
        first_edges = edge_frames[0]
        
        # Legend placeholder traces
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                                 line=dict(color='rgba(100,149,237,0.9)', width=4), name='Chemical'))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                                 line=dict(color='rgba(255,140,0,0.9)', width=4), name='Electrical'))
        
        # Add traces for chemical edges
        for i in range(max_chem):
            if i < len(first_edges['chemical_edges']):
                e = first_edges['chemical_edges'][i]
                fig.add_trace(go.Scatter(
                    x=e['x'], y=e['y'], mode='lines',
                    line=dict(color='rgba(100,149,237,0.9)', width=e['width']),
                    hoverinfo='text', text=e['hover_texts'],
                    hoverlabel=dict(bgcolor="rgba(100,149,237,0.8)", font_color="black"),
                    showlegend=False
                ))
            else:
                fig.add_trace(go.Scatter(x=[], y=[], mode='lines', showlegend=False))
        
        # Add traces for electrical edges
        for i in range(max_elec):
            if i < len(first_edges['electrical_edges']):
                e = first_edges['electrical_edges'][i]
                fig.add_trace(go.Scatter(
                    x=e['x'], y=e['y'], mode='lines',
                    line=dict(color='rgba(255,140,0,0.9)', width=e['width']),
                    hoverinfo='text', text=e['hover_texts'],
                    hoverlabel=dict(bgcolor="rgba(255,140,0,0.8)", font_color="black"),
                    showlegend=False
                ))
            else:
                fig.add_trace(go.Scatter(x=[], y=[], mode='lines', showlegend=False))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=first_node['x'], y=first_node['y'], mode='markers',
            marker=dict(size=first_node['marker_size'], color=first_node['marker_color'],
                        opacity=first_node['marker_opacity'], line=dict(color='white', width=2),
                        sizemode='diameter'),
            text=first_node['text'], textposition='top center',
            hovertemplate='%{text}<extra></extra>', name='neurons', showlegend=False
        ))
        
        # Add directional arrows for initial timepoint
        annotations = self._create_directional_annotations(first_edges)
        fig.update_layout(annotations=annotations)
        
        fig.update_layout(
            title={'text': f"<b>Neural Network Development</b><br><sub>Timepoint {timepoints[0]}: {edge_frames[0]['description']}</sub>",
                   'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=900, plot_bgcolor='rgba(240,240,240,0.95)', paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12)
        )
        
        fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
        fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
        
        self._max_chem = max_chem
        self._max_elec = max_elec
        
        return fig, node_frames, edge_frames, timepoints
    
    def _create_directional_annotations(self, edge_frame: Dict) -> List[Dict]:
        """
        Create directional arrow annotations for edges.
        
        Args:
            edge_frame (Dict): Edge frame data
            
        Returns:
            List[Dict]: List of annotation dictionaries
        """
        annotations = []
        
        # Create sets for fast lookup of bidirectional edges
        chem_edge_set = {(e['source_name'], e['target_name']) for e in edge_frame['chemical_edges']}
        elec_edge_set = {(e['source_name'], e['target_name']) for e in edge_frame['electrical_edges']}
        
        # Add arrows for directed chemical edges
        for e in edge_frame['chemical_edges']:
            s_name, t_name = e['source_name'], e['target_name']
            if (t_name, s_name) not in chem_edge_set:  # Directed edge
                s, t = e['arrow_start'], e['arrow_end']
                
                arrow_x = s[0] * 0.5 + t[0] * 0.5
                arrow_y = s[1] * 0.5 + t[1] * 0.5
                ax = s[0] * 0.6 + t[0] * 0.4
                ay = s[1] * 0.6 + t[1] * 0.4
                
                annotations.append(dict(
                    x=arrow_x, y=arrow_y, ax=ax, ay=ay,
                    xref='x', yref='y', axref='x', ayref='y',
                    showarrow=True, arrowhead=2, arrowsize=1.5, 
                    arrowwidth=max(1, e['width'] / 2.0),
                    arrowcolor='rgba(100,149,237,0.9)', opacity=0.9
                ))
        
        # Add arrows for directed electrical edges
        for e in edge_frame['electrical_edges']:
            s_name, t_name = e['source_name'], e['target_name']
            if (t_name, s_name) not in elec_edge_set:  # Directed edge
                s, t = e['arrow_start'], e['arrow_end']
                
                arrow_x = s[0] * 0.5 + t[0] * 0.5
                arrow_y = s[1] * 0.5 + t[1] * 0.5
                ax = s[0] * 0.6 + t[0] * 0.4
                ay = s[1] * 0.6 + t[1] * 0.4
                
                annotations.append(dict(
                    x=arrow_x, y=arrow_y, ax=ax, ay=ay,
                    xref='x', yref='y', axref='x', ayref='y',
                    showarrow=True, arrowhead=2, arrowsize=1.5,
                    arrowwidth=max(1, e['width'] / 2.0),
                    arrowcolor='rgba(255,140,0,0.9)', opacity=0.9
                ))
        
        return annotations
    
    def create_animation_buttons(self, fig: go.Figure, node_frames: List[Dict], 
                               edge_frames: List[Dict], timepoints: List[int]) -> go.Figure:
        """
        Create frames with identical trace count/order and add animation controls.
        
        Args:
            fig (go.Figure): Base figure
            node_frames (List[Dict]): Node frame data
            edge_frames (List[Dict]): Edge frame data
            timepoints (List[int]): List of timepoints
            
        Returns:
            go.Figure: Figure with animation controls
        """
        logger.info("Building frames and controls...")
        frames = []
        
        max_chem = getattr(self, "_max_chem", max(len(f['chemical_edges']) for f in edge_frames))
        max_elec = getattr(self, "_max_elec", max(len(f['electrical_edges']) for f in edge_frames))
        
        for i, tp in enumerate(timepoints):
            node_frame = node_frames[i]
            edge_frame = edge_frames[i]
            
            frame_data = []
            
            # Placeholders for legend traces
            frame_data.append(go.Scatter(x=[], y=[]))
            frame_data.append(go.Scatter(x=[], y=[]))
            
            # Chemical edges
            for slot in range(max_chem):
                if slot < len(edge_frame['chemical_edges']):
                    e = edge_frame['chemical_edges'][slot]
                    frame_data.append(go.Scatter(
                        x=e['x'], y=e['y'], mode='lines',
                        line=dict(color='rgba(100,149,237,0.9)', width=e['width']),
                        text=e['hover_texts'], hoverinfo='text',
                        hoverlabel=dict(bgcolor='rgba(100,149,237,0.8)', font_color='black'),
                        showlegend=False
                    ))
                else:
                    frame_data.append(go.Scatter(x=[], y=[], mode='lines'))
            
            # Electrical edges
            for slot in range(max_elec):
                if slot < len(edge_frame['electrical_edges']):
                    e = edge_frame['electrical_edges'][slot]
                    frame_data.append(go.Scatter(
                        x=e['x'], y=e['y'], mode='lines',
                        line=dict(color='rgba(255,140,0,0.9)', width=e['width']),
                        text=e['hover_texts'], hoverinfo='text',
                        hoverlabel=dict(bgcolor='rgba(255,140,0,0.8)', font_color='black'),
                        showlegend=False
                    ))
                else:
                    frame_data.append(go.Scatter(x=[], y=[], mode='lines'))
            
            # Nodes
            frame_data.append(go.Scatter(
                x=node_frame['x'], y=node_frame['y'], mode='markers',
                marker=dict(size=node_frame['marker_size'],
                            color=node_frame['marker_color'],
                            opacity=node_frame['marker_opacity'],
                            line=dict(color='white', width=2)),
                text=node_frame['text'], hoverinfo='text'
            ))
            
            # Create frame annotations for directional arrows
            frame_annotations = self._create_directional_annotations(edge_frame)
            
            frames.append(go.Frame(
                data=frame_data,
                name=str(tp),
                layout=go.Layout(
                    annotations=frame_annotations,
                    title=f"<b>Neural Network Development</b><br><sub>Timepoint {tp}: "
                          f"{edge_frame['stage']} - {edge_frame['description']} ({edge_frame['time_hours']}h) | "
                          f"{edge_frame['num_chemical']} Chemical + {edge_frame['num_electrical']} Electrical = "
                          f"{edge_frame['num_edges']} Total | {edge_frame['total_synapses']} Synapses</sub>"
                )
            ))
        
        fig.frames = frames
        
        # Add animation controls
        fig.update_layout(
            updatemenus=[dict(
                type="buttons", direction="left",
                buttons=[
                    dict(args=[None, {"frame": {"duration": 1600, "redraw": True},
                                      "fromcurrent": True, "transition": {"duration": 400}}],
                         label="▶ Play", method="animate"),
                    dict(args=[[None], {"frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate", "transition": {"duration": 0}}],
                         label="⏸ Pause", method="animate")
                ],
                pad={"r": 10, "t": 70},
                showactive=False, x=0.01, xanchor="left", y=0.95, yanchor="top")
            ],
            sliders=[dict(
                active=0, currentvalue={"prefix": "Timepoint: "}, pad={"t": 50},
                steps=[dict(args=[[tp], {"frame": {"duration": 500, "redraw": True},
                                         "mode": "immediate", "transition": {"duration": 500}}],
                            label=f"TP {tp}", method="animate") for tp in timepoints]
            )]
        )
        
        return fig
    
    def create_full_animation(self, save_path: Optional[str] = None) -> go.Figure:
        """
        Create the complete interactive neural network animation.
        
        Args:
            save_path (str, optional): Path to save the HTML file
            
        Returns:
            go.Figure: Complete animated figure
        """
        logger.info("Creating Interactive Neural Network Development Animation...")
        
        fig, node_frames, edge_frames, timepoints = self.create_interactive_plot()
        fig = self.create_animation_buttons(fig, node_frames, edge_frames, timepoints)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Animation saved to {save_path}")
        
        return fig


def create_neural_network_animation(nodes_csv_path: str,
                                  edges_csv_path: str,
                                  summary_csv_path: str,
                                  output_path: str = "neural_network_animation.html") -> go.Figure:
    """
    Create interactive neural network development animation.
    
    Args:
        nodes_csv_path (str): Path to nodes CSV file
        edges_csv_path (str): Path to edges CSV file  
        summary_csv_path (str): Path to summary CSV file
        output_path (str): Path to save animation HTML file
        
    Returns:
        go.Figure: Interactive animation figure
    """
    # Load data
    nodes_df = pd.read_csv(nodes_csv_path)
    edges_df = pd.read_csv(edges_csv_path)
    summary_df = pd.read_csv(summary_csv_path)
    
    # Create animator
    animator = NeuralNetworkAnimator(nodes_df, edges_df, summary_df)
    
    # Generate animation
    fig = animator.create_full_animation(output_path)
    
    return fig


def generate_network_summary(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive network summary statistics.
    
    Args:
        nodes_df (pd.DataFrame): Node information data
        edges_df (pd.DataFrame): Temporal edge data
        
    Returns:
        Dict[str, Any]: Summary statistics dictionary
    """
    logger.info("Generating Network Summary Statistics...")
    logger.info("=" * 60)
    
    # Overall statistics
    total_timepoints = len(edges_df['timepoint'].unique())
    total_nodes = len(nodes_df)
    max_edges = edges_df.groupby('timepoint').size().max()
    max_synapses = edges_df.groupby('timepoint')['weight'].sum().max()
    
    logger.info("NEURAL NETWORK DEVELOPMENT SUMMARY")
    logger.info(f"   Total Timepoints: {total_timepoints}")
    logger.info(f"   Total Neurons: {total_nodes}")
    logger.info(f"   Maximum Connections: {max_edges}")
    logger.info(f"   Maximum Total Synapses: {max_synapses}")
    
    chem_count = len(edges_df[edges_df['type'] == 'chemical'])
    elec_count = len(edges_df[edges_df['type'] == 'electrical'])
    logger.info(f"   Chemical/Electrical Ratio: {chem_count}/{elec_count}")
    
    # Development stages
    logger.info("\nDEVELOPMENT STAGES:")
    if 'stage' in edges_df.columns:
        for stage in edges_df['stage'].unique():
            stage_data = edges_df[edges_df['stage'] == stage]
            min_tp, max_tp = stage_data['timepoint'].min(), stage_data['timepoint'].max()
            if 'time_hours' in stage_data.columns:
                min_hours, max_hours = stage_data['time_hours'].min(), stage_data['time_hours'].max()
                max_synapses_stage = stage_data.groupby('timepoint')['weight'].sum().max()
                logger.info(f"   {stage}: Timepoints {min_tp}-{max_tp} ({min_hours}-{max_hours}h), Max synapses: {max_synapses_stage}")
    
    # Node type distribution
    logger.info("\nNEURON TYPE DISTRIBUTION:")
    type_counts = nodes_df['node_type'].value_counts()
    for node_type, count in type_counts.items():
        percentage = (count / total_nodes) * 100
        logger.info(f"   {node_type.capitalize()}: {count} ({percentage:.1f}%)")
    
    # Network complexity metrics for final timepoint
    logger.info("\nNETWORK COMPLEXITY METRICS:")
    final_timepoint = max(edges_df['timepoint'])
    final_edges = edges_df[edges_df['timepoint'] == final_timepoint]
    
    G = nx.from_pandas_edgelist(
        final_edges,
        source='source_name',
        target='target_name',
        create_using=nx.DiGraph()
    )
    
    density = nx.density(G)
    logger.info(f"   Network Density: {density:.3f}")
    
    if nx.is_weakly_connected(G):
        logger.info("   Network is weakly connected")
    else:
        logger.info("   Network is not connected")
    
    # Most important nodes (highest degree)
    degree_centrality = nx.degree_centrality(G)
    top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    
    logger.info("\nTOP 5 MOST CONNECTED NEURONS:")
    for i, (node, centrality) in enumerate(top_nodes, 1):
        node_type = nodes_df[nodes_df['node_name'] == node]['node_type'].iloc[0] if len(
            nodes_df[nodes_df['node_name'] == node]) > 0 else 'unknown'
        logger.info(f"   {i}. {node} ({node_type}) - Centrality: {centrality:.3f}")
    
    logger.info("=" * 60)
    
    return {
        'total_timepoints': total_timepoints,
        'total_nodes': total_nodes,
        'max_edges': max_edges,
        'max_synapses': max_synapses,
        'type_distribution': type_counts,
        'top_nodes': top_nodes,
        'final_density': density
    }