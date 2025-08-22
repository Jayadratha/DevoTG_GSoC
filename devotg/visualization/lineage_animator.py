"""
Lineage Tree Animation

This module contains functionality for creating animated lineage tree visualizations.
"""

import plotly.graph_objects as go
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class LineageAnimator:
    """
    Class for creating animated lineage tree visualizations.
    
    Provides interactive animations showing the path from root cells to target cells
    through the developmental lineage tree.
    """
    
    def __init__(self, visualizer):
        """
        Initialize the lineage animator.
        
        Args:
            visualizer: CellDivisionVisualizer instance containing the data
        """
        self.visualizer = visualizer
    
    def draw_animated_lineage_tree_full_edges(self, target_cell="ABpl", max_generations=None):
        """
        Create an animated lineage tree showing the path from root to target cell.
        
        Args:
            target_cell: Name of the target cell to trace lineage to
            max_generations: Maximum number of generations to include
            
        Returns:
            Plotly figure with animation controls
        """
        # Step 1: Build full graph from lineage
        G_full = nx.DiGraph()
        for parent, daughters in self.visualizer.cell_lineage.items():
            for daughter in daughters:
                G_full.add_edge(parent, daughter)

        if target_cell not in G_full:
            print(f"⚠ Cell '{target_cell}' not found.")
            return

        # Step 2: Find root
        def find_root(G, node):
            current = node
            while True:
                preds = list(G.predecessors(current))
                if not preds:
                    return current
                current = preds[0]

        root = find_root(G_full, target_cell)
        print(f"🔍 Root of '{target_cell}' is '{root}'")

        # Step 3: Use the generation map from the visualizer
        generation = self.visualizer.generation_map.copy()

        # If some nodes in the graph don't have generation assignments, assign them
        for node in G_full.nodes():
            if node not in generation:
                # Find the shortest path from root to this node to determine generation
                try:
                    path_from_root = nx.shortest_path(G_full, root, node)
                    generation[node] = len(path_from_root) - 1
                except nx.NetworkXNoPath:
                    generation[node] = 0  # Fallback

        max_tree_depth = max(generation.values()) if generation else 0
        print(f"Max tree depth: {max_tree_depth}")
        if max_generations is None or max_generations > max_tree_depth:
            max_generations = max_tree_depth

        # Step 4: Trim graph by generation
        allowed_nodes = {n for n, gen in generation.items() if gen <= max_generations}
        G = G_full.subgraph(allowed_nodes).copy()
        generation = {k: v for k, v in generation.items() if k in G}

        if target_cell not in G:
            print(f"⚠ Warning: '{target_cell}' is beyond generation {max_generations}. Will animate up to max.")
            # Find the deepest descendant within the allowed generations
            descendants = nx.descendants(G, root)
            if descendants:
                target_cell = max(descendants, key=lambda x: generation.get(x, 0))
            else:
                target_cell = root

        # Ensure there's a path from root to target_cell
        try:
            path = nx.shortest_path(G, root, target_cell)
        except nx.NetworkXNoPath:
            print(f"⚠ No path found from root '{root}' to target '{target_cell}'")
            return

        # Step 5: Radial layout by generation
        positions = {}
        nodes_by_level = {}
        for node, gen in generation.items():
            if node in G:  # Only include nodes that are in our trimmed graph
                nodes_by_level.setdefault(gen, []).append(node)

        for level, nodes in nodes_by_level.items():
            for i, node in enumerate(nodes):
                angle = 2 * np.pi * i / len(nodes)
                radius = 30 + 10 * level
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                z = -level * 20
                positions[node] = (x, y, z)

        # Step 6: Color nodes by generation
        cmap = plt.cm.get_cmap('Set3', max_tree_depth + 1)
        node_colors = {}
        for n in G.nodes():
            rgba = cmap(generation[n])[:3]  # RGB float tuple
            r, g, b = [int(c * 255) for c in rgba]  # Convert to 0-255
            node_colors[n] = f"rgb({r}, {g}, {b})"

        # Step 7: Animate lineage path
        frames = []
        for i in range(1, len(path) + 1):
            current_path = path[:i]
            edge_x, edge_y, edge_z = [], [], []
            for j in range(i - 1):
                u, v = current_path[j], current_path[j + 1]
                x0, y0, z0 = positions[u]
                x1, y1, z1 = positions[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
                edge_z += [z0, z1, None]

            node_trace = go.Scatter3d(
                x=[positions[n][0] for n in G.nodes()],
                y=[positions[n][1] for n in G.nodes()],
                z=[positions[n][2] for n in G.nodes()],
                mode='markers',
                text=list(G.nodes()),
                textposition='top center',
                marker=dict(
                    size=[12 if n == target_cell else 8 for n in G.nodes()],
                    color=[node_colors[n] for n in G.nodes()],
                    line=dict(width=1, color='black')
                ),
                hovertemplate="Cell: %{text}<br>Gen: %{customdata}<extra></extra>",
                customdata=[generation[n] for n in G.nodes()]
            )

            # Static edges (all)
            edge_lines_all = []
            for u, v in G.edges():
                x0, y0, z0 = positions[u]
                x1, y1, z1 = positions[v]
                edge_lines_all.append(go.Scatter3d(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    z=[z0, z1, None],
                    mode='lines',
                    line=dict(color=node_colors[u], width=2),
                    hoverinfo='none',
                    showlegend=False
                ))

            # Animated red path
            red_trace = go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='crimson', width=10),
                hoverinfo='none'
            )

            frames.append(go.Frame(data=[node_trace, red_trace] + edge_lines_all, name=f"step_{i}"))

        # Initial plot
        initial_edge_lines_all = []
        for u, v in G.edges():
            x0, y0, z0 = positions[u]
            x1, y1, z1 = positions[v]
            initial_edge_lines_all.append(go.Scatter3d(
                x=[x0, x1, None], y=[y0, y1, None], z=[z0, z1, None],
                mode='lines',
                line=dict(color=node_colors[u], width=2),
                hoverinfo='none',
                showlegend=False
            ))

        initial_red_trace = go.Scatter3d(
            x=[], y=[], z=[], mode='lines',
            line=dict(color='crimson', width=5),
            hoverinfo='none'
        )

        initial_node_trace = go.Scatter3d(
            x=[positions[n][0] for n in G.nodes()],
            y=[positions[n][1] for n in G.nodes()],
            z=[positions[n][2] for n in G.nodes()],
            mode='markers+text',
            text=list(G.nodes()),
            textposition='top center',
            marker=dict(
                size=[12 if n == target_cell else 8 for n in G.nodes()],
                color=[node_colors[n] for n in G.nodes()],
                line=dict(width=1, color='black')
            ),
            hovertemplate="Cell: %{text}<br>Gen: %{customdata}<extra></extra>",
            customdata=[generation[n] for n in G.nodes()]
        )

        fig = go.Figure(
            data=[initial_node_trace, initial_red_trace] + initial_edge_lines_all,
            layout=go.Layout(
                title=f"Lineage Tree to '{target_cell}' (≤ Gen {max_generations})",
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Generation',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                width=1000,
                height=800,
                updatemenus=[dict(
                    type="buttons",
                    buttons=[
                        dict(label="Play", method="animate", 
                             args=[None, {"frame": {"duration": 1000}, "fromcurrent": True}]),
                        dict(label="Pause", method="animate", 
                             args=[[None], {"mode": "immediate", "frame": {"duration": 0}}])
                    ]
                )],
                sliders=[dict(
                    steps=[
                        dict(method='animate', 
                             args=[[f"step_{i}"], {"frame": {"duration": 1000, "redraw": True}, 
                                                   "mode": "immediate"}], 
                             label=f'Step {i}')
                        for i in range(1, len(path) + 1)
                    ],
                    active=0
                )]
            ),
            frames=frames
        )

        print(f"✅ Created animation with {len(frames)} steps showing path: {' → '.join(path)}")
        return fig

    def create_simple_lineage_animation(self, target_cell="ABpl", max_generations=5):
        """
        Create a simplified animated lineage tree.
        
        Args:
            target_cell: Target cell to animate to
            max_generations: Maximum generations to include
            
        Returns:
            Plotly figure with animation
        """
        # Build graph
        G = nx.DiGraph()
        for parent, daughters in self.visualizer.cell_lineage.items():
            for daughter in daughters:
                G.add_edge(parent, daughter)

        if target_cell not in G:
            print(f"Cell '{target_cell}' not found.")
            return None

        # Find root and path
        def find_root(graph, node):
            current = node
            while True:
                predecessors = list(graph.predecessors(current))
                if not predecessors:
                    return current
                current = predecessors[0]

        root = find_root(G, target_cell)
        
        try:
            path = nx.shortest_path(G, root, target_cell)
        except nx.NetworkXNoPath:
            print(f"No path found from {root} to {target_cell}")
            return None

        # Create positions using spring layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Create animation frames
        frames = []
        for i in range(1, len(path) + 1):
            current_path = path[:i]
            current_node = path[i-1]
            
            # Node trace
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            node_text = list(G.nodes())
            node_colors = ['red' if node == current_node else 'lightblue' for node in G.nodes()]
            node_sizes = [15 if node == current_node else 10 for node in G.nodes()]
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_text,
                textposition='middle center',
                marker=dict(size=node_sizes, color=node_colors, line=dict(width=1, color='black')),
                hovertemplate='Cell: %{text}<extra></extra>'
            )
            
            # Edge traces
            edge_traces = []
            for u, v in G.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_color = 'red' if (u in current_path and v in current_path) else 'gray'
                edge_width = 3 if (u in current_path and v in current_path) else 1
                
                edge_trace = go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=edge_width, color=edge_color),
                    hoverinfo='none',
                    showlegend=False
                )
                edge_traces.append(edge_trace)
            
            frames.append(go.Frame(
                data=[node_trace] + edge_traces,
                name=f"frame_{i}",
                layout=dict(title=f"Lineage Animation: Step {i} - {current_node}")
            ))
        
        # Initial frame
        initial_node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            text=list(G.nodes()),
            textposition='middle center',
            marker=dict(size=10, color='lightblue', line=dict(width=1, color='black'))
        )
        
        initial_edge_traces = []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            initial_edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode='lines',
                line=dict(width=1, color='gray'),
                hoverinfo='none',
                showlegend=False
            ))
        
        fig = go.Figure(
            data=[initial_node_trace] + initial_edge_traces,
            layout=go.Layout(
                title=f"Lineage Tree Animation: {root} → {target_cell}",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                showlegend=False,
                hovermode='closest',
                width=800,
                height=600,
                updatemenus=[dict(
                    type="buttons",
                    buttons=[
                        dict(label="Play", method="animate", 
                             args=[None, {"frame": {"duration": 1500}, "fromcurrent": True}]),
                        dict(label="Pause", method="animate", 
                             args=[[None], {"mode": "immediate"}])
                    ]
                )],
                sliders=[dict(
                    steps=[dict(
                        method='animate',
                        args=[[f"frame_{i}"], {"frame": {"duration": 1500}}],
                        label=f"Step {i}"
                    ) for i in range(1, len(path) + 1)],
                    active=0,
                    currentvalue={"prefix": "Step: "}
                )]
            ),
            frames=frames
        )
        
        return fig