"""
Cell Division Visualizer

This module contains the main visualization functionality for C. elegans cell division data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go
import io


class CellDivisionVisualizer:
    """
    Main class for visualizing C. elegans cell division data.
    
    Provides static plots, interactive visualizations, and animations
    for analyzing cell division patterns.
    """
    
    def __init__(self, csv_data=None, size_threshold_small=50.0, size_threshold_large=200.0,
                 birth_time_threshold_low=50.0, birth_time_threshold_high=500.0):
        """
        Initialize the cell division visualizer

        Parameters:
        csv_data: pandas DataFrame or CSV string/file path
        size_threshold_small: small threshold for cell size-based coloring
        size_threshold_large: large threshold for cell size-based coloring
        birth_time_threshold_low: low threshold for time-based coloring
        birth_time_threshold_high: high threshold for time-based coloring
        """
        self.size_threshold_small = size_threshold_small
        self.size_threshold_large = size_threshold_large
        self.birth_time_threshold_low = birth_time_threshold_low
        self.birth_time_threshold_high = birth_time_threshold_high
        self.cell_data = []
        self.division_events = []
        self.active_cells = {}
        self.cell_lineage = {}
        self.generation_map = {}

        # Color schemes
        self.size_colors = {
            'small': '#3498db',      # Blue for small cells
            'medium': '#f39c12',     # Orange for medium cells
            'large': '#e74c3c'       # Red for large cells
        }

        self.time_colors = {
            'early': '#2ecc71',      # Green for early divisions
            'mid': '#9b59b6',        # Purple for mid-time divisions
            'late': '#e67e22'        # Orange for late divisions
        }

        if csv_data is not None:
            self.load_data(csv_data)

    def load_data(self, data):
        """Load cell division data from CSV"""
        if isinstance(data, str):
            if '\n' in data:  # CSV string
                self.df = pd.read_csv(io.StringIO(data))
            else:  # File path
                self.df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.df = data
        else:
            raise ValueError("Data must be a CSV file path, CSV string, or pandas DataFrame")

        # Clean column names
        self.df.columns = self.df.columns.str.strip()

        # Parse the data
        self.parse_division_events()
        self.calculate_cell_properties()

    def parse_division_events(self):
        """Parse division events from the dataframe"""
        self.division_events = []

        for _, row in self.df.iterrows():
            event = {
                'parent_cell': row['Parent Cell'].strip(),
                'parent_x': float(row['parent_x']),
                'parent_y': float(row['parent_y']),
                'parent_z': float(row['parent_z']),
                'daughter1': row['Daughter 1'].strip(),
                'daughter2': row['Daughter 2'].strip(),
                'birth_time': float(row['Birth Time']),
                'processed': False
            }
            self.division_events.append(event)

        # Sort by birth time
        self.division_events.sort(key=lambda x: x['birth_time'])

        # Build lineage tree
        self.build_lineage_tree()

    def build_lineage_tree(self):
        """Build the cell lineage tree and assign generations"""
        self.cell_lineage = {}
        self.generation_map = {}

        # Find root cell (appears as parent but not as daughter)
        daughters = set()
        parents = set()

        for event in self.division_events:
            parents.add(event['parent_cell'])
            daughters.add(event['daughter1'])
            daughters.add(event['daughter2'])

        root_cells = parents - daughters

        # Assign generations starting from root
        for root in root_cells:
            self.generation_map[root] = 0
            self._assign_generations(root, 0)

    def _assign_generations(self, cell_name, generation):
        """Recursively assign generations to cells"""
        for event in self.division_events:
            if event['parent_cell'] == cell_name:
                self.generation_map[event['daughter1']] = generation + 1
                self.generation_map[event['daughter2']] = generation + 1
                self.cell_lineage[cell_name] = [event['daughter1'], event['daughter2']]

                # Recurse for daughters
                self._assign_generations(event['daughter1'], generation + 1)
                self._assign_generations(event['daughter2'], generation + 1)

    def calculate_cell_properties(self):
        """Calculate cell properties for coloring"""
        # Calculate cell distances from origin (as size proxy)
        for event in self.division_events:
            distance = np.sqrt(event['parent_x']**2 + event['parent_y']**2 + event['parent_z']**2)
            event['cell_size'] = distance

            # Assign size category
            if distance < self.size_threshold_small:
                event['size_category'] = 'small'
            elif distance < self.size_threshold_large:
                event['size_category'] = 'medium'
            else:
                event['size_category'] = 'large'

            # Assign time category
            max_time = max(e['birth_time'] for e in self.division_events)
            if event['birth_time'] < self.birth_time_threshold_low:
                event['time_category'] = 'early'
            elif event['birth_time'] < self.birth_time_threshold_high:
                event['time_category'] = 'mid'
            else:
                event['time_category'] = 'late'

    def get_cell_color(self, event, color_by='generation'):
        """Get color for a cell based on the specified criteria"""
        if color_by == 'size':
            return self.size_colors[event['size_category']]
        elif color_by == 'time':
            return self.time_colors[event['time_category']]
        elif color_by == 'generation':
            generation = self.generation_map.get(event['parent_cell'], 0)
            # Use a color palette based on generation
            colors = plt.cm.Set3(np.linspace(0, 1, 12))
            return colors[generation % len(colors)]
        else:
            return '#3498db'  # Default blue

    def create_static_plot(self, time_point=None, color_by='generation', figsize=(12, 8)):
        """Create a static 3D plot at a specific time point"""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Get events up to time_point
        if time_point is None:
            time_point = max(e['birth_time'] for e in self.division_events)

        active_events = [e for e in self.division_events if e['birth_time'] <= time_point]

        # Plot cells
        for event in active_events:
            color = self.get_cell_color(event, color_by)
            size = 100 + event['cell_size'] * 2  # Scale size for visibility

            # Plot parent cell
            ax.scatter(event['parent_x'], event['parent_y'], event['parent_z'],
                      c=[color], s=size, alpha=0.7, edgecolors='black', linewidth=0.5)

            # Add cell label
            ax.text(event['parent_x'], event['parent_y'], event['parent_z'] + 2,
                   event['parent_cell'], fontsize=8)

        # Customize plot
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        ax.set_title(f'C. elegans Cell Division at Time {time_point}\nColored by {color_by.title()}')

        # Add legend based on color scheme
        self._add_legend(ax, color_by)

        plt.tight_layout()
        return fig, ax

    def _add_legend(self, ax, color_by):
        """Add appropriate legend to the plot"""
        if color_by == 'size':
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=color, markersize=10, label=category.title())
                             for category, color in self.size_colors.items()]
        elif color_by == 'time':
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=color, markersize=10, label=category.title())
                             for category, color in self.time_colors.items()]
        else:  # generation
            max_gen = max(self.generation_map.values()) if self.generation_map else 0
            colors = plt.cm.Set3(np.linspace(0, 1, 12))
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=colors[i % len(colors)],
                                        markersize=10, label=f'Gen {i}')
                             for i in range(min(max_gen + 1, 6))]  # Limit legend size

        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

    def create_interactive_plot(self, color_by='generation'):
        """Create an interactive 3D plot using Plotly"""
        # Prepare data for all time points
        plot_data = []

        for event in self.division_events:
            color = self.get_cell_color(event, color_by)

            plot_data.append({
                'x': event['parent_x'],
                'y': event['parent_y'],
                'z': event['parent_z'],
                'cell_name': event['parent_cell'],
                'birth_time': event['birth_time'],
                'size': event['cell_size'],
                'generation': self.generation_map.get(event['parent_cell'], 0),
                'size_category': event['size_category'],
                'time_category': event['time_category'],
                'color': color
            })

        df_plot = pd.DataFrame(plot_data)

        # Create the plotly figure
        fig = go.Figure()

        # Add traces for different time points
        time_points = sorted(df_plot['birth_time'].unique())

        for i, time_point in enumerate(time_points):
            df_time = df_plot[df_plot['birth_time'] <= time_point]

            visible = True if i == len(time_points) - 1 else False

            fig.add_trace(go.Scatter3d(
                x=df_time['x'],
                y=df_time['y'],
                z=df_time['z'],
                mode='markers+text',
                text=df_time['cell_name'],
                textposition='top center',
                marker=dict(
                    size=8,
                    color=df_time['size'] if color_by == 'size' else df_time['generation'],
                    colorscale='Viridis' if color_by in ['size', 'generation'] else 'RdYlBu',
                    showscale=True,
                    colorbar=dict(title=color_by.title())
                ),
                name=f'Time {time_point}',
                visible=visible,
                hovertemplate=(
                    '<b>%{text}</b><br>' +
                    'X: %{x:.1f}<br>' +
                    'Y: %{y:.1f}<br>' +
                    'Z: %{z:.1f}<br>' +
                    f'Birth Time: {time_point}<br>' +
                    'Generation: ' + df_time['generation'].astype(str) + '<br>' +
                    '<extra></extra>'
                )
            ))

        # Create slider
        steps = []
        for i, time_point in enumerate(time_points):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(time_points)},
                      {"title": f"C. elegans Cell Division - Time {time_point}"}],
                label=f"t={time_point}"
            )
            step["args"][0]["visible"][i] = True
            steps.append(step)

        slider = dict(
            active=len(time_points) - 1,
            currentvalue={"prefix": "Time: "},
            pad={"t": 50},
            steps=steps
        )

        fig.update_layout(
            sliders=[slider],
            title=f'C. elegans Cell Division Animation - Colored by {color_by.title()}',
            scene=dict(
                xaxis_title='X Coordinate',
                yaxis_title='Y Coordinate',
                zaxis_title='Z Coordinate',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=900,
            height=700
        )

        return fig

    def create_matplotlib_animation(self, color_by='generation', interval=500, figsize=(12, 8)):
        """Create an animated matplotlib plot"""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Get time points
        time_points = sorted(set(e['birth_time'] for e in self.division_events))

        def animate(frame):
            ax.clear()
            current_time = time_points[frame] if frame < len(time_points) else time_points[-1]

            # Get active events
            active_events = [e for e in self.division_events if e['birth_time'] <= current_time]

            if active_events:
                x_coords = [e['parent_x'] for e in active_events]
                y_coords = [e['parent_y'] for e in active_events]
                z_coords = [e['parent_z'] for e in active_events]
                colors = [self.get_cell_color(e, color_by) for e in active_events]
                sizes = [100 + e['cell_size'] * 2 for e in active_events]

                scatter = ax.scatter(x_coords, y_coords, z_coords,
                                   c=colors, s=sizes, alpha=0.7,
                                   edgecolors='black', linewidth=0.5)

                # Add labels
                for event in active_events:
                    ax.text(event['parent_x'], event['parent_y'], event['parent_z'] + 2,
                           event['parent_cell'], fontsize=8)

            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_zlabel('Z Coordinate')
            ax.set_title(f'C. elegans Cell Division - Time {current_time:.1f}\nColored by {color_by.title()}')

            # Set consistent axis limits
            all_x = [e['parent_x'] for e in self.division_events]
            all_y = [e['parent_y'] for e in self.division_events]
            all_z = [e['parent_z'] for e in self.division_events]

            ax.set_xlim(min(all_x) - 10, max(all_x) + 10)
            ax.set_ylim(min(all_y) - 10, max(all_y) + 10)
            ax.set_zlim(min(all_z) - 2, max(all_z) + 2)

        anim = animation.FuncAnimation(fig, animate, frames=len(time_points),
                                     interval=interval, repeat=True, blit=False)

        return fig, anim

    def analyze_division_patterns(self):
        """Analyze and visualize division patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Division timing histogram
        birth_times = [e['birth_time'] for e in self.division_events]
        axes[0, 0].hist(birth_times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Birth Time')
        axes[0, 0].set_ylabel('Number of Divisions')
        axes[0, 0].set_title('Distribution of Cell Division Times')
        axes[0, 0].axvline(self.birth_time_threshold_low, color='red', linestyle='--',
                          label=f'Lower Time Threshold ({self.birth_time_threshold_low})')
        axes[0, 0].axvline(self.birth_time_threshold_high, color='red', linestyle='--',
                          label=f'Upper Time Threshold ({self.birth_time_threshold_high})')
        axes[0, 0].legend()

        # 2. Cell size distribution
        cell_sizes = [e['cell_size'] for e in self.division_events]
        axes[0, 1].hist(cell_sizes, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_xlabel('Cell Size (Distance from Origin)')
        axes[0, 1].set_ylabel('Number of Cells')
        axes[0, 1].set_title('Distribution of Cell Sizes')
        axes[0, 1].axvline(self.size_threshold_small, color='red', linestyle='--',
                          label=f'Lower Size Threshold ({self.size_threshold_small})')
        axes[0, 1].axvline(self.size_threshold_large, color='red', linestyle='--',
                          label=f'Upper Size Threshold ({self.size_threshold_large})')
        axes[0, 1].legend()

        # 3. Generation distribution
        generations = list(self.generation_map.values())
        gen_counts = pd.Series(generations).value_counts().sort_index()
        axes[1, 0].bar(gen_counts.index, gen_counts.values, alpha=0.7, color='coral', edgecolor='black')
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Number of Cells')
        axes[1, 0].set_title('Cells per Generation')

        # 4. Size vs Time scatter
        sizes = [e['cell_size'] for e in self.division_events]
        times = [e['birth_time'] for e in self.division_events]
        scatter = axes[1, 1].scatter(times, sizes, alpha=0.6, c=times, cmap='viridis')
        axes[1, 1].set_xlabel('Birth Time')
        axes[1, 1].set_ylabel('Cell Size')
        axes[1, 1].set_title('Cell Size vs Birth Time')
        plt.colorbar(scatter, ax=axes[1, 1], label='Birth Time')

        plt.tight_layout()
        return fig

    def export_data(self, filename='cell_division_analysis.csv'):
        """Export processed data with additional analysis"""
        export_data = []

        for event in self.division_events:
            export_data.append({
                'parent_cell': event['parent_cell'],
                'parent_x': event['parent_x'],
                'parent_y': event['parent_y'],
                'parent_z': event['parent_z'],
                'daughter1': event['daughter1'],
                'daughter2': event['daughter2'],
                'birth_time': event['birth_time'],
                'cell_size': event['cell_size'],
                'size_category': event['size_category'],
                'time_category': event['time_category'],
                'generation': self.generation_map.get(event['parent_cell'], 0)
            })

        df_export = pd.DataFrame(export_data)
        df_export.to_csv(filename, index=False)
        print(f"Data exported to {filename}")
        return df_export