#!/usr/bin/env python3
"""Generate publication-quality figures for DevoTG NeurIPS ML4Bio workshop paper.

Usage:
    python scripts/generate_paper_figures.py

Outputs are written to outputs/paper_figures/ relative to the project root.
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3d projection
import networkx as nx

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Project paths                                                                 #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROCESSED_DIR = DATA_DIR / "processed_datasets"
LINEAGE_DIR = DATA_DIR / "cell_lineage_datasets"
STATS_DIR = OUTPUTS_DIR / "connectome_analysis" / "statistics"
MODELS_DIR = OUTPUTS_DIR / "models"

# --------------------------------------------------------------------------- #
# Global style                                                                  #
# --------------------------------------------------------------------------- #
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Nature-inspired palette
PALETTE = {
    "stable": "#2166AC",       # deep blue
    "developmental": "#F4A582",  # salmon/orange
    "variable": "#D6604D",     # brick red
    "chemical": "#4393C3",     # blue
    "electrical": "#F46D43",   # orange
    "ava": "#D73027",          # red  – AVAL/AVAR
    "avb": "#4575B4",          # blue – AVBL/AVBR
    "ave": "#1A9850",          # green – AVEL/AVER
    "neighbor": "#BBBBBB",
    "witvliet_stable": "#92C5DE",
    "witvliet_dynamic": "#FDDBC7",
    "witvliet_variable": "#F4A582",
}

PANEL_LABEL_KWARGS = dict(
    fontsize=13, fontweight="bold", transform=None,  # overridden per axis
    va="top", ha="left",
)


def _panel_label(ax, text):
    """Place bold panel label in upper-left corner of *ax*."""
    ax.text(
        -0.12, 1.05, text,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


# =========================================================================== #
# Figure 1 – Sample DevoTG: Time Evolution as Staggered Layers                 #
# =========================================================================== #
def generate_figure1_sample_devotg(output_dir: Path) -> None:
    """3-D staggered-layer visualisation of the temporal graph."""
    print("  Generating Figure 1 …")

    # --- Load data ---------------------------------------------------------- #
    nodes_path = PROCESSED_DIR / "dtdg_nodes.csv"
    edges_path = PROCESSED_DIR / "dtdg_edges_temporal.csv"
    stable_csv  = STATS_DIR / "stable_connections.csv"
    dev_csv     = STATS_DIR / "developmental_connections.csv"
    var_csv     = STATS_DIR / "variable_connections.csv"
    if not nodes_path.exists() or not edges_path.exists():
        print(f"    ERROR: data files missing ({nodes_path}, {edges_path})")
        return

    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    # --- Classify each neuron by its DOMINANT connection type --------------- #
    # Count how many stable / developmental / variable connections involve each node
    from collections import defaultdict
    cnt = defaultdict(lambda: {"s": 0, "d": 0, "v": 0})
    for csv_path, key in [(stable_csv, "s"), (dev_csv, "d"), (var_csv, "v")]:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for col in ["source_name", "target_name"]:
                for name in df[col]:
                    cnt[name][key] += 1

    def dominant_class(name):
        c = cnt[name]
        mx = max(c, key=c.get)
        return {"s": "stable", "d": "developmental", "v": "variable"}[mx]

    def node_color(name):
        cl = dominant_class(name)
        if cl == "stable":
            return PALETTE["stable"]
        if cl == "developmental":
            return PALETTE["developmental"]
        return "#AAAAAA"

    # --- Select ~5 nodes from each class (top degree within class) ---------- #
    deg_name = defaultdict(int)
    for _, row in edges_df.iterrows():
        deg_name[row["source_name"]] += 1
        deg_name[row["target_name"]] += 1

    deg_sorted = sorted(deg_name, key=lambda x: -deg_name[x])
    per_class = {"stable": [], "developmental": [], "variable": []}
    for name in deg_sorted:
        cl = dominant_class(name)
        if len(per_class[cl]) < 5:
            per_class[cl].append(name)
        if all(len(v) == 5 for v in per_class.values()):
            break

    selected_names = per_class["stable"] + per_class["developmental"] + per_class["variable"]

    # Timepoints to show: 1, 3, 5, 8
    tp_indices = [1, 3, 5, 8]
    tp_z      = {tp: z for z, tp in enumerate(tp_indices)}
    tp_labels = {1: "L1 (0h)", 3: "L1 (8h)", 5: "L2 (23h)", 8: "Adult (45h)"}

    # Subgraph edges
    sub_edges = edges_df[
        (edges_df["timepoint"].isin(tp_indices))
        & (edges_df["source_name"].isin(selected_names))
        & (edges_df["target_name"].isin(selected_names))
    ].copy()

    # 2D layout from accumulated graph
    G_all = nx.Graph()
    G_all.add_nodes_from(selected_names)
    for _, row in sub_edges.iterrows():
        G_all.add_edge(row["source_name"], row["target_name"])
    pos2d = nx.spring_layout(G_all, seed=42, k=1.8)

    # --- Plot --------------------------------------------------------------- #
    fig = plt.figure(figsize=(10, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    # Left margin gives room for z-axis tick labels
    fig.subplots_adjust(top=0.94, bottom=0.04, left=0.08, right=0.98)

    z_scale = 1.2

    for tp in tp_indices:
        z = tp_z[tp] * z_scale
        tp_sub = sub_edges[sub_edges["timepoint"] == tp]

        # Nodes — sized by degree in this timepoint
        for name in selected_names:
            x, y = pos2d[name]
            col = node_color(name)
            sz = 60 + deg_name[name] * 0.4
            ax.scatter(x, y, z, color=col, s=sz, zorder=5,
                       edgecolors="white", linewidths=0.5)

        # Within-layer edges
        drawn = set()
        for _, row in tp_sub.iterrows():
            key = (min(row["source_name"], row["target_name"]),
                   max(row["source_name"], row["target_name"]))
            if key in drawn:
                continue
            drawn.add(key)
            x0, y0 = pos2d[row["source_name"]]
            x1, y1 = pos2d[row["target_name"]]
            ax.plot([x0, x1], [y0, y1], [z, z],
                    color="#888888", lw=0.7, alpha=0.5)

    # Dashed vertical lines for new connections between layers
    prev_edges = None
    for i, tp in enumerate(tp_indices):
        cur_edges = set(zip(
            sub_edges[sub_edges["timepoint"] == tp]["source_name"],
            sub_edges[sub_edges["timepoint"] == tp]["target_name"],
        ))
        if prev_edges is not None:
            new_edges = cur_edges - prev_edges
            z_lo = tp_z[tp_indices[i - 1]] * z_scale
            z_hi = tp_z[tp] * z_scale
            for (src, tgt) in list(new_edges)[:20]:
                if src in pos2d and tgt in pos2d:
                    xm = (pos2d[src][0] + pos2d[tgt][0]) / 2
                    ym = (pos2d[src][1] + pos2d[tgt][1]) / 2
                    ax.plot([xm, xm], [ym, ym], [z_lo, z_hi],
                            color="#E08020", lw=0.9, linestyle="--", alpha=0.65)
        prev_edges = cur_edges

    ax.set_xlabel("X", fontsize=8, labelpad=2)
    ax.set_ylabel("Y", fontsize=8, labelpad=2)
    ax.set_zlabel("")   # tick labels are self-explanatory; no redundant label
    # Use built-in z-tick labels — matplotlib positions them correctly on the axis
    ax.set_zticks([tp_z[tp] * z_scale for tp in tp_indices])
    ax.set_zticklabels([tp_labels[tp] for tp in tp_indices],
                       fontsize=8.5, fontweight="bold", color="#222222")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.view_init(elev=22, azim=-55)

    # Title above the plot
    fig.suptitle("DevoTG: Temporal Evolution of Neural Connectivity",
                 fontsize=12, fontweight="bold", y=0.99)

    # Legend inside the figure at the bottom — avoids external whitespace
    legend_handles = [
        mpatches.Patch(color=PALETTE["stable"],       label="Stable neurons"),
        mpatches.Patch(color=PALETTE["developmental"], label="Developmental neurons"),
        mpatches.Patch(color="#AAAAAA",               label="Variable neurons"),
        Line2D([0], [0], color="#888888", lw=1.5,                label="Within-layer edge"),
        Line2D([0], [0], color="#E08020", lw=1.5, linestyle="--", label="New connection"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              ncol=3, framealpha=0.85, fontsize=8,
              bbox_to_anchor=(0.5, -0.12))

    out_path = output_dir / "fig1_sample_devotg.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"    Saved {out_path}")


# =========================================================================== #
# Figure 2 – Connection Stability Comparison                                    #
# =========================================================================== #
def generate_figure2_stability_comparison(output_dir: Path) -> None:
    """Grouped bar chart (DevoTG vs Witvliet) + edge-type line plot."""
    print("  Generating Figure 2 …")

    edges_path = PROCESSED_DIR / "dtdg_edges_temporal.csv"
    if not edges_path.exists():
        print(f"    ERROR: {edges_path} not found")
        return

    edges_df = pd.read_csv(edges_path)

    # --- Panel A: Stability proportions ------------------------------------- #
    # DevoTG counts (from prompt): stable=650, developmental=1207, variable=2440
    stable_count = 650
    dev_count = 1207
    var_count = 2440
    total_devotg = stable_count + dev_count + var_count
    devotg_pct = np.array([stable_count, dev_count, var_count]) / total_devotg * 100

    # Witvliet reference proportions
    wit_pct = np.array([43.0, 14.0, 43.0])

    categories = ["Stable", "Developmental\n/ Dynamic", "Variable"]
    x = np.arange(len(categories))
    width = 0.35

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

    bars_devotg = ax_a.bar(
        x - width / 2, devotg_pct, width,
        color=[PALETTE["stable"], PALETTE["developmental"], PALETTE["variable"]],
        alpha=0.9, label="DevoTG", edgecolor="white", linewidth=0.5,
    )
    bars_wit = ax_a.bar(
        x + width / 2, wit_pct, width,
        color=[PALETTE["witvliet_stable"], PALETTE["witvliet_dynamic"],
               PALETTE["witvliet_variable"]],
        alpha=0.85, label="Witvliet et al.", edgecolor="white", linewidth=0.5,
    )

    # Value labels
    for bar in bars_devotg:
        h = bar.get_height()
        ax_a.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                  f"{h:.1f}%", ha="center", va="bottom", fontsize=8)
    for bar in bars_wit:
        h = bar.get_height()
        ax_a.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                  f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(categories, fontsize=9)
    ax_a.set_ylabel("Proportion of connections (%)")
    ax_a.set_ylim(0, 70)
    ax_a.set_title("Connection Stability Classification")
    ax_a.legend(framealpha=0.8)
    ax_a.text(
        0.5, -0.18,
        "Witvliet: individual variability across animals\n"
        "DevoTG: temporal variability across development",
        transform=ax_a.transAxes,
        ha="center", fontsize=7.5, color="#555555",
        style="italic",
    )
    _panel_label(ax_a, "(A)")

    # --- Panel B: Edge counts by type over time ----------------------------- #
    tp_info = (
        edges_df
        .groupby(["timepoint", "time_hours", "stage", "type"])
        .size()
        .reset_index(name="count")
        .pivot_table(index=["timepoint", "time_hours", "stage"],
                     columns="type", values="count", fill_value=0)
        .reset_index()
    )
    tp_info.columns.name = None
    tp_info = tp_info.sort_values("timepoint").reset_index(drop=True)

    # Stage background shading
    stage_colors = {"L1": "#EAF4FB", "L2": "#FFF8E7", "L3": "#F0FFF0", "Adult": "#FFF0F0"}
    stage_groups = []
    for stage, grp in tp_info.groupby("stage", sort=False):
        idxs = grp.index.tolist()
        stage_groups.append((stage, min(idxs), max(idxs)))

    x_ticks = np.arange(len(tp_info))
    x_labels = [
        f"{row['stage']}\n{int(row['time_hours'])}h"
        for _, row in tp_info.iterrows()
    ]

    for stage, i_start, i_end in stage_groups:
        color = stage_colors.get(stage, "#F8F8F8")
        ax_b.axvspan(i_start - 0.4, i_end + 0.4, color=color, alpha=0.5, zorder=0)
        mid = (i_start + i_end) / 2
        ax_b.text(mid, ax_b.get_ylim()[1] if ax_b.get_ylim()[1] > 0 else 2500,
                  stage, ha="center", va="top", fontsize=7.5, color="#666666", zorder=5)

    if "chemical" in tp_info.columns:
        ax_b.plot(x_ticks, tp_info["chemical"], "o-",
                  color=PALETTE["chemical"], lw=2, ms=6, label="Chemical synapses", zorder=6)
    if "electrical" in tp_info.columns:
        ax_b.plot(x_ticks, tp_info["electrical"], "s-",
                  color=PALETTE["electrical"], lw=2, ms=6, label="Electrical (gap junctions)", zorder=6)

    ax_b.set_xticks(x_ticks)
    ax_b.set_xticklabels(x_labels, fontsize=8)
    ax_b.set_ylabel("Number of connections")
    ax_b.set_title("Synapse Type Growth Over Development")
    ax_b.legend(framealpha=0.8)
    ax_b.set_xlim(-0.5, len(tp_info) - 0.5)
    _panel_label(ax_b, "(B)")

    # Re-draw stage labels now y-limits are set
    ymax = ax_b.get_ylim()[1]
    ax_b.set_ylim(0, ymax * 1.08)
    for stage, i_start, i_end in stage_groups:
        mid = (i_start + i_end) / 2
        ax_b.text(mid, ymax * 1.04, stage,
                  ha="center", va="top", fontsize=7.5, color="#555555",
                  fontweight="bold", zorder=5)

    plt.tight_layout()
    out_path = output_dir / "fig2_stability_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {out_path}")


# =========================================================================== #
# Figure 3 – AVA / AVB / AVE Command Interneuron Circuit                        #
# =========================================================================== #
def generate_figure3_ava_avb_ave(output_dir: Path) -> None:
    """Ego-network of AVA/AVB/AVE command interneurons at birth and adult."""
    print("  Generating Figure 3 …")

    nodes_path = PROCESSED_DIR / "dtdg_nodes.csv"
    edges_path = PROCESSED_DIR / "dtdg_edges_temporal.csv"
    if not nodes_path.exists() or not edges_path.exists():
        print(f"    ERROR: data files missing")
        return

    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    CMD = ["AVAL", "AVAR", "AVBL", "AVBR", "AVEL", "AVER"]
    cmd_colors = {
        "AVAL": PALETTE["ava"], "AVAR": PALETTE["ava"],
        "AVBL": PALETTE["avb"], "AVBR": PALETTE["avb"],
        "AVEL": PALETTE["ave"], "AVER": PALETTE["ave"],
    }

    def build_ego_graph(tp):
        tp_edges = edges_df[edges_df["timepoint"] == tp]
        G = nx.MultiDiGraph()
        # Restrict to 1-hop neighbours of CMD neurons
        cmd_edges = tp_edges[
            (tp_edges["source_name"].isin(CMD)) | (tp_edges["target_name"].isin(CMD))
        ]
        for _, row in cmd_edges.iterrows():
            G.add_edge(row["source_name"], row["target_name"],
                       weight=row["weight"], etype=row["type"])
        return G

    def hexagonal_fixed_positions(cmd_nodes):
        """Place CMD neurons at vertices of a regular hexagon."""
        pos = {}
        n = len(cmd_nodes)
        for i, name in enumerate(cmd_nodes):
            angle = 2 * np.pi * i / n - np.pi / 2
            pos[name] = (np.cos(angle) * 0.4, np.sin(angle) * 0.4)
        return pos

    def draw_ego_panel(ax, G, tp_label):
        cmd_fixed = hexagonal_fixed_positions(CMD)
        other_nodes = [n for n in G.nodes() if n not in CMD]

        # Spring layout for neighbours, anchored around fixed CMD nodes
        if other_nodes:
            H = nx.Graph(G)
            for name, pos in cmd_fixed.items():
                if name not in H:
                    H.add_node(name)
            pos_all = nx.spring_layout(
                H, pos=cmd_fixed, fixed=list(cmd_fixed.keys()), seed=7, k=0.3
            )
        else:
            pos_all = dict(cmd_fixed)

        # Degree-proportional node sizes
        deg = dict(G.degree())
        max_deg = max(deg.values()) if deg else 1
        node_sizes = {n: 40 + 250 * (deg.get(n, 1) / max_deg) for n in G.nodes()}

        # Draw neighbour nodes
        nx.draw_networkx_nodes(
            G, pos_all, ax=ax,
            nodelist=other_nodes,
            node_size=[node_sizes[n] for n in other_nodes],
            node_color=PALETTE["neighbor"],
            alpha=0.7,
        )
        # Draw CMD nodes
        for name in CMD:
            if name in G.nodes():
                nx.draw_networkx_nodes(
                    G, pos_all, ax=ax,
                    nodelist=[name],
                    node_size=node_sizes[name],
                    node_color=cmd_colors[name],
                    alpha=0.95,
                )

        # Edges: split chemical vs electrical
        chem_edges = [(u, v) for u, v, d in G.edges(data=True) if d["etype"] == "chemical"]
        elec_edges = [(u, v) for u, v, d in G.edges(data=True) if d["etype"] == "electrical"]

        edge_weights = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
        max_w = max(edge_weights.values()) if edge_weights else 1

        def draw_edges(elist, style, color):
            if not elist:
                return
            widths = [0.4 + 2.5 * (edge_weights.get((u, v), 1) / max_w)
                      for u, v in elist]
            nx.draw_networkx_edges(
                G, pos_all, ax=ax,
                edgelist=elist,
                width=widths,
                style=style,
                edge_color=color,
                alpha=0.55,
                arrows=True,
                arrowsize=8,
                connectionstyle="arc3,rad=0.1",
            )

        draw_edges(chem_edges, "solid", PALETTE["chemical"])
        draw_edges(elec_edges, "dashed", PALETTE["electrical"])

        # Labels for CMD nodes only
        cmd_in_graph = {n: n for n in CMD if n in G.nodes()}
        nx.draw_networkx_labels(G, pos_all, labels=cmd_in_graph, ax=ax,
                                font_size=6.5, font_weight="bold")

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        ax.set_title(f"{tp_label}\n({n_nodes} neurons, {n_edges} connections)",
                     fontsize=10)
        ax.axis("off")

    # Build graphs
    G_birth = build_ego_graph(tp=1)
    G_adult = build_ego_graph(tp=8)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 6))
    draw_ego_panel(ax_a, G_birth, "Birth (t = 0h, L1)")
    draw_ego_panel(ax_b, G_adult, "Adult (t = 45h)")

    _panel_label(ax_a, "(A)")
    _panel_label(ax_b, "(B)")

    # Global legend
    legend_handles = [
        mpatches.Patch(color=PALETTE["ava"], label="AVA (AVAL / AVAR)"),
        mpatches.Patch(color=PALETTE["avb"], label="AVB (AVBL / AVBR)"),
        mpatches.Patch(color=PALETTE["ave"], label="AVE (AVEL / AVER)"),
        mpatches.Patch(color=PALETTE["neighbor"], label="Direct neighbour"),
        Line2D([0], [0], color=PALETTE["chemical"], lw=1.5, label="Chemical synapse"),
        Line2D([0], [0], color=PALETTE["electrical"], lw=1.5, linestyle="--",
               label="Electrical (gap junction)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               framealpha=0.9, fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("AVA, AVB, AVE Command Interneurons Across Development",
                 fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    out_path = output_dir / "fig3_ava_avb_ave_circuit.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {out_path}")


# =========================================================================== #
# Figure 4 – Spatiotemporal Development Graph                                   #
# =========================================================================== #
def generate_figure4_spatiotemporal(output_dir: Path) -> None:
    """Lineage tree (first 5 generations) + spatial scatter of cell births."""
    print("  Generating Figure 4 …")

    lineage_path = LINEAGE_DIR / "cells_birth_and_pos.csv"
    if not lineage_path.exists():
        print(f"    ERROR: {lineage_path} not found")
        return

    lin = pd.read_csv(lineage_path)

    # Build parent→children dict and all-cell info
    parent_to_children = {}
    birth_time = {}
    pos_xy = {}
    for _, row in lin.iterrows():
        parent = row["Parent Cell"]
        d1, d2 = row["Daughter 1"], row["Daughter 2"]
        parent_to_children[parent] = [str(d1), str(d2)]
        birth_time[parent] = row["Birth Time"]
        pos_xy[parent] = (row["parent_x"], row["parent_y"])

    # Count descendants (for node sizing)
    all_cells_in_lineage = set(parent_to_children.keys())
    for children in parent_to_children.values():
        all_cells_in_lineage.update(children)

    def count_descendants(cell, cache={}):
        if cell in cache:
            return cache[cell]
        if cell not in parent_to_children:
            cache[cell] = 0
            return 0
        total = sum(1 + count_descendants(c) for c in parent_to_children[cell])
        cache[cell] = total
        return total

    # ---- Panel A: Lineage tree, first 5 generations ----------------------- #
    MAX_GEN = 5

    # BFS to collect nodes up to MAX_GEN
    gen_nodes = {0: ["P0"]}
    for g in range(1, MAX_GEN + 1):
        cur = []
        for p in gen_nodes[g - 1]:
            if p in parent_to_children:
                cur.extend(parent_to_children[p])
        gen_nodes[g] = cur

    tree_nodes = [n for nodes in gen_nodes.values() for n in nodes]
    tree_edges = []
    for g in range(MAX_GEN):
        for p in gen_nodes[g]:
            if p in parent_to_children:
                for ch in parent_to_children[p]:
                    if ch in set(tree_nodes):
                        tree_edges.append((p, ch))

    # Build a DiGraph and compute hierarchical (Reingold-Tilford-like) layout
    T = nx.DiGraph()
    T.add_nodes_from(tree_nodes)
    T.add_edges_from(tree_edges)

    # Assign x positions by generation, y = -generation (top-down)
    def assign_x_positions(root, parent_to_children, max_gen):
        """Assign fractional x positions using DFS leaf-ordering."""
        leaf_counter = [0]

        def dfs(node, gen):
            if gen >= max_gen or node not in parent_to_children:
                x = leaf_counter[0]
                leaf_counter[0] += 1
                return {node: x}
            positions = {}
            child_positions = {}
            for ch in parent_to_children[node]:
                cp = dfs(ch, gen + 1)
                child_positions.update(cp)
                positions.update(cp)
            child_xs = [child_positions[ch] for ch in parent_to_children[node]
                        if ch in child_positions]
            positions[node] = np.mean(child_xs) if child_xs else leaf_counter[0]
            return positions

        return dfs(root, 0)

    x_pos = assign_x_positions("P0", parent_to_children, MAX_GEN)
    gen_of = {}
    for g, nodes in gen_nodes.items():
        for n in nodes:
            gen_of[n] = g

    tree_pos = {}
    for n in tree_nodes:
        g = gen_of.get(n, 0)
        x = x_pos.get(n, 0)
        tree_pos[n] = (x, -g)

    # Birth times for colour
    node_bt = [birth_time.get(n, 0) for n in tree_nodes]
    bt_max = max(node_bt) if node_bt else 1
    cmap = plt.get_cmap("viridis")
    node_colors = [cmap(bt / bt_max) for bt in node_bt]

    # Descendant count for node size
    desc_counts = [count_descendants(n) for n in tree_nodes]
    max_desc = max(desc_counts) if desc_counts else 1
    node_sizes = [30 + 300 * (d / max_desc) ** 0.5 for d in desc_counts]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 6))

    # Draw edges
    for u, v in tree_edges:
        xu, yu = tree_pos[u]
        xv, yv = tree_pos[v]
        ax_a.plot([xu, xv], [yu, yv], color="#CCCCCC", lw=0.9, zorder=1)

    # Draw nodes
    scatter_a = ax_a.scatter(
        [tree_pos[n][0] for n in tree_nodes],
        [tree_pos[n][1] for n in tree_nodes],
        c=node_bt, cmap="viridis", vmin=0, vmax=bt_max,
        s=node_sizes, zorder=3, edgecolors="white", linewidths=0.4,
    )

    # Label first 3 generations
    for n in tree_nodes:
        if gen_of.get(n, 99) <= 3:
            x, y = tree_pos[n]
            ax_a.text(x, y + 0.12, n, ha="center", va="bottom",
                      fontsize=6.5, fontweight="bold" if gen_of[n] <= 1 else "normal")

    cb_a = plt.colorbar(scatter_a, ax=ax_a, pad=0.02, fraction=0.03)
    cb_a.set_label("Birth time (min)", fontsize=8)
    cb_a.ax.tick_params(labelsize=7)

    ax_a.set_yticks(range(0, -(MAX_GEN + 1), -1))
    ax_a.set_yticklabels([f"Gen {g}" for g in range(MAX_GEN + 1)], fontsize=8)
    ax_a.set_xticks([])
    ax_a.spines["bottom"].set_visible(False)
    ax_a.spines["left"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.set_title("Cell Lineage Tree (first 5 generations)", fontsize=10)
    _panel_label(ax_a, "(A)")

    # ---- Panel B: Spatial distribution ------------------------------------ #
    # Use all cells in the lineage CSV (one row per division = one parent)
    spatial_bt = lin["Birth Time"].values
    max_sbt = spatial_bt.max()
    spatial_desc = np.array([count_descendants(p) for p in lin["Parent Cell"]])
    max_sdesc = max(spatial_desc) if spatial_desc.max() > 0 else 1
    spatial_sizes = 15 + 200 * (spatial_desc / max_sdesc) ** 0.5

    scatter_b = ax_b.scatter(
        lin["parent_x"], lin["parent_y"],
        c=spatial_bt, cmap="viridis", vmin=0, vmax=max_sbt,
        s=spatial_sizes, alpha=0.75, edgecolors="none",
    )

    cb_b = plt.colorbar(scatter_b, ax=ax_b, pad=0.02, fraction=0.03)
    cb_b.set_label("Birth time (min)", fontsize=8)
    cb_b.ax.tick_params(labelsize=7)

    ax_b.set_xlabel("X position (µm)", fontsize=9)
    ax_b.set_ylabel("Y position (µm)", fontsize=9)
    ax_b.set_title("Spatial Distribution of Cell Births", fontsize=10)
    _panel_label(ax_b, "(B)")

    fig.suptitle(
        "Spatiotemporal C. elegans Development: Lineage & Space",
        fontsize=13, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    out_path = output_dir / "fig4_spatiotemporal.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {out_path}")


# =========================================================================== #
# Figure 5 – TGN Training Performance                                           #
# =========================================================================== #
def generate_figure5_tgn_performance(output_dir: Path) -> None:
    """Training curves + baseline comparison bar chart."""
    print("  Generating Figure 5 …")

    perf_path = MODELS_DIR / "performance_summary.json"
    if not perf_path.exists():
        print(f"    ERROR: {perf_path} not found")
        return

    with open(perf_path) as f:
        perf = json.load(f)

    history = perf.get("training_history", {})
    train_loss = history.get("train_loss", [])
    val_auc = history.get("val_auc", [])
    test_auc = history.get("test_auc", [])
    best_epoch = perf.get("best_epoch", 16)
    final_metrics = perf.get("final_metrics", {})
    final_test_auc = final_metrics.get("Final Test AUC", 0.825)

    epochs = np.arange(1, len(train_loss) + 1)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Panel A: Training curves ------------------------------------------ #
    color_loss = "#888888"
    color_val = PALETTE["chemical"]
    color_test = PALETTE["electrical"]

    ax_a2 = ax_a.twinx()

    line_loss, = ax_a.plot(epochs, train_loss, "--", color=color_loss,
                           lw=1.5, label="Train loss (left)")
    line_val, = ax_a2.plot(epochs, val_auc, "-o", color=color_val,
                            lw=2, ms=4, label="Val AUC (right)")
    line_test, = ax_a2.plot(epochs, test_auc, "-s", color=color_test,
                             lw=2, ms=4, label="Test AUC (right)")

    # Mark best epoch
    ax_a2.axvline(x=best_epoch, color="#555555", linestyle=":", lw=1.5,
                  label=f"Best epoch ({best_epoch})")

    # Annotate final test AUC
    ax_a2.annotate(
        f"Test AUC = {final_test_auc:.3f}",
        xy=(len(test_auc), test_auc[-1]),
        xytext=(len(test_auc) - 6, test_auc[-1] - 0.08),
        fontsize=8,
        color=color_test,
        arrowprops=dict(arrowstyle="->", color=color_test, lw=1.0),
    )

    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel("Training Loss", color=color_loss)
    ax_a.tick_params(axis="y", labelcolor=color_loss)
    ax_a2.set_ylabel("AUC", color=color_val)
    ax_a2.tick_params(axis="y", labelcolor=color_val)
    ax_a2.set_ylim(0.5, 1.0)
    ax_a2.spines["right"].set_visible(True)

    # Combined legend
    lines = [line_loss, line_val, line_test]
    labels = [l.get_label() for l in lines]
    ax_a.legend(lines, labels, framealpha=0.85, fontsize=8, loc="upper right")
    ax_a.set_title("TGN Training Curves")
    _panel_label(ax_a, "(A)")

    # --- Panel B: Baseline comparison bar chart ----------------------------- #
    # Load real baseline results if available
    baseline_file = output_dir.parent.parent / "models" / "baseline_results.json"
    if baseline_file.exists():
        with open(baseline_file) as f:
            bl = json.load(f)
        baselines = {
            "Random\n(chance)": bl["random"]["test_auc"],
            "Degree\nheuristic": bl["degree_heuristic"]["test_auc"],
            "Static\nGNN": bl["static_gnn"]["test_auc"],
            "DevoTG\n(TGN)": bl["tgn"]["test_auc"],
        }
        footnote = ""
    else:
        baselines = {
            "Random\n(chance)": 0.478,
            "Degree\nheuristic": 0.538,
            "Static\nGNN": 0.535,
            "DevoTG\n(TGN)": 0.825,
        }
        footnote = "* Run scripts/run_baselines.py to refresh"

    bar_colors = ["#BBBBBB", "#ff7f0e", "#2ca02c", PALETTE["stable"]]
    tgn_auc = baselines["DevoTG\n(TGN)"]

    x_pos = np.arange(len(baselines))
    bars = ax_b.bar(x_pos, list(baselines.values()), color=bar_colors,
                    edgecolor="white", linewidth=0.5, width=0.5)

    # Value labels on bars
    for bar, val in zip(bars, baselines.values()):
        ax_b.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.008,
                  f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                  fontweight="bold" if val == tgn_auc else "normal")

    # Highlight DevoTG bar
    bars[-1].set_edgecolor("#003366")
    bars[-1].set_linewidth(1.5)

    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(list(baselines.keys()), fontsize=8.5)
    ax_b.set_ylabel("Test AUC")
    ax_b.set_ylim(0.4, 0.93)
    ax_b.axhline(y=0.5, color="#AAAAAA", linestyle="--", lw=1, label="Chance level")
    ax_b.set_title("Baseline Comparison (Test AUC)")
    if footnote:
        ax_b.text(0.5, 0.02, footnote, transform=ax_b.transAxes,
                  ha="center", fontsize=7, color="#888888", style="italic")
    _panel_label(ax_b, "(B)")

    fig.suptitle("TGN Training Performance on C. elegans Cell Lineage",
                 fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    out_path = output_dir / "fig5_tgn_performance.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {out_path}")


# =========================================================================== #
# Main                                                                          #
# =========================================================================== #
if __name__ == "__main__":
    output_dir = PROJECT_ROOT / "outputs" / "paper_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_fns = [
        generate_figure1_sample_devotg,
        generate_figure2_stability_comparison,
        generate_figure3_ava_avb_ave,
        generate_figure4_spatiotemporal,
        generate_figure5_tgn_performance,
    ]

    success = 0
    for fn in figure_fns:
        try:
            fn(output_dir)
            success += 1
        except Exception as exc:
            print(f"    FAILED [{fn.__name__}]: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\nAll figures generated ({success}/{len(figure_fns)}) in {output_dir}")
