#!/usr/bin/env python3
"""
Baseline Comparison Script for Cell Division Link Prediction

Compares three baseline models against the existing TGN on C. elegans lineage data:
  1. Random Baseline      — uniform random scores
  2. Degree Heuristic     — preferential-attachment scoring from training degrees
  3. Static GNN           — 2-layer GAT with no memory/temporal encoding

Usage:
    python scripts/run_baselines.py

Outputs:
    outputs/models/baseline_results.json
    outputs/paper_figures/baseline_comparison.png
"""

import sys
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score

from torch_geometric.nn import GATConv
from torch_geometric.loader import TemporalDataLoader

# Make the project root importable when running from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from devotg.data import build_cell_ctdg

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "cell_lineage_datasets" / "cells_birth_and_pos.csv"
PERF_SUMMARY = PROJECT_ROOT / "outputs" / "models" / "performance_summary.json"
OUTPUT_JSON = PROJECT_ROOT / "outputs" / "models" / "baseline_results.json"
OUTPUT_FIG = PROJECT_ROOT / "outputs" / "paper_figures" / "baseline_comparison.png"

# Hyper-parameters (kept identical to TGN for fair comparison)
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 200
EPOCHS = 20
LR = 0.001
HIDDEN_DIM = 100
EMBEDDING_DIM = 100
SEED = 42

# Publication colours
COLORS = {
    "random": "#888888",
    "degree_heuristic": "#ff7f0e",
    "static_gnn": "#2ca02c",
    "tgn": "#1f77b4",
}

LABELS = {
    "random": "Random",
    "degree_heuristic": "Degree Heuristic",
    "static_gnn": "Static GNN",
    "tgn": "TGN (ours)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_and_load(data, device):
    """Return (train_data, val_data, test_data) moved to device."""
    data = data.to(device)
    train_data, val_data, test_data = data.train_val_test_split(
        val_ratio=VAL_RATIO, test_ratio=TEST_RATIO
    )
    return train_data, val_data, test_data


def make_loaders(train_data, val_data, test_data):
    train_loader = TemporalDataLoader(
        train_data, batch_size=BATCH_SIZE, neg_sampling_ratio=1.0
    )
    val_loader = TemporalDataLoader(
        val_data, batch_size=BATCH_SIZE, neg_sampling_ratio=1.0
    )
    test_loader = TemporalDataLoader(
        test_data, batch_size=BATCH_SIZE, neg_sampling_ratio=1.0
    )
    return train_loader, val_loader, test_loader


def evaluate_scores(loader, score_fn):
    """
    Generic evaluator.

    score_fn(src, dst) -> 1-D numpy array of scores for positive pairs.
    score_fn(src, neg_dst) -> 1-D numpy array of scores for negative pairs.

    Returns (ap, auc).
    """
    all_y_pred, all_y_true = [], []
    for batch in loader:
        src = batch.src.cpu().numpy()
        dst = batch.dst.cpu().numpy()
        neg_dst = batch.neg_dst.cpu().numpy()

        pos_scores = score_fn(src, dst)
        neg_scores = score_fn(src, neg_dst)

        y_pred = np.concatenate([pos_scores, neg_scores])
        y_true = np.concatenate(
            [np.ones(len(pos_scores)), np.zeros(len(neg_scores))]
        )
        all_y_pred.append(y_pred)
        all_y_true.append(y_true)

    y_pred = np.concatenate(all_y_pred)
    y_true = np.concatenate(all_y_true)
    ap = average_precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    return float(ap), float(auc)


# ===========================================================================
# BASELINE 1 — Random
# ===========================================================================

def run_random_baseline(val_loader, test_loader):
    """Assign uniformly random scores to every edge."""
    print("\n" + "=" * 60)
    print("BASELINE 1: Random")
    print("=" * 60)

    rng = np.random.default_rng(SEED)

    def score_fn(src, dst):
        return rng.uniform(0.0, 1.0, size=len(src))

    val_ap, val_auc = evaluate_scores(val_loader, score_fn)
    test_ap, test_auc = evaluate_scores(test_loader, score_fn)

    print(f"  Val  AP={val_ap:.4f}  AUC={val_auc:.4f}")
    print(f"  Test AP={test_ap:.4f}  AUC={test_auc:.4f}")

    return {"val_ap": val_ap, "val_auc": val_auc,
            "test_ap": test_ap, "test_auc": test_auc}


# ===========================================================================
# BASELINE 2 — Degree Heuristic (Preferential Attachment)
# ===========================================================================

def run_degree_heuristic(train_data, val_loader, test_loader, num_nodes):
    """
    Count node degrees from training events only, then score
    (src, dst) as sqrt(deg[src] * deg[dst]) / max_degree.
    No ML involved — pure structural heuristic.
    """
    print("\n" + "=" * 60)
    print("BASELINE 2: Degree Heuristic (Preferential Attachment)")
    print("=" * 60)

    # Build degree counts from training edges only (no test leakage)
    degree = np.zeros(num_nodes, dtype=np.float64)
    src_tr = train_data.src.cpu().numpy()
    dst_tr = train_data.dst.cpu().numpy()
    for s, d in zip(src_tr, dst_tr):
        degree[s] += 1.0
        degree[d] += 1.0

    max_deg = degree.max() if degree.max() > 0 else 1.0
    print(f"  Training degrees: max={int(max_deg)}, "
          f"mean={degree.mean():.2f}, non-zero={int((degree > 0).sum())}")

    def score_fn(src, dst):
        scores = np.sqrt(degree[src] * degree[dst]) / max_deg
        return scores

    val_ap, val_auc = evaluate_scores(val_loader, score_fn)
    test_ap, test_auc = evaluate_scores(test_loader, score_fn)

    print(f"  Val  AP={val_ap:.4f}  AUC={val_auc:.4f}")
    print(f"  Test AP={test_ap:.4f}  AUC={test_auc:.4f}")

    return {"val_ap": val_ap, "val_auc": val_auc,
            "test_ap": test_ap, "test_auc": test_auc}


# ===========================================================================
# BASELINE 3 — Static GNN (2-layer GAT, ablation of temporal component)
# ===========================================================================

class StaticGAT(torch.nn.Module):
    """
    Two-layer Graph Attention Network that operates on a fixed static graph.
    Intentionally has no memory module and no temporal encoding — it is a
    direct ablation of the TGN's temporal component.

    Dimension accounting (with heads=2, hidden_dim=100, embedding_dim=100):
      conv1: input_dim  -> hidden_dim // heads per head, concat=True
             output dim = hidden_dim  (50*2 = 100)
      conv2: hidden_dim -> embedding_dim, heads=1, concat=False
             output dim = embedding_dim  (100)
    Using heads=1 for conv2 keeps the output dimension clean and equal to
    embedding_dim regardless of the number of attention heads in conv1.
    """

    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int,
                 heads: int = 2, dropout: float = 0.1):
        super().__init__()
        # Layer 1: input_dim -> hidden_dim  (multi-head concat)
        # Each head produces hidden_dim // heads features; concat gives hidden_dim total.
        self.conv1 = GATConv(
            input_dim, hidden_dim // heads, heads=heads,
            dropout=dropout, concat=True
        )
        # Layer 2: hidden_dim -> embedding_dim  (single head, no concat)
        # Single head avoids any dimension ambiguity: output is exactly embedding_dim.
        self.conv2 = GATConv(
            hidden_dim, embedding_dim, heads=1,
            dropout=dropout, concat=False
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class StaticLinkPredictor(torch.nn.Module):
    """
    Identical architecture to the TGN's LinkPredictor for fair comparison.
    Predicts edge probability from a pair of node embeddings.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)


def build_static_edge_index(train_data, num_nodes: int, device: torch.device):
    """
    Accumulate all training edges into a single static adjacency.
    We add both directions to make the graph undirected (standard for GAT
    on undirected biological networks).
    """
    src = train_data.src.cpu()
    dst = train_data.dst.cpu()
    # Both directions
    edge_index = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src])
    ], dim=0)
    # Remove duplicate edges
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index.to(device)


def train_static_gnn_epoch(gat, predictor, optimizer, criterion,
                            train_loader, x, edge_index, device):
    """
    One training epoch for the Static GNN.

    The GNN runs once per batch to recompute embeddings with gradient flow,
    then the LinkPredictor scores positive and negative edges.
    """
    gat.train()
    predictor.train()

    total_loss = 0.0
    num_events = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Recompute embeddings so gradients flow back through GNN weights
        z = gat(x, edge_index)  # [num_nodes, embedding_dim]

        pos_out = predictor(z[batch.src], z[batch.dst])
        neg_out = predictor(z[batch.src], z[batch.neg_dst])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.num_events
        num_events += batch.num_events

    return total_loss / num_events if num_events > 0 else 0.0


@torch.no_grad()
def evaluate_static_gnn(loader, gat, predictor, x_train, edge_index, device):
    """
    Evaluate Static GNN on val or test set.

    Embeddings are computed once over the static training graph and reused
    for all batches (no memory update, no temporal encoding).
    """
    gat.eval()
    predictor.eval()

    # Compute static embeddings once
    z = gat(x_train, edge_index)  # [num_nodes, embedding_dim]

    aps, aucs = [], []
    for batch in loader:
        batch = batch.to(device)

        pos_out = predictor(z[batch.src], z[batch.dst])
        neg_out = predictor(z[batch.src], z[batch.neg_dst])

        y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu().numpy()
        y_true = np.concatenate([
            np.ones(pos_out.size(0)), np.zeros(neg_out.size(0))
        ])

        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))

    return float(np.mean(aps)), float(np.mean(aucs))


def run_static_gnn(train_data, val_loader, test_loader,
                   num_nodes: int, input_dim: int, device: torch.device):
    """Full Static GNN training + evaluation pipeline."""
    print("\n" + "=" * 60)
    print("BASELINE 3: Static GNN (2-layer GAT, no temporal encoding)")
    print("=" * 60)

    set_seed(SEED)

    # Build static adjacency from all training edges
    edge_index = build_static_edge_index(train_data, num_nodes, device)
    print(f"  Static graph: {num_nodes} nodes, {edge_index.size(1)} edges "
          f"(after undirected dedup)")

    # Node features from the TemporalData object
    x = train_data.x.to(device)   # [num_nodes, input_dim]  (172-D)

    # Models
    gat = StaticGAT(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
    ).to(device)

    predictor = StaticLinkPredictor(in_channels=EMBEDDING_DIM).to(device)

    optimizer = torch.optim.Adam(
        list(gat.parameters()) + list(predictor.parameters()), lr=LR
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    num_params = (sum(p.numel() for p in gat.parameters()) +
                  sum(p.numel() for p in predictor.parameters()))
    print(f"  Static GNN parameters: {num_params:,}")

    best_val_auc = 0.0
    best_val_ap = 0.0
    best_test_ap = 0.0
    best_test_auc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Rebuild loader each epoch (TemporalDataLoader resamples negatives)
        train_loader_ep = TemporalDataLoader(
            train_data, batch_size=BATCH_SIZE, neg_sampling_ratio=1.0
        )

        epoch_loss = train_static_gnn_epoch(
            gat, predictor, optimizer, criterion,
            train_loader_ep, x, edge_index, device,
        )

        # Evaluation (no gradient)
        val_ap, val_auc = evaluate_static_gnn(
            val_loader, gat, predictor, x, edge_index, device
        )
        test_ap, test_auc = evaluate_static_gnn(
            test_loader, gat, predictor, x, edge_index, device
        )

        # Track best by val AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_ap = val_ap
            best_test_ap = test_ap
            best_test_auc = test_auc

        print(f"  Epoch {epoch:02d}  loss={epoch_loss:.4f}  "
              f"val_ap={val_ap:.4f}  val_auc={val_auc:.4f}  "
              f"test_ap={test_ap:.4f}  test_auc={test_auc:.4f}")

    print(f"\n  Best val_auc={best_val_auc:.4f}  (val_ap={best_val_ap:.4f})")
    print(f"  Corresponding test_ap={best_test_ap:.4f}  test_auc={best_test_auc:.4f}")

    return {
        "val_ap": best_val_ap,
        "val_auc": best_val_auc,
        "test_ap": best_test_ap,
        "test_auc": best_test_auc,
    }


# ===========================================================================
# Chart generation
# ===========================================================================

def make_comparison_chart(results: dict, output_path: Path):
    """
    Publication-quality grouped bar chart comparing AP and AUC across models.
    """
    model_keys = ["random", "degree_heuristic", "static_gnn", "tgn"]
    model_labels = [LABELS[k] for k in model_keys]
    colors = [COLORS[k] for k in model_keys]

    ap_vals = [results[k].get("test_ap", float("nan")) for k in model_keys]
    auc_vals = [results[k].get("test_auc", float("nan")) for k in model_keys]

    x = np.arange(len(model_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("white")

    bars_ap = ax.bar(x - width / 2, ap_vals, width,
                     color=colors, alpha=0.85, label="Average Precision (AP)",
                     edgecolor="black", linewidth=0.6)
    bars_auc = ax.bar(x + width / 2, auc_vals, width,
                      color=colors, alpha=0.55, label="ROC AUC",
                      edgecolor="black", linewidth=0.6, hatch="///")

    # Value annotations
    for bar in bars_ap:
        h = bar.get_height()
        if not math.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")

    for bar in bars_auc:
        h = bar.get_height()
        if not math.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Baseline Comparison — Cell Division Link Prediction\n"
                 r"(\textit{C. elegans} lineage data)", fontsize=13,
                 usetex=False)
    ax.set_ylim(0.0, 1.08)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8,
               alpha=0.4, label="Random chance (0.5)")
    ax.legend(fontsize=10, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComparison chart saved to: {output_path}")


# ===========================================================================
# Main
# ===========================================================================

def load_tgn_results() -> dict:
    """Load the best TGN test metrics from the stored performance summary."""
    if not PERF_SUMMARY.exists():
        print(f"  WARNING: {PERF_SUMMARY} not found — using hard-coded values.")
        return {"test_ap": 0.732, "test_auc": 0.825, "val_auc": 0.921,
                "val_ap": 0.878}

    with open(PERF_SUMMARY) as f:
        summary = json.load(f)

    metrics = summary.get("final_metrics", {})

    # The JSON stores the last epoch's metrics; also extract best val values
    history = summary.get("training_history", {})
    best_val_auc = max(history.get("val_auc", [0.0]))
    best_val_ap = max(history.get("val_ap", [0.0]))

    return {
        "test_ap": metrics.get("Final Test AP", 0.732),
        "test_auc": metrics.get("Final Test AUC", 0.825),
        "val_ap": best_val_ap,
        "val_auc": best_val_auc,
    }


def main():
    t0_total = time.time()
    print("=" * 60)
    print("DevoTG Baseline Comparison")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    # ------------------------------------------------------------------ data
    print(f"\nLoading data from: {CSV_PATH}")
    data = build_cell_ctdg(str(CSV_PATH))
    num_nodes = data.num_nodes
    input_dim = data.x.size(-1)   # 172
    print(f"  num_nodes={num_nodes}, num_events={data.num_events}, "
          f"input_dim={input_dim}")

    # Split (temporal, same ratios as TGN)
    train_data, val_data, test_data = split_and_load(data, device)
    print(f"  Split: train={train_data.num_events}, "
          f"val={val_data.num_events}, test={test_data.num_events}")

    # Loaders for heuristic baselines (val and test only)
    _, val_loader, test_loader = make_loaders(train_data, val_data, test_data)

    # ---------------------------------------------------------------- TGN reference
    print("\n" + "=" * 60)
    print("Loading existing TGN results")
    print("=" * 60)
    tgn_results = load_tgn_results()
    print(f"  TGN  Val AUC={tgn_results['val_auc']:.4f}  "
          f"Val AP={tgn_results['val_ap']:.4f}")
    print(f"  TGN  Test AUC={tgn_results['test_auc']:.4f}  "
          f"Test AP={tgn_results['test_ap']:.4f}")

    # ---------------------------------------------------------------- baselines
    t0 = time.time()
    random_results = run_random_baseline(val_loader, test_loader)
    print(f"  [Random done in {time.time()-t0:.1f}s]")

    t0 = time.time()
    degree_results = run_degree_heuristic(
        train_data, val_loader, test_loader, num_nodes
    )
    print(f"  [Degree heuristic done in {time.time()-t0:.1f}s]")

    t0 = time.time()
    static_gnn_results = run_static_gnn(
        train_data, val_loader, test_loader,
        num_nodes=num_nodes, input_dim=input_dim, device=device
    )
    print(f"  [Static GNN done in {time.time()-t0:.1f}s]")

    # ---------------------------------------------------------------- aggregate
    results = {
        "random": {
            "test_ap": random_results["test_ap"],
            "test_auc": random_results["test_auc"],
            "val_ap": random_results["val_ap"],
            "val_auc": random_results["val_auc"],
        },
        "degree_heuristic": {
            "test_ap": degree_results["test_ap"],
            "test_auc": degree_results["test_auc"],
            "val_ap": degree_results["val_ap"],
            "val_auc": degree_results["val_auc"],
        },
        "static_gnn": {
            "test_ap": static_gnn_results["test_ap"],
            "test_auc": static_gnn_results["test_auc"],
            "val_ap": static_gnn_results["val_ap"],
            "val_auc": static_gnn_results["val_auc"],
        },
        "tgn": {
            "test_ap": tgn_results["test_ap"],
            "test_auc": tgn_results["test_auc"],
            "val_ap": tgn_results["val_ap"],
            "val_auc": tgn_results["val_auc"],
        },
    }

    # ---------------------------------------------------------------- save JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBaseline results saved to: {OUTPUT_JSON}")

    # ---------------------------------------------------------------- chart
    make_comparison_chart(results, OUTPUT_FIG)

    # ---------------------------------------------------------------- summary
    total_time = time.time() - t0_total
    print("\n" + "=" * 60)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 60)
    header = f"{'Model':<22}  {'Test AP':>9}  {'Test AUC':>9}  {'Val AUC':>9}"
    print(header)
    print("-" * len(header))
    for key in ["random", "degree_heuristic", "static_gnn", "tgn"]:
        r = results[key]
        print(f"{LABELS[key]:<22}  "
              f"{r['test_ap']:>9.4f}  "
              f"{r['test_auc']:>9.4f}  "
              f"{r.get('val_auc', float('nan')):>9.4f}")
    print(f"\nTotal runtime: {total_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
