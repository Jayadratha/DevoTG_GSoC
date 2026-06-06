#!/usr/bin/env python3
"""
Multi-Seed Evaluation Script for DevoTG Paper

Runs TGN and all baselines across N random seeds, reports mean ± std.
For each run, test metrics are taken at the BEST VALIDATION epoch
(not final epoch) — standard ML practice.

Usage:
    python scripts/run_multi_seed_eval.py [--n_seeds 5] [--epochs 20]

Outputs:
    outputs/models/multi_seed_results.json   — per-seed and summary stats
    outputs/paper_figures/baseline_comparison_multiseed.png
"""

import os
import sys
import json
import time
import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score

from torch_geometric.nn import GATConv, TransformerConv
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, LastNeighborLoader
from torch_geometric.nn import TGNMemory
from torch_geometric.loader import TemporalDataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from devotg.data import build_cell_ctdg

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
CSV_PATH     = PROJECT_ROOT / "data" / "cell_lineage_datasets" / "cells_birth_and_pos.csv"
OUT_JSON     = PROJECT_ROOT / "outputs" / "models" / "multi_seed_results.json"
OUT_FIG      = PROJECT_ROOT / "outputs" / "paper_figures" / "baseline_comparison_multiseed.png"

VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
BATCH_SIZE   = 200
EPOCHS       = 20
LR           = 0.001
EMBED_DIM    = 100
MEM_DIM      = 100
TIME_DIM     = 100
NEIGHBORS    = 10

PALETTE = dict(random="#BBBBBB", degree="#ff7f0e", static_gnn="#2ca02c", tgn="#1f77b4")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Fully deterministic seeding across Python, NumPy, and PyTorch (CPU + GPU)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Force deterministic algorithms where available (PyTorch >= 1.8)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_data(data, seed: int):
    """Temporal split — deterministic on split boundaries, seed affects neg sampling."""
    return data.train_val_test_split(val_ratio=VAL_RATIO, test_ratio=TEST_RATIO)


def evaluate_scores(loader, score_fn, device):
    """Generic evaluator: score_fn(src, dst, neg_dst, batch) -> (pos_scores, neg_scores)."""
    aps, aucs = [], []
    for batch in loader:
        batch = batch.to(device)
        pos_scores, neg_scores = score_fn(batch)
        y_pred = torch.cat([pos_scores, neg_scores]).sigmoid().cpu().numpy()
        y_true = np.concatenate([
            np.ones(len(pos_scores)), np.zeros(len(neg_scores))
        ])
        if len(np.unique(y_true)) < 2:
            continue
        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))
    ap  = float(np.mean(aps))  if aps  else 0.0
    auc = float(np.mean(aucs)) if aucs else 0.0
    return ap, auc


# ---------------------------------------------------------------------------
# Baseline 1 — Random
# ---------------------------------------------------------------------------

def run_random(train_loader, val_loader, test_loader, device):
    rng = np.random.default_rng(0)

    def score(batch):
        n = batch.src.shape[0]
        pos = torch.tensor(rng.uniform(size=n), dtype=torch.float32)
        neg = torch.tensor(rng.uniform(size=n), dtype=torch.float32)
        return pos, neg

    val_ap,  val_auc  = evaluate_scores(val_loader,  score, device)
    test_ap, test_auc = evaluate_scores(test_loader, score, device)
    return dict(val_ap=val_ap, val_auc=val_auc, test_ap=test_ap, test_auc=test_auc)


# ---------------------------------------------------------------------------
# Baseline 2 — Degree Heuristic
# ---------------------------------------------------------------------------

def run_degree_heuristic(train_loader, val_loader, test_loader, data, device):
    deg = torch.zeros(data.num_nodes, dtype=torch.float32)
    for batch in train_loader:
        deg[batch.src.cpu()] += 1
        deg[batch.dst.cpu()] += 1
    max_deg = deg.max().item() or 1.0

    def score(batch):
        s, d, nd = batch.src.cpu(), batch.dst.cpu(), batch.neg_dst.cpu()
        pos = (deg[s] * deg[d]).sqrt() / max_deg
        neg = (deg[s] * deg[nd]).sqrt() / max_deg
        return pos, neg

    val_ap,  val_auc  = evaluate_scores(val_loader,  score, device)
    test_ap, test_auc = evaluate_scores(test_loader, score, device)
    return dict(val_ap=val_ap, val_auc=val_auc, test_ap=test_ap, test_auc=test_auc)


# ---------------------------------------------------------------------------
# Baseline 3 — Static GNN (GAT, no memory)
# ---------------------------------------------------------------------------

class StaticLinkPredictor(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin_src   = Linear(dim, dim)
        self.lin_dst   = Linear(dim, dim)
        self.lin_final = Linear(dim, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        return self.lin_final(h.relu())


class StaticGNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=100):
        super().__init__()
        self.conv1 = GATConv(in_dim,      hidden_dim // 2, heads=2, dropout=0.1, concat=True)
        self.conv2 = GATConv(hidden_dim,  hidden_dim,      heads=1, dropout=0.1, concat=False)
        self.pred  = StaticLinkPredictor(hidden_dim)

    def embed(self, x, edge_index):
        z = F.elu(self.conv1(x, edge_index))
        z = F.elu(self.conv2(z, edge_index))
        return z

    def forward(self, x, edge_index, src, dst):
        z = self.embed(x, edge_index)
        return self.pred(z[src], z[dst])


def run_static_gnn(train_loader, val_loader, test_loader, data, device, seed):
    set_seed(seed)
    x = data.x.to(device)
    feat_dim = x.shape[1]

    # Build static training adjacency
    train_src, train_dst = [], []
    for b in train_loader:
        train_src.append(b.src); train_dst.append(b.dst)
    t_src = torch.cat(train_src)
    t_dst = torch.cat(train_dst)
    edge_index = torch.stack([
        torch.cat([t_src, t_dst]),
        torch.cat([t_dst, t_src])   # undirected
    ]).to(device)

    model = StaticGNN(feat_dim, EMBED_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_metrics = dict(val_ap=0.0, val_auc=0.0, test_ap=0.0, test_auc=0.0)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, n = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            pos_out = model(x, edge_index, batch.src, batch.dst)
            neg_out = model(x, edge_index, batch.src, batch.neg_dst)
            loss = criterion(pos_out, torch.ones_like(pos_out))
            loss += criterion(neg_out, torch.zeros_like(neg_out))
            loss.backward(); optimizer.step()
            total_loss += loss.item() * batch.num_events
            n += batch.num_events

        model.eval()
        with torch.no_grad():
            def score(batch):
                b = batch.to(device)
                return (model(x, edge_index, b.src, b.dst),
                        model(x, edge_index, b.src, b.neg_dst))

            val_ap,  val_auc  = evaluate_scores(val_loader,  score, device)
            test_ap, test_auc = evaluate_scores(test_loader, score, device)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_metrics = dict(val_ap=val_ap, val_auc=val_auc,
                                test_ap=test_ap, test_auc=test_auc,
                                best_epoch=epoch)

    return best_metrics


# ---------------------------------------------------------------------------
# TGN
# ---------------------------------------------------------------------------

class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2,
                                    heads=2, dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t     = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src   = Linear(in_channels, in_channels)
        self.lin_dst   = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        return self.lin_final(h.relu())


def run_tgn(train_loader, val_loader, test_loader, data, device, seed):
    set_seed(seed)
    num_nodes = data.num_nodes
    msg_dim   = data.msg.size(-1)

    memory = TGNMemory(
        num_nodes, msg_dim, MEM_DIM, TIME_DIM,
        message_module=IdentityMessage(msg_dim, MEM_DIM, TIME_DIM),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=MEM_DIM, out_channels=EMBED_DIM,
        msg_dim=msg_dim, time_enc=memory.time_enc,
    ).to(device)

    link_pred = LinkPredictor(in_channels=EMBED_DIM).to(device)

    optimizer = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters()) | set(link_pred.parameters()),
        lr=LR)
    criterion = torch.nn.BCEWithLogitsLoss()
    assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
    neighbor_loader = LastNeighborLoader(num_nodes, size=NEIGHBORS, device=device)

    # Keep data on CPU; move only the lookup tensors (t, msg) to device once
    data_t   = data.t.to(device)
    data_msg = data.msg.to(device)

    def train_epoch():
        memory.train(); gnn.train(); link_pred.train()
        memory.reset_state(); neighbor_loader.reset_state()
        total_loss, n = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            b = batch.to(device)
            n_id, edge_index, e_id = neighbor_loader(b.n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)
            z, last_update = memory(n_id)
            z = gnn(z, last_update, edge_index,
                    data_t[e_id], data_msg[e_id])
            pos_out = link_pred(z[assoc[b.src]], z[assoc[b.dst]])
            neg_out = link_pred(z[assoc[b.src]], z[assoc[b.neg_dst]])
            loss = criterion(pos_out, torch.ones_like(pos_out))
            loss += criterion(neg_out, torch.zeros_like(neg_out))
            memory.update_state(b.src, b.dst, b.t, b.msg)
            neighbor_loader.insert(b.src, b.dst)
            loss.backward(); optimizer.step(); memory.detach()
            total_loss += float(loss) * b.num_events
            n += b.num_events
        return total_loss / n if n else 0.0

    @torch.no_grad()
    def eval_loader(loader):
        memory.eval(); gnn.eval(); link_pred.eval()
        torch.manual_seed(12345)
        aps, aucs = [], []
        for batch in loader:
            b = batch.to(device)
            n_id, edge_index, e_id = neighbor_loader(b.n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)
            z, last_update = memory(n_id)
            z = gnn(z, last_update, edge_index,
                    data_t[e_id], data_msg[e_id])
            pos_out = link_pred(z[assoc[b.src]], z[assoc[b.dst]])
            neg_out = link_pred(z[assoc[b.src]], z[assoc[b.neg_dst]])
            y_pred = torch.cat([pos_out, neg_out]).sigmoid().cpu()
            y_true = torch.cat([torch.ones(pos_out.size(0)),
                                torch.zeros(neg_out.size(0))])
            aps.append(average_precision_score(y_true.numpy(), y_pred.numpy()))
            aucs.append(roc_auc_score(y_true.numpy(), y_pred.numpy()))
            memory.update_state(b.src, b.dst, b.t, b.msg)
            neighbor_loader.insert(b.src, b.dst)
        return float(np.mean(aps)), float(np.mean(aucs))

    best_val_auc = -1.0
    best_metrics = dict(val_ap=0.0, val_auc=0.0, test_ap=0.0, test_auc=0.0)

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch()
        val_ap,  val_auc  = eval_loader(val_loader)
        test_ap, test_auc = eval_loader(test_loader)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_metrics = dict(val_ap=val_ap, val_auc=val_auc,
                                test_ap=test_ap, test_auc=test_auc,
                                best_epoch=epoch)
        print(f"  [TGN] Epoch {epoch:02d}  loss={loss:.4f}"
              f"  val_auc={val_auc:.4f}  test_auc={test_auc:.4f}"
              f"  (best val_auc={best_val_auc:.4f})")

    return best_metrics


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------

def run_all_seeds(n_seeds: int):
    device = get_device()
    print(f"\nDevice: {device}")
    print(f"Loading data from {CSV_PATH} ...")
    data = build_cell_ctdg(str(CSV_PATH), feature_dim=172)
    print(f"  {data.num_nodes} nodes, {data.num_events} events\n")

    seeds = list(range(n_seeds))
    results = {m: [] for m in ("random", "degree_heuristic", "static_gnn", "tgn")}

    for seed in seeds:
        print(f"{'='*60}")
        print(f"SEED {seed}  ({seed+1}/{n_seeds})")
        print(f"{'='*60}")
        # Full seed reset before EVERY run — covers weight init AND neg sampling
        set_seed(seed)

        train_data, val_data, test_data = split_data(data, seed)
        # Recreate loaders inside the seed loop so their internal RNG state is seeded
        train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE, neg_sampling_ratio=1.0)
        val_loader   = TemporalDataLoader(val_data,   batch_size=BATCH_SIZE, neg_sampling_ratio=1.0)
        test_loader  = TemporalDataLoader(test_data,  batch_size=BATCH_SIZE, neg_sampling_ratio=1.0)

        # --- Random ---
        print("[Random] ...")
        m = run_random(train_loader, val_loader, test_loader, device)
        results["random"].append(m)
        print(f"  test_auc={m['test_auc']:.4f}  test_ap={m['test_ap']:.4f}")

        # --- Degree heuristic ---
        print("[Degree heuristic] ...")
        m = run_degree_heuristic(train_loader, val_loader, test_loader, data, device)
        results["degree_heuristic"].append(m)
        print(f"  test_auc={m['test_auc']:.4f}  test_ap={m['test_ap']:.4f}")

        # --- Static GNN ---
        print("[Static GNN] ...")
        t0 = time.time()
        m = run_static_gnn(train_loader, val_loader, test_loader, data, device, seed)
        results["static_gnn"].append(m)
        print(f"  test_auc={m['test_auc']:.4f}  test_ap={m['test_ap']:.4f}"
              f"  (best epoch {m.get('best_epoch','?')}, {time.time()-t0:.1f}s)")

        # --- TGN ---
        print("[TGN] ...")
        t0 = time.time()
        m = run_tgn(train_loader, val_loader, test_loader, data, device, seed)
        results["tgn"].append(m)
        print(f"  test_auc={m['test_auc']:.4f}  test_ap={m['test_ap']:.4f}"
              f"  (best epoch {m.get('best_epoch','?')}, {time.time()-t0:.1f}s)")

    return results


def summarise(results: dict) -> dict:
    """Compute mean ± std for each model over all seeds."""
    summary = {}
    for model, runs in results.items():
        summary[model] = {}
        for metric in ("test_auc", "test_ap", "val_auc", "val_ap"):
            vals = [r[metric] for r in runs if metric in r]
            summary[model][f"{metric}_mean"] = float(np.mean(vals))
            summary[model][f"{metric}_std"]  = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            summary[model][f"{metric}_all"]  = [float(v) for v in vals]
    return summary


def print_table(summary: dict):
    print(f"\n{'='*70}")
    print("MULTI-SEED RESULTS  (test metrics at best-validation epoch)")
    print(f"{'='*70}")
    header = f"{'Model':<22}  {'Test AUC':>18}  {'Test AP':>18}  {'Val AUC':>18}"
    print(header)
    print("-" * 78)
    labels = [("random", "Random"),
              ("degree_heuristic", "Degree Heuristic"),
              ("static_gnn", "Static GNN"),
              ("tgn", "DevoTG TGN (ours)")]
    for key, label in labels:
        s = summary[key]
        auc  = f"{s['test_auc_mean']:.3f} ± {s['test_auc_std']:.3f}"
        ap   = f"{s['test_ap_mean']:.3f} ± {s['test_ap_std']:.3f}"
        vauc = f"{s['val_auc_mean']:.3f} ± {s['val_auc_std']:.3f}"
        bold = "**" if key == "tgn" else "  "
        print(f"{bold}{label:<20}{bold}  {auc:>18}  {ap:>18}  {vauc:>18}")
    print(f"{'='*70}\n")


def make_figure(summary: dict, n_seeds: int):
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.rcParams.update({"font.family": "serif", "font.size": 11})

    labels  = ["Random\n(chance)", "Degree\nheuristic", "Static\nGNN", "DevoTG\n(TGN)"]
    keys    = ["random", "degree_heuristic", "static_gnn", "tgn"]
    colors  = [PALETTE["random"], PALETTE["degree"], PALETTE["static_gnn"], PALETTE["tgn"]]
    means   = [summary[k]["test_auc_mean"] for k in keys]
    stds    = [summary[k]["test_auc_std"]  for k in keys]

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  edgecolor="white", linewidth=0.5, width=0.5,
                  error_kw=dict(elinewidth=1.5, ecolor="#444444"))

    bars[-1].set_edgecolor("#003366"); bars[-1].set_linewidth(1.5)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.012,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold" if m == max(means) else "normal")

    ax.axhline(y=0.5, color="#AAAAAA", linestyle="--", lw=1, label="Chance")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Test AUC"); ax.set_ylim(0.35, 1.0)
    ax.set_title(f"Baseline Comparison — Test AUC  (mean ± std, {n_seeds} seeds)",
                 fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)
    plt.tight_layout()

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {OUT_FIG}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global EPOCHS
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=5,
                        help="Number of random seeds (default 5)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Epochs per run (default 20)")
    args = parser.parse_args()
    EPOCHS = args.epochs

    t_start = time.time()
    per_seed = run_all_seeds(args.n_seeds)
    summary  = summarise(per_seed)

    # Save full results
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"per_seed": per_seed, "summary": summary,
                   "n_seeds": args.n_seeds, "epochs": EPOCHS}, f, indent=2)
    print(f"\nResults saved → {OUT_JSON}")

    print_table(summary)
    make_figure(summary, args.n_seeds)
    print(f"Total runtime: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
