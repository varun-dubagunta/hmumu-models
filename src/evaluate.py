"""
evaluate.py
-----------
Load predictions produced by train.py or predict.py and generate plots.

Usage:
    python src/evaluate.py --model dnn_focal --dataset v2602c_samples_2024
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from statsmodels.stats.weightstats import DescrStatsW

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    BRIGHT_COLORS, DATA_DIR, OUTPUT_DIR,
    SIGNAL_COLORS, SIGNAL_PROCESSES,
)

ALL_PLOTS = ["hist", "hist_quantile", "cms", "sensitivity", "roc", "proba"]

VBF_PROCESS = "VBFHto2Mu_m125_amcatnlo"
WEIGHT_COL  = "Class_Weight"


def _model_paths(model_key, dataset_key):
    pred_dir = DATA_DIR / dataset_key / model_key
    out_dir  = OUTPUT_DIR / dataset_key / model_key
    return (
        pred_dir / "predictions_with_metadata.pkl",
        pred_dir / "predictions_df.pkl",
        out_dir,
    )


def _save(fig, name, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def _get_bg_processes(y):
    return [p for p in y["process"].unique() if p not in SIGNAL_PROCESSES]


def _quantile_edges(signal_df, num_bins):
    wq = DescrStatsW(data=signal_df["predictions"], weights=signal_df[WEIGHT_COL])
    return wq.quantile(np.linspace(0, 1, num_bins + 1), return_pandas=False)


# Uniform histogram 

def plot_weighted_histogram(y, output_dir, num_bins=50):
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(y["process"].unique()):
        data = y[y["process"] == name]
        h = np.histogram(data["predictions"], weights=data[WEIGHT_COL],
                         range=(-1, 1), bins=num_bins)
        ax.stairs(*h, label=name, color=BRIGHT_COLORS[i % len(BRIGHT_COLORS)])
    ax.set_xlim(-1.1, 1.1)
    ax.set_yscale("log")
    ax.set_title("Prediction Distribution by Process (Weighted)")
    ax.set_xlabel("Prediction Score")
    ax.set_ylabel("Weighted Frequency (log scale)")
    ax.legend(title="Process")
    _save(fig, "hist_weighted_uniform.png", output_dir)


#  Quantile histogram 

def plot_quantile_histogram(y, output_dir, num_bins=12):
    signal = y[y["process"] == VBF_PROCESS]
    if signal.empty:
        print(f"  [skip] {VBF_PROCESS} not in data.")
        return
    bin_edges = _quantile_edges(signal, num_bins)
    fig, ax   = plt.subplots(figsize=(10, 6))
    bin_idx   = np.arange(num_bins + 1)
    for i, name in enumerate(y["process"].unique()):
        data = y[y["process"] == name]
        counts, _ = np.histogram(data["predictions"], weights=data[WEIGHT_COL], bins=bin_edges)
        ax.stairs(counts, bin_idx, label=name, color=BRIGHT_COLORS[i % len(BRIGHT_COLORS)])
    tick_pos = np.linspace(0, num_bins, 11)
    tick_lab = [f"{bin_edges[int(p)]:.2f}" if int(p) < len(bin_edges) else "" for p in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=45, ha="right")
    ax.set_xlim(-1, num_bins + 1)
    ax.set_yscale("log")
    ax.set_title("Prediction Distribution – Quantile Binning")
    ax.set_xlabel("Prediction Score (Quantile Bins)")
    ax.set_ylabel("Weighted Frequency (log scale)")
    ax.legend(title="Process")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, "hist_weighted_quantile.png", output_dir)


# CMS stack 

def plot_cms_stack(y, output_dir, num_bins=12):
    signal = y[y["process"] == VBF_PROCESS]
    if signal.empty:
        print(f"  [skip] {VBF_PROCESS} not in data.")
        return
    bg_processes = _get_bg_processes(y)
    bin_edges    = _quantile_edges(signal, num_bins)

    sig_counts = np.zeros(num_bins)
    for name in SIGNAL_PROCESSES:
        d = y[y["process"] == name]
        if d.empty:
            continue
        c, _ = np.histogram(d["predictions"], weights=d[WEIGHT_COL], bins=bin_edges)
        sig_counts += c

    bg_counts_dict = {}
    bg_total = np.zeros(num_bins)
    for name in bg_processes:
        d = y[y["process"] == name]
        c, _ = np.histogram(d["predictions"], weights=d[WEIGHT_COL], bins=bin_edges)
        bg_counts_dict[name] = c
        bg_total += c

    valid       = bg_total > 0
    sens_per_bin = np.where(valid, sig_counts / np.sqrt(np.where(valid, bg_total, 1)), 0)
    sensitivity  = np.sqrt(np.sum(sens_per_bin ** 2))
    print(f"  Sensitivity = {sensitivity:.4f}")

    fig, ax   = plt.subplots(figsize=(10, 6))
    bin_idx   = np.arange(num_bins + 1)
    bottom    = np.zeros(num_bins)
    sorted_bg = sorted(bg_counts_dict.items(), key=lambda x: np.sum(x[1]))
    for i, (name, counts) in enumerate(sorted_bg):
        ci = list(bg_processes).index(name)
        ax.bar(bin_idx[:-1], counts, width=1, bottom=bottom, align="edge",
               label=name, color=BRIGHT_COLORS[ci % len(BRIGHT_COLORS)],
               edgecolor="black", linewidth=0.3)
        bottom += counts
    for i, name in enumerate(SIGNAL_PROCESSES):
        d = y[y["process"] == name]
        if d.empty:
            continue
        c, _ = np.histogram(d["predictions"], weights=d[WEIGHT_COL], bins=bin_edges)
        ax.stairs(c, bin_idx, label=name,
                  color=SIGNAL_COLORS[i % len(SIGNAL_COLORS)], linewidth=2.5)

    tick_pos = np.linspace(0, num_bins, 11)
    tick_lab = [f"{bin_edges[int(p)]:.2f}" if int(p) < len(bin_edges) else "" for p in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=45, ha="right")
    ax.set_xlim(-1, num_bins + 1)
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.1)
    ax.text(0.02, 0.98, f"Sensitivity: {sensitivity:.4f}",
            transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_title("Prediction Distribution – CMS style", fontsize=14, fontweight="bold")
    ax.set_xlabel("Prediction Score (Quantile Bins)", fontsize=12)
    ax.set_ylabel("Weighted Frequency (log scale)", fontsize=12)
    ax.legend(title="Process", loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    _save(fig, "hist_cms_stack.png", output_dir)


#Sensitivity scan 

def plot_sensitivity_scan(y, output_dir, bin_range=(5, 100), step=2, window=5):
    signal = y[y["process"] == VBF_PROCESS]
    if signal.empty:
        print(f"  [skip] {VBF_PROCESS} not in data.")
        return
    bg_processes  = _get_bg_processes(y)
    bin_counts    = np.arange(bin_range[0], bin_range[1] + 1, step)
    sensitivities = []
    low_flags     = []

    for nb in bin_counts:
        edges = _quantile_edges(signal, nb)
        sig_total = np.zeros(nb)
        for name in SIGNAL_PROCESSES:
            d = y[y["process"] == name]
            if d.empty:
                continue
            c, _ = np.histogram(d["predictions"], weights=d[WEIGHT_COL], bins=edges)
            sig_total += c
        bg_total = np.zeros(nb)
        for name in bg_processes:
            d = y[y["process"] == name]
            c, _ = np.histogram(d["predictions"], weights=d[WEIGHT_COL], bins=edges)
            bg_total += c
        low_flags.append(np.any(sig_total + bg_total <= 10))
        valid = bg_total > 0
        s = np.where(valid, sig_total / np.sqrt(np.where(valid, bg_total, 1)), 0)
        sensitivities.append(np.sqrt(np.sum(s ** 2)))

    sensitivities = np.array(sensitivities)
    low_flags     = np.array(low_flags)
    smoothed      = np.convolve(sensitivities, np.ones(window) / window, mode="valid")
    pad           = len(sensitivities) - len(smoothed)
    smoothed      = np.pad(smoothed, (pad // 2, pad - pad // 2), mode="edge")
    opt_idx       = np.argmax(smoothed)
    opt_bins      = bin_counts[opt_idx]
    print(f"  Optimal bins = {opt_bins}  |  smoothed sensitivity = {smoothed[opt_idx]:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bin_counts[~low_flags], sensitivities[~low_flags],
            "b-o", lw=1, ms=4, alpha=0.5, label="Raw sensitivity (safe bins)")
    ax.plot(bin_counts[low_flags], sensitivities[low_flags],
            "ro", ms=6, label="Raw sensitivity (≤10 entries in a bin)")
    ax.plot(bin_counts, smoothed, "r-", lw=2.5, label=f"Running avg (window={window})")
    ax.axvline(opt_bins, color="green", ls="--",
               label=f"Optimal: {opt_bins} bins ({smoothed[opt_idx]:.4f})")
    ax.set_xlabel("Number of Bins", fontsize=12)
    ax.set_ylabel("Sensitivity", fontsize=12)
    ax.set_title("Sensitivity vs Number of Quantile Bins", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "sensitivity_scan.png", output_dir)
    return opt_bins


# ROC 

def plot_roc(y, output_dir, tag=""):
    y2 = y.copy()
    y2["binary_label"] = y2["process"].apply(lambda p: 1 if p in SIGNAL_PROCESSES else 0)
    fpr, tpr, _ = roc_curve(y2["binary_label"], y2["predictions"],
                             sample_weight=y2[WEIGHT_COL])
    order    = np.argsort(fpr)
    fpr, tpr = fpr[order], tpr[order]
    roc_auc  = auc(fpr, tpr)
    print(f"  AUC ({tag}) = {roc_auc:.4f}")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(tpr, fpr, color="blue", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=2, ls="--", label="Random")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("True Positive Rate", fontsize=12)
    ax.set_ylabel("False Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve {tag}", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    fname = f"roc_{tag.lower().replace(' ', '_')}.png" if tag else "roc.png"
    _save(fig, fname, output_dir)


# Per-process proba

def plot_per_process_proba(predictions_df, y, process_name, output_dir, num_bins=30):
    subset = predictions_df[predictions_df["Actual_Process"] == process_name]
    if subset.empty:
        print(f"  [skip] No rows for process '{process_name}'.")
        return
    meta_cols  = {"Actual_Process", "Predicted_Process", "Class_Weight"}
    class_cols = [c for c in predictions_df.columns if c not in meta_cols]
    avg_weights = y.groupby("process")[WEIGHT_COL].mean().to_dict() \
                  if WEIGHT_COL in y.columns else {}

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, col in enumerate(class_cols):
        w = np.full(len(subset), avg_weights.get(col, 1.0))
        h = np.histogram(subset[col].values, weights=w, bins=num_bins, range=(0, 1))
        ax.stairs(*h, label=col, color=BRIGHT_COLORS[i % len(BRIGHT_COLORS)])
    ax.set_title(f"Predicted Probability Distribution – Actual Process: {process_name}")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Weighted Frequency (log scale)")
    ax.set_yscale("log")
    ax.legend(title="Predicted Class")
    ax.grid(True)
    fig.tight_layout()
    _save(fig, f"proba_dist_{process_name}.png", output_dir)


# Entry point

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="transformer_focal",
                        choices=["transformer_focal", "transformer_ce", "transformer_supcon",
                                 "gat", "dnn", "dnn_focal", "dnn_aux", "combination",
                                 "interaction_net", "significance", "significance_ft"])
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--plots",   nargs="+", default=ALL_PLOTS,
                        choices=ALL_PLOTS, metavar="PLOT")
    args = parser.parse_args()

    plots = set(args.plots)
    from config import RAW_DATA_FILE, PROCESS_REMAP
    dataset_key = args.dataset if args.dataset else RAW_DATA_FILE.stem
    pred_file, preds_df_file, output_dir = _model_paths(args.model, dataset_key)

    print(f"\n{'='*60}")
    print(f"  Model   : {args.model}")
    print(f"  Dataset : {dataset_key}")
    print(f"  Output  : {output_dir}")
    print(f"  Plots   : {', '.join(sorted(plots))}")
    print(f"{'='*60}\n")

    if not pred_file.exists():
        print(f"ERROR: {pred_file} not found.")
        sys.exit(1)

    y = pd.read_pickle(str(pred_file))
    y["process"] = y["process"].replace(PROCESS_REMAP)

    has_weight = WEIGHT_COL in y.columns
    if not has_weight:
        print(f"  [warn] '{WEIGHT_COL}' not in data — skipping weighted plots.")

    predictions_df = None
    if "proba" in plots:
        if preds_df_file.exists():
            predictions_df = pd.read_pickle(str(preds_df_file))
            predictions_df["Actual_Process"] = predictions_df["Actual_Process"].replace(PROCESS_REMAP)
        else:
            print(f"  [skip] proba – file not found.")
            plots.discard("proba")

    print("── Histograms ──────────────────────────────────────────────────────")
    if "hist" in plots and has_weight:
        plot_weighted_histogram(y, output_dir)

    if not has_weight:
        for p in ("hist_quantile", "cms", "sensitivity", "roc"):
            plots.discard(p)
            print(f"  [skip] {p} – no weight column.")

    if "hist_quantile" in plots:
        plot_quantile_histogram(y, output_dir)
    if "cms" in plots:
        plot_cms_stack(y, output_dir)
    if "sensitivity" in plots:
        print("── Sensitivity scan ────────────────────────────────────────────────")
        plot_sensitivity_scan(y, output_dir)
    if "roc" in plots:
        print("── ROC curves ──────────────────────────────────────────────────────")
        plot_roc(y, output_dir, tag="Class Weight")
    if "proba" in plots and predictions_df is not None:
        print("── Per-process probability distributions ────────────────────────────")
        for proc in y["process"].unique():
            plot_per_process_proba(predictions_df, y, proc, output_dir)

    print(f"\nDone – plots saved to {output_dir}")


if __name__ == "__main__":
    main()
