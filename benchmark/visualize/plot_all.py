#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_all.py
===========

整合所有基准测试可视化绘图函数，包括：
  - LOD 曲线
  - ROC/PR 曲线
  - 性能对比柱状图
  - 边界误差箱线图
  - 概率 KDE 分布

每个绘图功能封装为一个函数，可通过命令行参数选择调用。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # 非交互后端
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置全局样式
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12

# 增加第四个条件，调整配色方案
COLORS = {
    "MicroDNA Map (WGS)": "#E63946",            # 红色
    "MicroDNA Map (Circle-seq)": "#F4A261",     # 橙色
    "Circle-Map (Circle-seq)": "#457B9D",       # 深蓝
    "Circle-Map (WGS)": "#A8DADC",              # 浅蓝
}

# 默认 LOD 阈值
DEFAULT_LOD_THRESHOLD = 0.8


# -----------------------------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------------------------

def parse_summary_metrics(file_path: Path) -> Dict[int, Dict[str, float]]:
    if not file_path or not file_path.exists():
        return {}
    df = pd.read_csv(file_path, sep="\t")
    if df.empty:
        return {}
    df.columns = [c.strip() for c in df.columns]
    result = {}
    for _, row in df.iterrows():
        cn = int(row["copy_number"])
        metrics = {}
        for col in df.columns:
            if col != "copy_number":
                try:
                    val = float(row[col])
                except (ValueError, TypeError):
                    val = np.nan
                metrics[col] = val
        result[cn] = metrics
    return result


def parse_lod_json(file_path: Path, threshold: float) -> Optional[int]:
    if not file_path or not file_path.exists():
        return None
    with open(file_path, "r") as f:
        data = json.load(f)
    lod = data.get("lod")
    return int(lod) if lod is not None else None


def save_figure(fig: plt.Figure, output_dir: Path, basename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{basename}.pdf"
    png_path = output_dir / f"{basename}.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] 已保存: {pdf_path}")
    print(f"[INFO] 已保存: {png_path}")


# -----------------------------------------------------------------------------
# 绘图函数
# -----------------------------------------------------------------------------

def plot_lod(
    microdna_metrics: Optional[Path],
    microdna_circseq_metrics: Optional[Path],
    circseq_metrics: Optional[Path],
    circseq_wgs_metrics: Optional[Path],
    output_dir: Path,
    lod_threshold: float = DEFAULT_LOD_THRESHOLD,
    lod_json_microdna: Optional[Path] = None,
    lod_json_microdna_circseq: Optional[Path] = None,
    lod_json_circseq: Optional[Path] = None,
    lod_json_circseq_wgs: Optional[Path] = None,
) -> None:
    
    methods = {
        "MicroDNA Map (WGS)": microdna_metrics,
        "MicroDNA Map (Circle-seq)": microdna_circseq_metrics,
        "Circle-Map (Circle-seq)": circseq_metrics,
        "Circle-Map (WGS)": circseq_wgs_metrics,
    }
    
    data = {}
    for method, path in methods.items():
        if path and path.exists():
            df = pd.read_csv(path, sep="\t")
            if not df.empty and "recall" in df.columns and "copy_number" in df.columns:
                data[method] = dict(zip(df["copy_number"].astype(int), df["recall"].astype(float)))

    if not data:
        print("[ERROR] 没有任何方法的有效数据，无法绘制 LOD 曲线", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for method, recalls in data.items():
        sorted_cns = sorted(recalls.keys())
        sorted_recalls = [recalls[cn] for cn in sorted_cns]
        ax.plot(sorted_cns, sorted_recalls, marker="o", linewidth=2,
                color=COLORS.get(method, "#000000"), label=method)

    # 标注 LOD
    lod_files = {
        "MicroDNA Map (WGS)": lod_json_microdna,
        "MicroDNA Map (Circle-seq)": lod_json_microdna_circseq,
        "Circle-Map (Circle-seq)": lod_json_circseq,
        "Circle-Map (WGS)": lod_json_circseq_wgs,
    }
    
    for method, lod_file in lod_files.items():
        if lod_file is not None and lod_file.exists() and method in data:
            lod = parse_lod_json(lod_file, lod_threshold)
            if lod is not None and lod in data[method]:
                recall_at_lod = data[method][lod]
                ax.scatter([lod], [recall_at_lod], s=100, facecolors='none',
                           edgecolors=COLORS.get(method, "#000000"), linewidths=2, zorder=5)
                ax.annotate(f"LOD={lod}x", xy=(lod, recall_at_lod),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=10, color=COLORS.get(method, "#000000"))

    ax.set_xlabel("Copy number (x)")
    ax.set_ylabel("Recall")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 5, 10, 50])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title("Limit of Detection (LOD) across copy numbers")
    plt.tight_layout()
    save_figure(fig, output_dir, "lod_curve")


def plot_comparison(
    microdna_metrics: Optional[Path],
    microdna_circseq_metrics: Optional[Path],
    circseq_metrics: Optional[Path],
    circseq_wgs_metrics: Optional[Path],
    output_dir: Path,
) -> None:
    
    methods_data = {}
    if microdna_metrics and microdna_metrics.exists():
        methods_data["MicroDNA Map (WGS)"] = parse_summary_metrics(microdna_metrics)
    if microdna_circseq_metrics and microdna_circseq_metrics.exists():
        methods_data["MicroDNA Map (Circle-seq)"] = parse_summary_metrics(microdna_circseq_metrics)
    if circseq_metrics and circseq_metrics.exists():
        methods_data["Circle-Map (Circle-seq)"] = parse_summary_metrics(circseq_metrics)
    if circseq_wgs_metrics and circseq_wgs_metrics.exists():
        methods_data["Circle-Map (WGS)"] = parse_summary_metrics(circseq_wgs_metrics)

    all_cns = sorted(set().union(*[set(d.keys()) for d in methods_data.values() if d]))
    if not all_cns:
        print("[ERROR] 没有可用数据绘制 comparison 图", file=sys.stderr)
        return

    metrics = ["precision", "recall", "f1"]
    n_methods = len(methods_data)
    n_groups = len(all_cns)
    bar_width = 0.8 / max(n_methods, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        for m_idx, (method, data_dict) in enumerate(methods_data.items()):
            values = []
            for cn in all_cns:
                if cn in data_dict and metric in data_dict[cn]:
                    values.append(data_dict[cn][metric])
                else:
                    values.append(0.0)
            x = np.arange(n_groups) + (m_idx - (n_methods-1)/2) * bar_width
            ax.bar(x, values, bar_width, label=method, color=COLORS.get(method, "#000000"))
            
        ax.set_xticks(np.arange(n_groups))
        ax.set_xticklabels([f"{cn}x" for cn in all_cns])
        ax.set_ylim(0, 1.0)
        ax.set_title(metric.capitalize())
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("Performance comparison across copy numbers", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, output_dir, "comparison_metrics")


def plot_roc_pr(roc_data: Path, output_dir: Path) -> None:
    if not roc_data.exists():
        print(f"[ERROR] ROC 数据文件不存在: {roc_data}", file=sys.stderr)
        return
    df = pd.read_csv(roc_data, sep="\t")
    required_cols = {"copy_number", "fpr", "tpr", "precision", "recall"}
    if not required_cols.issubset(set(df.columns)):
        return

    copy_numbers = sorted(df["copy_number"].unique())
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))

    for cn in copy_numbers:
        sub = df[df["copy_number"] == cn].sort_values("threshold")
        ax_roc.plot(sub["fpr"], sub["tpr"], marker=".", label=f"{cn}x")
        ax_pr.plot(sub["recall"], sub["precision"], marker=".", label=f"{cn}x")

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves")
    ax_roc.legend(title="Copy number")
    ax_roc.grid(alpha=0.3)

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curves")
    ax_pr.legend(title="Copy number")
    ax_pr.grid(alpha=0.3)
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1)

    plt.tight_layout()
    save_figure(fig, output_dir, "roc_pr_curves")


def plot_boundary(boundary_data: Path, output_dir: Path) -> None:
    if not boundary_data.exists():
        return
    df = pd.read_csv(boundary_data, sep="\t")
    if "copy_number" not in df.columns or "error" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    df.boxplot(column="error", by="copy_number", ax=ax)
    ax.set_xlabel("Copy number")
    ax.set_ylabel("Boundary error (bp)")
    ax.set_title("Boundary error distribution")
    plt.suptitle("")
    plt.tight_layout()
    save_figure(fig, output_dir, "boundary_error_boxplot")


def main() -> None:
    parser = argparse.ArgumentParser(description="整合的绘图工具")
    parser.add_argument("--plot", required=True,
                        choices=["lod", "comparison", "roc_pr", "boundary", "kde"],
                        help="要绘制的图类型")
    # 共有参数
    parser.add_argument("--microdna_metrics", type=Path, help="MicroDNA Map (WGS) summary_metrics.tsv")
    parser.add_argument("--microdna_circseq_metrics", type=Path, help="MicroDNA Map (Circle-seq) summary_metrics.tsv")
    parser.add_argument("--circseq_metrics", type=Path, help="Circle-Map (Circle-seq) summary_metrics.tsv")
    parser.add_argument("--circseq_wgs_metrics", type=Path, help="Circle-Map (WGS) summary_metrics.tsv")
    
    # LOD 特有参数
    parser.add_argument("--lod_threshold", type=float, default=DEFAULT_LOD_THRESHOLD,
                        help=f"LOD 阈值 (默认: {DEFAULT_LOD_THRESHOLD})")
    parser.add_argument("--lod_json_microdna", type=Path, help="MicroDNA Map (WGS) lod_analysis.json")
    parser.add_argument("--lod_json_microdna_circseq", type=Path, help="MicroDNA Map (Circle-seq) lod_analysis.json")
    parser.add_argument("--lod_json_circseq", type=Path, help="Circle-Map (Circle-seq) lod_analysis.json")
    parser.add_argument("--lod_json_circseq_wgs", type=Path, help="Circle-Map (WGS) lod_analysis.json")
    
    # 其它参数
    parser.add_argument("--roc_data", type=Path, help="ROC/PR 数据文件（TSV）")
    parser.add_argument("--boundary_data", type=Path, help="边界误差数据文件（TSV）")
    parser.add_argument("--output_dir", required=True, type=Path, help="输出目录")

    args = parser.parse_args()

    if args.plot == "lod":
        plot_lod(
            args.microdna_metrics,
            args.microdna_circseq_metrics,
            args.circseq_metrics,
            args.circseq_wgs_metrics,
            args.output_dir,
            args.lod_threshold,
            args.lod_json_microdna,
            args.lod_json_microdna_circseq,
            args.lod_json_circseq,
            args.lod_json_circseq_wgs,
        )
    elif args.plot == "comparison":
        plot_comparison(
            args.microdna_metrics,
            args.microdna_circseq_metrics,
            args.circseq_metrics,
            args.circseq_wgs_metrics,
            args.output_dir,
        )
    elif args.plot == "roc_pr":
        plot_roc_pr(args.roc_data, args.output_dir)
    elif args.plot == "boundary":
        plot_boundary(args.boundary_data, args.output_dir)

if __name__ == "__main__":
    main()