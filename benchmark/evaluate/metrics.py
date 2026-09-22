#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics.py
==========

评估 microDNA 检测性能。
采用基因组变异检测标准评测准则：
- Recall: 真实位点被检测到的比例 (Truth-perspective Recall)
- Precision: 检测结果落在真实位点上的比例 (Detected-perspective Precision)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "results" / "evaluation"

DEFAULT_OVERLAP_THRESHOLD = 0.5
DEFAULT_LOD_THRESHOLD = 0.8


def parse_bed(file_path: Path) -> List[Tuple[str, int, int]]:
    regions = []
    if not file_path.exists():
        print(f"[WARN] BED 文件不存在: {file_path}", file=sys.stderr)
        return regions
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            except ValueError:
                continue
            if start < end:
                regions.append((chrom, start, end))
    return regions


def overlap_length(a: Tuple[str, int, int], b: Tuple[str, int, int]) -> int:
    if a[0] != b[0]:
        return 0
    return max(0, min(a[2], b[2]) - max(a[1], b[1]))


def evaluate_pair(
    truth_regions: List[Tuple[str, int, int]],
    detected_regions: List[Tuple[str, int, int]],
    overlap_threshold: float,
) -> Dict:
    total_truth = len(truth_regions)
    total_detected = len(detected_regions)

    if total_truth == 0:
        return {
            "tp_truth": 0,
            "tp_detected": 0,
            "tp": 0,
            "fp": total_detected,
            "fn": 0,
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "boundary_errors": [],
            "mean_boundary_error": None,
        }

    if total_detected == 0:
        return {
            "tp_truth": 0,
            "tp_detected": 0,
            "tp": 0,
            "fp": 0,
            "fn": total_truth,
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "boundary_errors": [],
            "mean_boundary_error": None,
        }

    # 1. Truth 视角: 每个 truth 是否被至少一个 detected 命中 (计算 Recall)
    tp_truth = 0
    for truth in truth_regions:
        truth_len = truth[2] - truth[1]
        is_hit = False
        for det in detected_regions:
            ov = overlap_length(truth, det)
            if ov <= 0:
                continue
            # 重叠比例判定: 覆盖 truth 或互反重叠达标
            if (ov / truth_len) >= overlap_threshold:
                is_hit = True
                break
        if is_hit:
            tp_truth += 1

    fn = total_truth - tp_truth
    recall = tp_truth / total_truth if total_truth > 0 else 0.0

    # 2. Detected 视角: 预测的区间中有多少真正命中了真实位点 (计算 Precision)
    tp_detected = 0
    boundary_errors = []

    for det in detected_regions:
        det_len = det[2] - det[1]
        best_overlap = 0
        best_truth = None

        for truth in truth_regions:
            ov = overlap_length(truth, det)
            if ov <= 0:
                continue
            truth_len = truth[2] - truth[1]
            # 满足任一方向的重叠率
            ratio_truth = ov / truth_len if truth_len > 0 else 0
            ratio_det = ov / det_len if det_len > 0 else 0

            if max(ratio_truth, ratio_det) >= overlap_threshold:
                if ov > best_overlap:
                    best_overlap = ov
                    best_truth = truth

        if best_truth is not None:
            tp_detected += 1
            error = abs(det[1] - best_truth[1]) + abs(det[2] - best_truth[2])
            boundary_errors.append(error)

    fp = total_detected - tp_detected
    precision = tp_detected / total_detected if total_detected > 0 else 0.0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_bd = sum(boundary_errors) / len(boundary_errors) if boundary_errors else None

    return {
        "tp_truth": tp_truth,
        "tp_detected": tp_detected,
        "tp": tp_detected,  # 映射回通用 tp 字段，表示有效真阳性预测数
        "fp": fp,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "boundary_errors": boundary_errors,
        "mean_boundary_error": mean_bd,
    }


def parse_detected_inputs(
    detected_dir: Optional[Path],
    detected_bed_args: Optional[List[str]],
) -> Dict[int, Path]:
    detected_dict: Dict[int, Path] = {}
    if detected_bed_args:
        for arg in detected_bed_args:
            if ":" not in arg:
                continue
            cn_str, path_str = arg.split(":", 1)
            try:
                cn = int(cn_str)
            except ValueError:
                continue
            bed_path = Path(path_str)
            if bed_path.exists():
                detected_dict[cn] = bed_path

    if detected_dir:
        detected_dir = Path(detected_dir)
        if detected_dir.is_dir():
            for sub_dir in detected_dir.iterdir():
                if not sub_dir.is_dir() or not sub_dir.name.startswith("cn"):
                    continue
                try:
                    cn = int(sub_dir.name[2:])
                except ValueError:
                    continue
                bed_files = list(sub_dir.glob("*.bed"))
                if not bed_files:
                    continue
                preferred = [f for f in bed_files if "detected" in f.name.lower()]
                detected_dict[cn] = preferred[0] if preferred else bed_files[0]

    return dict(sorted(detected_dict.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="评估检测性能 (双向真阳性标准)")
    parser.add_argument("--truth_bed", required=True, type=Path)
    parser.add_argument("--detected_dir", type=Path, default=None)
    parser.add_argument("--detected_bed", action="append", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overlap_threshold", type=float, default=DEFAULT_OVERLAP_THRESHOLD)
    parser.add_argument("--lod_threshold", type=float, default=DEFAULT_LOD_THRESHOLD)
    args = parser.parse_args()

    truth_regions = parse_bed(args.truth_bed)
    if not truth_regions:
        print("[ERROR] truth BED 为空", file=sys.stderr)
        sys.exit(1)

    detected_dict = parse_detected_inputs(args.detected_dir, args.detected_bed)
    if not detected_dict:
        print("[ERROR] 未提供检测结果", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    lod_data = {"copy_numbers": [], "recalls": [], "lod": None}

    for cn, bed_path in detected_dict.items():
        detected_regions = parse_bed(bed_path)
        res = evaluate_pair(truth_regions, detected_regions, args.overlap_threshold)

        metrics_file = output_dir / f"metrics_{cn}x.tsv"
        with open(metrics_file, "w") as f:
            f.write(
                "copy_number\ttotal_truth\ttotal_detected\ttp\tfp\tfn\t"
                "recall\tprecision\tf1\tmean_boundary_error\n"
            )
            f.write(
                f"{cn}\t{len(truth_regions)}\t{len(detected_regions)}\t"
                f"{res['tp']}\t{res['fp']}\t{res['fn']}\t"
                f"{res['recall']:.6f}\t{res['precision']:.6f}\t{res['f1']:.6f}\t"
                f"{res['mean_boundary_error'] if res['mean_boundary_error'] is not None else 'N/A'}\n"
            )

        summary_rows.append({
            "copy_number": cn,
            "total_truth": len(truth_regions),
            "total_detected": len(detected_regions),
            **res
        })
        lod_data["copy_numbers"].append(cn)
        lod_data["recalls"].append(res["recall"])

        boundary_file = output_dir / f"boundary_errors_{cn}x.tsv"
        with open(boundary_file, "w") as bf:
            bf.write("error\n")
            for e in res["boundary_errors"]:
                bf.write(f"{e}\n")

    summary_file = output_dir / "summary_metrics.tsv"
    with open(summary_file, "w") as f:
        f.write(
            "copy_number\ttotal_truth\ttotal_detected\ttp\tfp\tfn\t"
            "recall\tprecision\tf1\tmean_boundary_error\n"
        )
        for row in summary_rows:
            f.write(
                f"{row['copy_number']}\t{row['total_truth']}\t{row['total_detected']}\t"
                f"{row['tp']}\t{row['fp']}\t{row['fn']}\t"
                f"{row['recall']:.6f}\t{row['precision']:.6f}\t{row['f1']:.6f}\t"
                f"{row['mean_boundary_error'] if row['mean_boundary_error'] is not None else 'N/A'}\n"
            )

    for cn, recall in zip(lod_data["copy_numbers"], lod_data["recalls"]):
        if recall >= args.lod_threshold:
            lod_data["lod"] = cn
            break

    with open(output_dir / "lod_analysis.json", "w") as jf:
        json.dump(lod_data, jf, indent=2)

    print(f"[INFO] 评估完成，汇总表: {summary_file}")
    print(f"[INFO] LOD (recall >= {args.lod_threshold}): {lod_data['lod']}")


if __name__ == "__main__":
    main()