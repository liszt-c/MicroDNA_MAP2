#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_microdna_map_direct.py
==========================

直接使用 MicroDNA Map v2.0 深度学习模型扫描参考基因组序列,
不依赖 CNVkit 调用 CNV 区域。将指定基因组区域切分为连续长片段,
调用 scripts/predict.py --mode long 进行滑动窗口检测, 最后合并输出标准 BED。

与 run_microdna_map.py 的区别:
  - 无需比对 reads 和 CNV 调用
  - 直接从参考基因组序列中扫描 microDNA
  - 可作为 MicroDNA Map 独立于 CNV 的替代检测模式

依赖:
  - pysam (读取参考基因组)
  - scripts/predict.py 及模型文件

使用示例:
  python benchmark/detect/run_microdna_map_direct.py \
      --reference /path/to/hg19.fa \
      --output_dir benchmark/results/detect/microdna_direct \
      --model_path ./models/6.pth \
      --region chr21 \
      --limit 0.99
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pysam

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "results" / "detect" / "microdna_direct"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "6.pth"
DEFAULT_LIMIT = "0.99"
DEFAULT_SEGMENT_LENGTH = 1_000_000  # 1 Mb


def parse_region(region_str: str) -> Tuple[str, int, int]:
    """
    解析区域字符串: chr1 或 chr1:1000-2000
    返回 (chrom, start, end), 0-based 半开区间。end=None 表示到染色体末尾。
    """
    if ":" in region_str:
        chrom, coords = region_str.split(":", 1)
        if "-" in coords:
            start_s, end_s = coords.split("-", 1)
            return chrom, int(start_s), int(end_s)
        else:
            return chrom, int(coords), None
    else:
        return region_str, None, None


def read_region_file(bed_path: Path) -> List[Tuple[str, int, int]]:
    """从 BED 文件读取区域。"""
    regions = []
    with open(bed_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                regions.append((parts[0], int(parts[1]), int(parts[2])))
    return regions


def write_fasta_segments(
    fasta: pysam.FastaFile,
    chrom: str,
    start: int,
    end: int,
    segment_length: int,
    output_dir: Path,
) -> List[Path]:
    """
    将 [start, end) 按 segment_length 切分, 写入 FASTA。
    Header 格式: >chrom:seg_start-seg_end (predict.py 可解析)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    pos = start
    while pos < end:
        seg_end = min(pos + segment_length, end)
        seg_seq = fasta.fetch(chrom, pos, seg_end).upper()
        if len(seg_seq) == 0:
            pos = seg_end
            continue
        header = f">{chrom}:{pos}-{seg_end}"
        fasta_file = output_dir / f"segment_{chrom}_{pos}_{seg_end}.fa"
        with open(fasta_file, "w") as f:
            f.write(f"{header}\n")
            for i in range(0, len(seg_seq), 80):
                f.write(seg_seq[i:i+80] + "\n")
        files.append(fasta_file)
        pos = seg_end
    return files


def run_microdna_direct(
    reference: Path,
    output_dir: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
    limit: str = DEFAULT_LIMIT,
    regions: Optional[List[Tuple[str, int, int]]] = None,
    segment_length: int = DEFAULT_SEGMENT_LENGTH,
) -> Path:
    """执行直接扫描流程, 返回标准化 BED 文件路径。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "microdna_direct.log"
    print(f"[INFO] 输出目录: {output_dir}")

    # 确保 faidx 索引
    if not (reference.parent / (reference.name + ".fai")).exists():
        print("[INFO] 建立参考基因组 faidx 索引...")
        subprocess.run(["samtools", "faidx", str(reference)], check=True)

    fasta = pysam.FastaFile(str(reference))

    if not regions:
        print("[WARN] 未指定扫描区域, 将扫描所有染色体 (可能非常耗时)")
        regions = []
        for ref in fasta.references:
            length = fasta.get_reference_length(ref)
            regions.append((ref, 0, length))

    # 切分片段
    segment_dir = output_dir / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    for f in segment_dir.glob("*.fa"):
        f.unlink()

    total_segments = 0
    for chrom, start, end in regions:
        if chrom not in fasta.references:
            print(f"[WARN] 染色体 {chrom} 不在参考基因组中, 跳过")
            continue
        chrom_len = fasta.get_reference_length(chrom)
        if start is None:
            start = 0
        if end is None or end > chrom_len:
            end = chrom_len
        if start >= end:
            print(f"[WARN] 区域 {chrom}:{start}-{end} 无效, 跳过")
            continue
        print(f"[INFO] 扫描区域 {chrom}:{start}-{end}, 长度 {end-start} bp")
        files = write_fasta_segments(fasta, chrom, start, end, segment_length, segment_dir)
        total_segments += len(files)

    fasta.close()

    final_bed = output_dir / "detected_microdna_direct.bed"
    if total_segments == 0:
        print("[ERROR] 没有任何片段可扫描")
        final_bed.write_text("")
        return final_bed

    print(f"[INFO] 共生成 {total_segments} 个片段 FASTA")

    # 检查 predict.py
    predict_script = PROJECT_ROOT / "scripts" / "predict.py"
    if not predict_script.exists():
        raise FileNotFoundError(f"找不到 predict.py: {predict_script}")

    # 检查模型
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 运行 predict.py --mode long
    predict_output = output_dir / "predict_results"
    predict_output.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 运行 MicroDNA Map 直接扫描...")
    start_time = time.time()
    cmd = [
        sys.executable, str(predict_script),
        "--input", str(segment_dir),
        "--mode", "long",
        "--model", str(model_path),
        "--limit", str(limit),
        "--output-dir", str(predict_output),
    ]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"[ERROR] 直接扫描失败 (exit code {result.returncode})")
        raise RuntimeError("MicroDNA Map 直接扫描失败")

    print(f"[INFO] 扫描完成, 耗时 {elapsed:.2f} 秒")

    # 收集并合并 BED
    bed_files = list(predict_output.glob("*.bed"))
    if not bed_files:
        print("[WARN] 未检测到任何 microDNA (无 BED 文件)")
        final_bed.write_text("")
        return final_bed

    count = 0
    with open(final_bed, "w") as out_f:
        for bed in bed_files:
            with open(bed, "r") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    chrom, start, end = parts[0], parts[1], parts[2]
                    out_f.write(f"{chrom}\t{start}\t{end}\tmicrodna_{count}\t1.0\n")
                    count += 1

    print(f"[INFO] 合并完成, 共 {count} 条记录 -> {final_bed}")

    with open(log_file, "a") as log:
        log.write(f"Total time: {elapsed:.2f}s, detections: {count}\n")

    return final_bed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="直接使用 MicroDNA Map v2.0 模型扫描参考基因组 (不依赖 CNV)"
    )
    parser.add_argument("--reference", required=True, type=Path,
                        help="参考基因组 FASTA 文件")
    parser.add_argument("--output_dir", required=True, type=Path,
                        help="输出目录")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH,
                        help="模型文件路径 (默认: models/6.pth)")
    parser.add_argument("--limit", type=str, default=DEFAULT_LIMIT,
                        help="识别阈值 (默认: 0.99)")
    parser.add_argument("--region", action="append", type=str, default=None,
                        help="指定扫描区域, 格式 chr1 或 chr1:1000-2000, 可多次指定")
    parser.add_argument("--region_file", type=Path, default=None,
                        help="BED 文件指定扫描区域, 与 --region 可同时使用")
    parser.add_argument("--segment_length", type=int, default=DEFAULT_SEGMENT_LENGTH,
                        help=f"切分片段长度 (默认: {DEFAULT_SEGMENT_LENGTH})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.reference.exists():
        print(f"[ERROR] 参考基因组文件不存在: {args.reference}")
        sys.exit(1)

    regions = []
    if args.region:
        for region_str in args.region:
            chrom, start, end = parse_region(region_str)
            regions.append((chrom, start, end))
    if args.region_file:
        if args.region_file.exists():
            regions.extend(read_region_file(args.region_file))
        else:
            print(f"[ERROR] region_file 不存在: {args.region_file}")
            sys.exit(1)

    try:
        detected_bed = run_microdna_direct(
            reference=args.reference,
            output_dir=args.output_dir,
            model_path=args.model_path,
            limit=args.limit,
            regions=regions if regions else None,
            segment_length=args.segment_length,
        )
        print(f"\n[INFO] 直接扫描完成, 结果 BED: {detected_bed}")
    except Exception as e:
        print(f"[ERROR] 执行失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()