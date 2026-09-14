#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_microdna_map.py
===================

封装 MicroDNA Map v2.0 对 WGS spike-in 数据的检测流程。

流程:
  1. Bowtie2 比对 + samtools sort/index
  2. CNVkit batch + segment + call (引入动态基线裁剪，修复全局中位数偏移导致的假阳性)
  3. 从 .call.cns 提取候选 CNV 区域序列 (合并 FASTA)
  4. 调用 scripts/predict.py --mode long 进行滑窗预测
  5. 合并输出标准 BED
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "results" / "detect" / "microdna_map"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "6.pth"
DEFAULT_LIMIT = "0.99"
DEFAULT_THREADS = 8


def check_tool(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run_cmd(cmd: list, cwd: Optional[Path] = None, check: bool = True) -> int:
    print(f"[CMD] {' '.join(map(str, cmd))}")
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"命令失败: {' '.join(map(str, cmd))}")
    return result.returncode


def build_bowtie2_index(reference_fasta: Path, output_prefix: Path) -> None:
    idx_ext = [".1.bt2", ".2.bt2", ".3.bt2", ".4.bt2", ".rev.1.bt2", ".rev.2.bt2"]
    if all(Path(str(output_prefix) + ext).exists() for ext in idx_ext):
        print(f"[INFO] Bowtie2 索引已存在: {output_prefix}")
        return
    print(f"[INFO] 构建 Bowtie2 索引: {output_prefix}")
    run_cmd(["bowtie2-build", str(reference_fasta), str(output_prefix)], check=True)


def align_and_sort_reads(r1: Path, r2: Path, ref_prefix: Path, output_bam: Path, threads: int) -> None:
    print(f"[INFO] 比对 reads: {r1.name} & {r2.name}")
    cmd = (f"bowtie2 -p {threads} -x {ref_prefix} -1 {r1} -2 {r2} --no-unal | "
           f"samtools sort -@{threads} -o {output_bam}")
    ret = subprocess.run(cmd, shell=True, executable="/bin/bash", capture_output=True, text=True)
    if ret.returncode != 0:
        raise RuntimeError(f"比对失败: {ret.stderr}")
    print(f"[INFO] 比对完成: {output_bam}")


def index_bam(bam_path: Path, threads: int) -> None:
    run_cmd(["samtools", "index", "-@", str(threads), str(bam_path)], check=True)


def create_sub_reference(orig_cnn: Path, allowed_chroms: set, output_dir: Path) -> Path:
    """
    [核心修复] 动态裁剪 CNVkit Reference (.cnn)。
    如果 WGS 只模拟了 chr1-5，直接使用全基因组的 .cnn 会导致 CNVkit 的全局中位数计算完全崩盘
    （80% 无覆盖区域将中位数拉向极负，导致 chr1-5 的正常 30x 背景被错误地评估为染色体级别的巨大扩增）。
    通过丢弃未测序的染色体，让 CNVkit 只在局部“世界观”里计算，从而恢复绝对精准的拷贝数计算。
    """
    sub_cnn_path = output_dir / f"sub_reference.cnn"
    if not orig_cnn.exists():
        raise FileNotFoundError(f"Original .cnn not found: {orig_cnn}")
        
    print(f"[INFO] 正在裁剪 CNVkit 参考基线以匹配局部 WGS 模拟 (保留染色体: {allowed_chroms})...")
    with open(orig_cnn, 'r', encoding='utf-8') as fin, \
         open(sub_cnn_path, 'w', encoding='utf-8') as fout:
        # 读取并写入表头
        header = fin.readline()
        fout.write(header)
        
        # 假设第一列通常是 chromosome，解析列索引
        h_parts = header.strip().split('\t')
        chrom_idx = h_parts.index("chromosome") if "chromosome" in h_parts else 0
        
        kept = 0
        for line in fin:
            parts = line.strip().split('\t')
            if len(parts) > chrom_idx and parts[chrom_idx] in allowed_chroms:
                fout.write(line)
                kept += 1
                
    print(f"[INFO] 基线裁剪完成，保留了 {kept} 个 Reference Bins -> {sub_cnn_path.name}")
    return sub_cnn_path


def run_cnvkit(bam_path: Path, cnvkit_reference: Path, output_dir: Path, threads: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(["cnvkit.py", "batch", "-m", "wgs", "-r", str(cnvkit_reference),
             "-p", str(threads), "-d", str(output_dir), str(bam_path)], check=True)

    cnr_files = list(output_dir.glob("*.cnr"))
    if not cnr_files:
        raise FileNotFoundError("CNVkit batch 未生成 .cnr 文件")
    cnr_file = cnr_files[0]

    result_cns = output_dir / "result.cns"
    run_cmd(["cnvkit.py", "segment", str(cnr_file), "-p", str(threads),
             "-m", "cbs", "-o", str(result_cns)], check=True)

    result_call = output_dir / "result.call.cns"
    run_cmd(["cnvkit.py", "call", str(result_cns), "-o", str(result_call)], check=True)
    return result_call


def extract_cnv_regions_fasta(cnv_call_file: Path, reference_fasta: Path,
                              output_fa: Path, min_length: int = 200, allowed_chroms: set = None) -> int:
    """
    从 .call.cns 提取候选区域序列, 写入单个合并 FASTA。
    加入 allowed_chroms 过滤，作为双重保险，阻断一切错配产生的漏网之鱼。
    """
    output_fa.parent.mkdir(parents=True, exist_ok=True)

    if not (reference_fasta.parent / (reference_fasta.name + ".fai")).exists():
        run_cmd(["samtools", "faidx", str(reference_fasta)], check=True)

    lines = []
    with open(cnv_call_file, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    if len(lines) < 2:
        output_fa.write_text("")
        return 0

    header = lines[0].split("\t")
    try:
        chrom_idx = header.index("chromosome")
        start_idx = header.index("start")
        end_idx = header.index("end")
    except ValueError:
        chrom_idx, start_idx, end_idx = 0, 1, 2

    written = 0
    with open(output_fa, "w") as fout:
        for i, line in enumerate(lines[1:]):
            parts = line.split("\t")
            if len(parts) <= max(chrom_idx, start_idx, end_idx):
                continue
            
            chrom = parts[chrom_idx]
            
            # === 基于预设背景染色体的过滤 ===
            if allowed_chroms is not None and chrom not in allowed_chroms:
                continue

            try:
                start, end = int(parts[start_idx]), int(parts[end_idx])
            except ValueError:
                continue
            if end - start < min_length:
                continue

            # samtools faidx 使用 1-based 闭区间
            cmd = ["samtools", "faidx", str(reference_fasta), f"{chrom}:{start}-{end}"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                continue

            # 重写 header 为 predict.py 可解析的格式: >cnv_i|chr:start-end
            seq_lines = result.stdout.strip().split("\n")
            if not seq_lines:
                continue
            fout.write(f">cnv_{i}|{chrom}:{start}-{end}\n")
            for sl in seq_lines[1:]:
                fout.write(sl + "\n")
            written += 1

    print(f"[INFO] 提取了 {written} 个 CNV 区域序列 -> {output_fa}")
    return written


def run_microdna_predict(input_fa: Path, model_path: Path, limit: str,
                         output_dir: Path) -> Optional[Path]:
    predict_script = PROJECT_ROOT / "scripts" / "predict.py"
    if not predict_script.exists():
        raise FileNotFoundError(f"找不到 predict.py: {predict_script}")

    predict_output = output_dir / "predict_results"
    predict_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(predict_script),
        "--input", str(input_fa),
        "--mode", "long",
        "--model", str(model_path),
        "--limit", str(limit),
        "--output-dir", str(predict_output),
    ]

    start = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print("[ERROR] MicroDNA Map predict.py 执行失败")
        return None

    print(f"[INFO] MicroDNA Map 预测完成, 耗时 {elapsed:.2f}s")

    bed_files = list(predict_output.glob("*.bed"))
    if not bed_files:
        print("[WARN] predict.py 未生成任何 BED 文件")
        return None

    if len(bed_files) == 1:
        return bed_files[0]

    merged_bed = predict_output / "merged_predictions.bed"
    with open(merged_bed, "w") as out:
        for bf in bed_files:
            with open(bf, "r") as inp:
                out.write(inp.read())
    return merged_bed


def merge_bed_to_standard(input_bed: Path, output_bed: Path) -> int:
    output_bed.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_bed, "w") as fout:
        if not input_bed.exists():
            return 0
        with open(input_bed, "r") as fin:
            for line in fin:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                chrom, start, end = parts[0], parts[1], parts[2]
                fout.write(f"{chrom}\t{start}\t{end}\tmicrodna_{count}\t1.0\n")
                count += 1
    print(f"[INFO] 标准化 BED 完成, 共 {count} 条记录 -> {output_bed}")
    return count


def run_microdna_map(r1: Path, r2: Path, reference: Path, output_dir: Path,
                     model_path: Path = DEFAULT_MODEL_PATH, limit: str = DEFAULT_LIMIT,
                     threads: int = DEFAULT_THREADS, cnvkit_reference: Optional[Path] = None,
                     allowed_chroms: set = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "microdna_map.log"
    total_start = time.time()

    # 1. Bowtie2 比对
    ref_prefix = output_dir / "reference_index" / reference.stem
    ref_prefix.parent.mkdir(parents=True, exist_ok=True)
    build_bowtie2_index(reference, ref_prefix)

    bam_path = output_dir / "aligned.sorted.bam"
    align_and_sort_reads(r1, r2, ref_prefix, bam_path, threads)
    index_bam(bam_path, threads)

    # 2. 智能截取 CNVkit 基线模型
    cnvkit_dir = output_dir / "cnvkit"
    cnvkit_dir.mkdir(parents=True, exist_ok=True)
    
    if cnvkit_reference is None:
        cnvkit_reference = PROJECT_ROOT / "refs" / "cnvkit_ref.cnn"
        if not cnvkit_reference.exists():
            raise FileNotFoundError(f"未提供 CNVkit 参考文件: {cnvkit_reference}")
            
    # 【核心防御逻辑】：如果指定了部分染色体，则缩减基线以避免全局中位数漂移
    active_reference = cnvkit_reference
    if allowed_chroms is not None and len(allowed_chroms) > 0:
        active_reference = create_sub_reference(cnvkit_reference, allowed_chroms, cnvkit_dir)

    result_call = run_cnvkit(bam_path, active_reference, cnvkit_dir, threads)

    # 3. 提取候选区域序列 (合并 FASTA)
    candidates_fa = output_dir / "cnv_candidates.fa"
    n_extracted = extract_cnv_regions_fasta(result_call, reference, candidates_fa, allowed_chroms=allowed_chroms)

    final_bed = output_dir / "detected_microdna.bed"
    if n_extracted == 0:
        final_bed.write_text("")
        return final_bed

    # 4. 调用 predict.py --mode long
    pred_bed = run_microdna_predict(candidates_fa, model_path, limit, output_dir)
    if pred_bed is None:
        final_bed.write_text("")
        return final_bed

    # 5. 转为标准 BED
    merge_bed_to_standard(pred_bed, final_bed)

    elapsed = time.time() - total_start
    with open(log_file, "a") as log:
        log.write(f"Total time: {elapsed:.2f}s\n")
    print(f"[INFO] 全部流程完成, 总耗时 {elapsed:.2f} 秒")
    return final_bed


def main() -> None:
    parser = argparse.ArgumentParser(description="MicroDNA Map v2.0 检测封装")
    parser.add_argument("--r1", required=True, type=Path)
    parser.add_argument("--r2", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--limit", type=str, default=DEFAULT_LIMIT)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--cnvkit_reference", type=Path, default=None)
    parser.add_argument("--allowed_chroms", type=str, default=None, help="逗号分隔的合法染色体列表")
    args = parser.parse_args()

    for tool in ["bowtie2", "samtools", "cnvkit.py"]:
        if not check_tool(tool):
            print(f"[ERROR] 缺少工具 {tool}", file=sys.stderr)
            sys.exit(1)

    if not args.r1.exists() or not args.r2.exists() or not args.reference.exists():
        print("[ERROR] 输入文件不存在", file=sys.stderr)
        sys.exit(1)

    if not args.model_path.exists():
        print(f"[ERROR] 模型文件不存在: {args.model_path}", file=sys.stderr)
        sys.exit(1)

    allowed_chroms = set(c.strip() for c in args.allowed_chroms.split(",")) if args.allowed_chroms else None

    try:
        bed = run_microdna_map(args.r1, args.r2, args.reference, args.output_dir,
                               args.model_path, args.limit, args.threads,
                               args.cnvkit_reference, allowed_chroms)
        print(f"[INFO] 检测完成, BED: {bed}")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()