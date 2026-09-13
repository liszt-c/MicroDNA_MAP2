#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_microdna.py
====================

从参考基因组随机选取位置，或从已有的真实 microDNA 序列数据库
（FASTA 文件）中读取序列，生成模拟 microDNA 的 truth BED、FASTA
和 junction 信息。

支持两种模式：
  1. 随机模式（默认）：从参考基因组随机选取区域作为 microDNA body。
  2. 真实模式：指定 --input_fasta_dir 指向包含 .fa 文件的目录或特定的 .fa 文件，
     从这些文件中读取真实 microDNA 序列作为 body。
     *新增限制*: 若提供 --background_chroms，则仅使用来自指定背景染色体的真实序列。

输出：
  - microdna_truth.bed
  - microdna_sequences.fa
  - microdna_circular_templates.fa
  - junction_info.tsv
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pysam

# 项目路径定位
BENCHMARK_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "results" / "truth"
DEFAULT_CONFIG = BENCHMARK_DIR / "config.yaml"
DEFAULT_FLANK = 50
DEFAULT_NUM_SITES = 1000
DEFAULT_MIN_LEN = 200
DEFAULT_MAX_LEN = 800
DEFAULT_SEED = 42
N_MAX_FRACTION = 0.10


def load_yaml_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        print("[WARN] PyYAML 未安装，将使用默认参数。", file=sys.stderr)
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            cfg = yaml.safe_load(f)
            return cfg if isinstance(cfg, dict) else {}
        except yaml.YAMLError as e:
            print(f"[WARN] config.yaml 解析失败: {e}", file=sys.stderr)
            return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simulated microDNA.")
    parser.add_argument("--num_sites", type=int, default=None,
                        help="Number of microDNA sites to generate (default: from config or 1000)")
    parser.add_argument("--min_len", type=int, default=None,
                        help="Minimum microDNA body length (default: 200)")
    parser.add_argument("--max_len", type=int, default=None,
                        help="Maximum microDNA body length (default: 800)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: 42)")
    parser.add_argument("--genome_path", type=str, default=None,
                        help="Path to hg19.fa reference genome")
    parser.add_argument("--gap_bed", type=str, default=None,
                        help="Optional gap BED file (e.g., UCSC gap.txt converted to BED). "
                             "If not provided, N-content filtering is used.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: benchmark/results/truth)")
    parser.add_argument("--flank", type=int, default=None,
                        help="Length of upstream/downstream flanking sequence (default: 50)")
    parser.add_argument("--input_fasta_dir", type=str, default=None,
                        help="Directory or File containing real microDNA FASTA sequences. "
                             "If provided, will use these sequences as microDNA bodies.")
    parser.add_argument("--background_chroms", type=str, default=None,
                        help="Comma-separated list of allowed chromosomes (e.g. chr21,chr22). "
                             "Crucial for preventing WGS alignment to non-background chromosomes.")
    return parser.parse_args()


def weighted_chromosome_choice(chroms: List[str], lengths: List[int]) -> Tuple[str, int]:
    total = sum(lengths)
    r = random.uniform(0, total)
    for chrom, length in zip(chroms, lengths):
        if r < length:
            return chrom, length
        r -= length
    return chroms[-1], lengths[-1]


def n_fraction(seq: str) -> float:
    if len(seq) == 0:
        return 1.0
    return (seq.count("N") + seq.count("n")) / len(seq)


def load_gap_regions(gap_bed: Optional[str]) -> Dict[str, List[Tuple[int, int]]]:
    gaps: Dict[str, List[Tuple[int, int]]] = {}
    if not gap_bed:
        return gaps
    gap_path = Path(gap_bed)
    if not gap_path.exists():
        print(f"[WARN] gap BED 文件不存在: {gap_path}", file=sys.stderr)
        return gaps
    with open(gap_path, "r") as f:
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
            gaps.setdefault(chrom, []).append((start, end))
    return gaps


def overlaps_gap(chrom: str, start: int, end: int, gap_dict: Dict[str, List[Tuple[int, int]]]) -> bool:
    for g_start, g_end in gap_dict.get(chrom, []):
        if start < g_end and end > g_start:
            return True
    return False


def write_fasta_record(f, header: str, sequence: str, line_width: int = 80) -> None:
    f.write(f">{header}\n")
    for i in range(0, len(sequence), line_width):
        f.write(sequence[i:i + line_width] + "\n")


def parse_fasta_header(header: str) -> Optional[Tuple[str, int, int]]:
    """
    极度鲁棒的 FASTA 头部解析器。
    完美处理浮点数问题，例如: >chr13:52514242.0-52514642.0
    """
    header = header.strip()
    if header.startswith(">"):
        header = header[1:]
    
    # 忽略 | 分隔符后面的内容，直接找 chrom:start-end
    for seg in header.split("|"):
        # 匹配: 染色体名 + 冒号 + 数字(允许跟.0) + 连字符 + 数字(允许跟.0)
        match = re.search(r"([a-zA-Z0-9_]+):(\d+)(?:\.\d+)?-(\d+)(?:\.\d+)?", seg)
        if match:
            chrom = match.group(1).strip()
            start = int(match.group(2))
            end = int(match.group(3))
            
            if start >= 0 and end > start:
                return chrom, start, end
    return None


def read_real_microdna_fasta(input_path: Path, allowed_chroms: Optional[set] = None) -> List[Tuple[str, int, int, str]]:
    records = []
    
    # 修复：同时支持传入文件目录和单个文件
    if input_path.is_dir():
        fa_files = list(input_path.glob("*.fa")) + list(input_path.glob("*.fasta"))
    else:
        fa_files = [input_path]
        
    for fa_file in fa_files:
        with open(fa_file, "r", encoding="utf-8") as f:
            current_header = None
            current_seq = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_header is not None:
                        seq = "".join(current_seq).upper()
                        loc = parse_fasta_header(current_header)
                        if loc is not None:
                            chrom, start, end = loc
                            if allowed_chroms is None or chrom in allowed_chroms:
                                records.append((chrom, start, end, seq))
                    current_header = line
                    current_seq = []
                else:
                    if current_header is None:
                        continue
                    current_seq.append(line)
            if current_header is not None:
                seq = "".join(current_seq).upper()
                loc = parse_fasta_header(current_header)
                if loc is not None:
                    chrom, start, end = loc
                    if allowed_chroms is None or chrom in allowed_chroms:
                        records.append((chrom, start, end, seq))
    return records


def generate_from_real_fasta(
    fasta: pysam.FastaFile,
    records: List[Tuple[str, int, int, str]],
    num_sites: int,
    flank: int,
    output_dir: Path,
    seed: int,
    gap_dict: Dict[str, List[Tuple[int, int]]],
) -> None:
    random.seed(seed)

    if len(records) > num_sites:
        selected = random.sample(records, num_sites)
    else:
        selected = records
        print(f"[WARN] 符合条件的真实 microDNA 记录数 ({len(records)}) 少于目标数量 ({num_sites})，将使用全部记录。")

    truth_bed = output_dir / "microdna_truth.bed"
    truth_fasta = output_dir / "microdna_sequences.fa"
    circular_fasta = output_dir / "microdna_circular_templates.fa"
    junction_tsv = output_dir / "junction_info.tsv"

    with open(truth_bed, "w") as bed_f, \
         open(truth_fasta, "w") as lin_fa, \
         open(circular_fasta, "w") as cir_fa, \
         open(junction_tsv, "w") as junc_f:

        junc_f.write("id\tchr\tjunction_pos\tbody_start\tbody_end\n")

        for idx, (chrom, orig_start, orig_end, body_seq) in enumerate(selected):
            body_len = len(body_seq)
            start = orig_start
            end = start + body_len

            if body_len <= 0:
                continue

            if gap_dict and overlaps_gap(chrom, start, end, gap_dict):
                continue

            try:
                upstream = fasta.fetch(chrom, max(0, start - flank), start).upper()
            except Exception:
                upstream = ""
            try:
                downstream = fasta.fetch(chrom, end, min(end + flank, fasta.get_reference_length(chrom))).upper()
            except Exception:
                downstream = ""

            if len(upstream) < flank:
                upstream = "N" * (flank - len(upstream)) + upstream
            if len(downstream) < flank:
                downstream = downstream + "N" * (flank - len(downstream))

            linear_seq = upstream + body_seq + downstream
            circular_seq = body_seq + downstream + upstream

            micro_id = f"microdna_{idx}"

            bed_f.write(f"{chrom}\t{start}\t{end}\t{body_len}\t{micro_id}\n")
            write_fasta_record(lin_fa, f"{micro_id} chrom={chrom} start={start} end={end} length={body_len}", linear_seq)
            write_fasta_record(cir_fa, f"{micro_id} chrom={chrom} start={start} end={end} length={body_len}", circular_seq)
            junc_f.write(f"{micro_id}\t{chrom}\t{start}\t{start}\t{end}\n")

            if (idx + 1) % 100 == 0:
                print(f"[INFO] 已处理 {idx + 1}/{len(selected)} 条真实序列")


def generate_random(
    fasta: pysam.FastaFile,
    chroms: List[str],
    lengths: List[int],
    num_sites: int,
    min_len: int,
    max_len: int,
    flank: int,
    seed: int,
    gap_dict: Dict[str, List[Tuple[int, int]]],
    output_dir: Path,
) -> None:
    random.seed(seed)

    truth_bed = output_dir / "microdna_truth.bed"
    truth_fasta = output_dir / "microdna_sequences.fa"
    circular_fasta = output_dir / "microdna_circular_templates.fa"
    junction_tsv = output_dir / "junction_info.tsv"

    used_sites = set()
    with open(truth_bed, "w") as bed_f, \
         open(truth_fasta, "w") as lin_fa, \
         open(circular_fasta, "w") as cir_fa, \
         open(junction_tsv, "w") as junc_f:

        junc_f.write("id\tchr\tjunction_pos\tbody_start\tbody_end\n")
        site_idx = 0
        attempts_per_site = 5000

        while site_idx < num_sites:
            success = False
            for _ in range(attempts_per_site):
                chrom, chrom_len = weighted_chromosome_choice(chroms, lengths)
                body_len = random.randint(min_len, max_len)
                max_start = chrom_len - flank - body_len
                if max_start < flank:
                    continue
                start = random.randint(flank, max_start)
                end = start + body_len
                if (chrom, start, end) in used_sites:
                    continue
                if gap_dict and overlaps_gap(chrom, start - flank, end + flank, gap_dict):
                    continue

                try:
                    upstream = fasta.fetch(chrom, start - flank, start).upper()
                    body = fasta.fetch(chrom, start, end).upper()
                    downstream = fasta.fetch(chrom, end, end + flank).upper()
                except Exception:
                    continue

                if n_fraction(upstream) > N_MAX_FRACTION or \
                   n_fraction(body) > N_MAX_FRACTION or \
                   n_fraction(downstream) > N_MAX_FRACTION:
                    continue

                used_sites.add((chrom, start, end))
                linear_seq = upstream + body + downstream
                circular_seq = body + downstream + upstream
                micro_id = f"microdna_{site_idx}"

                bed_f.write(f"{chrom}\t{start}\t{end}\t{body_len}\t{micro_id}\n")
                write_fasta_record(lin_fa, f"{micro_id} chrom={chrom} start={start} end={end} length={body_len}", linear_seq)
                write_fasta_record(cir_fa, f"{micro_id} chrom={chrom} start={start} end={end} length={body_len}", circular_seq)
                junc_f.write(f"{micro_id}\t{chrom}\t{start}\t{start}\t{end}\n")

                if (site_idx + 1) % 100 == 0:
                    print(f"[INFO] 已生成 {site_idx + 1}/{num_sites}")
                site_idx += 1
                success = True
                break

            if not success:
                print(f"[ERROR] 位点 {site_idx} 生成失败。", file=sys.stderr)
                sys.exit(1)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(DEFAULT_CONFIG)

    genome_cfg = config.get("genome", {})
    sim_cfg = config.get("simulation", {})
    experiment_cfg = config.get("experiment", {})

    genome_path = args.genome_path or genome_cfg.get("reference")
    gap_bed = args.gap_bed or genome_cfg.get("gap_bed")
    num_sites = args.num_sites if args.num_sites is not None else sim_cfg.get("num_sites", DEFAULT_NUM_SITES)
    min_len = args.min_len if args.min_len is not None else sim_cfg.get("min_length", DEFAULT_MIN_LEN)
    max_len = args.max_len if args.max_len is not None else sim_cfg.get("max_length", DEFAULT_MAX_LEN)
    seed = args.seed if args.seed is not None else experiment_cfg.get("seed", DEFAULT_SEED)
    flank = args.flank if args.flank is not None else sim_cfg.get("flank", DEFAULT_FLANK)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if "output_dir" in experiment_cfg:
            output_dir = Path(experiment_cfg["output_dir"]) / "truth"
        else:
            output_dir = DEFAULT_OUTPUT_DIR

    if not genome_path:
        print("[ERROR] 未提供参考基因组路径。", file=sys.stderr)
        sys.exit(1)
    genome_path = Path(genome_path)
    if not genome_path.exists():
        print(f"[ERROR] 参考基因组文件不存在: {genome_path}", file=sys.stderr)
        sys.exit(1)

    if min_len <= 0 or max_len < min_len:
        print("[ERROR] 长度参数无效。", file=sys.stderr)
        sys.exit(1)
    if flank < 0:
        print("[ERROR] flank 不能为负数。", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        fasta = pysam.FastaFile(str(genome_path))
    except Exception as e:
        print(f"[ERROR] 无法打开参考基因组: {e}", file=sys.stderr)
        sys.exit(1)

    gap_dict = load_gap_regions(gap_bed)

    # 确定允许的染色体列表
    allowed_chroms = None
    if args.background_chroms:
        allowed_chroms = set(c.strip() for c in args.background_chroms.split(",") if c.strip())
        print(f"[INFO] 限制 microDNA 提取来源为以下染色体: {allowed_chroms}")

    if args.input_fasta_dir:
        input_path = Path(args.input_fasta_dir)
        if not input_path.exists():
            print(f"[ERROR] 指定的真实 FASTA 输入不存在: {input_path}", file=sys.stderr)
            sys.exit(1)
        print(f"[INFO] 从真实 microDNA FASTA 读取序列: {input_path}")
        records = read_real_microdna_fasta(input_path, allowed_chroms)
        print(f"[INFO] 读取到 {len(records)} 条可用的真实序列")
        if len(records) == 0:
            print("[ERROR] 未读取到有效序列。如果使用了 quick 模式，请确保原始数据中包含背景染色体的序列。", file=sys.stderr)
            sys.exit(1)

        generate_from_real_fasta(
            fasta=fasta,
            records=records,
            num_sites=num_sites,
            flank=flank,
            output_dir=output_dir,
            seed=seed,
            gap_dict=gap_dict,
        )
    else:
        print(f"[INFO] 随机模式：从参考基因组选取 microDNA 位点")
        # 如果提供了 background_chroms，则从中抽取；否则用主染色体
        if allowed_chroms:
            chroms = [c for c in fasta.references if c in allowed_chroms]
        else:
            default_chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
            chroms = [c for c in fasta.references if c in default_chroms]
            
        if not chroms:
            print("[ERROR] 参考基因组中未找到指定的染色体。", file=sys.stderr)
            sys.exit(1)
        lengths = [fasta.get_reference_length(c) for c in chroms]

        generate_random(
            fasta=fasta,
            chroms=chroms,
            lengths=lengths,
            num_sites=num_sites,
            min_len=min_len,
            max_len=max_len,
            flank=flank,
            seed=seed,
            gap_dict=gap_dict,
            output_dir=output_dir,
        )

    fasta.close()
    print("[INFO] 模拟 microDNA 生成完成。")


if __name__ == "__main__":
    main()