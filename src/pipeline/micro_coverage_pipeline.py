#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/pipeline/micro_coverage_pipeline.py - 微尺度局部覆盖度分析流程 (独立模块)
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pysam

from config import (
    BOWTIE2,
    BOWTIE2_BUILD,
    DEFAULT_MICRO_CLUSTER_MAX_LEN,
    DEFAULT_MICRO_FOLD_CHANGE,
    DEFAULT_MICRO_MAX_LEN,
    DEFAULT_MICRO_MIN_LEN,
    DEFAULT_MICRO_WINDOW_SIZE,
    HG19_FA,
    MICRO_COVERAGE_TEMP_DIR,
    SAMTOOLS,
)
from ..utils import ensure_bam_index, ensure_faidx, extract_regions, run_command, setup_logger

logger = setup_logger("micro_coverage_pipeline")

CALL_CNS_REQUIRED_COLS = ("chromosome", "start", "end", "log2")


class MicroCoveragePipeline:
    """微尺度局部覆盖度分析类"""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        ref_genome: Optional[Path] = None,
        window_size: int = DEFAULT_MICRO_WINDOW_SIZE,
        fold_change: float = DEFAULT_MICRO_FOLD_CHANGE,
        min_region_len: int = DEFAULT_MICRO_MIN_LEN,
        max_region_len: int = DEFAULT_MICRO_MAX_LEN,
        cluster_max_len: int = DEFAULT_MICRO_CLUSTER_MAX_LEN,
        flank_bins: int = 1,
    ):
        self.output_dir = Path(output_dir) if output_dir else MICRO_COVERAGE_TEMP_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ref_genome = Path(ref_genome) if ref_genome else HG19_FA
        if not self.ref_genome.exists():
            raise FileNotFoundError(
                f"Reference genome not found: {self.ref_genome}\n"
                f"Please place reference genome into {self.ref_genome.parent}/"
            )
        self.index_prefix = self.ref_genome.parent / self.ref_genome.stem

        self.window_size = int(window_size)
        self.fold_change = float(fold_change)
        self.min_region_len = int(min_region_len)
        self.max_region_len = int(max_region_len)
        self.cluster_max_len = int(cluster_max_len)
        self.flank_bins = int(flank_bins)

    def build_index(self) -> None:
        """构建 Bowtie2 索引 (若缺失)"""
        idx_exts = [".1.bt2", ".2.bt2", ".3.bt2", ".4.bt2", ".rev.1.bt2", ".rev.2.bt2"]
        if all(Path(str(self.index_prefix) + ext).exists() for ext in idx_exts):
            logger.debug("Bowtie2 index already exists, skip building.")
            return
        logger.info(f"Building Bowtie2 index: {self.ref_genome} -> {self.index_prefix}")
        run_command([BOWTIE2_BUILD, "-f", str(self.ref_genome), str(self.index_prefix)])

    def align(self, fastq1: Path, fastq2: Path, sample_name: str, threads: int = 8) -> Path:
        """FASTQ -> sorted & indexed BAM"""
        fastq1, fastq2 = Path(fastq1), Path(fastq2)
        for fq in (fastq1, fastq2):
            if not fq.exists():
                raise FileNotFoundError(f"FASTQ not found: {fq}")

        self.build_index()

        sam_file = self.output_dir / f"{sample_name}.sam"
        bam_file = self.output_dir / f"{sample_name}.bam"

        logger.info(f"[{sample_name}] bowtie2 aligning ({threads} threads) ...")
        run_command([
            BOWTIE2, "-p", str(threads),
            "-x", str(self.index_prefix),
            "-1", str(fastq1), "-2", str(fastq2),
            "-S", str(sam_file),
            "--no-unal",
        ])

        logger.info(f"[{sample_name}] samtools sort -> {bam_file.name} ...")
        run_command([SAMTOOLS, "sort", f"-@{threads}", "-o", str(bam_file), str(sam_file)])
        sam_file.unlink(missing_ok=True)

        run_command([SAMTOOLS, "index", str(bam_file)])
        logger.info(f"[{sample_name}] BAM ready: {bam_file}")
        return bam_file

    def _merge_candidate_regions(self, regions: List[Tuple[str, int, int, float]]) -> List[Tuple[str, int, int, float]]:
        """合并扩展后发生重叠或相邻的候选区间"""
        if not regions:
            return []

        # 按染色体、起始坐标排序
        sorted_regions = sorted(regions, key=lambda x: (x[0], x[1], x[2]))
        merged = []

        for chrom, s, e, l2 in sorted_regions:
            if not merged:
                merged.append([chrom, s, e, l2])
                continue

            last = merged[-1]
            if chrom == last[0] and s <= last[2]:
                # 产生重叠，更新终止位置，并保留更高的 log2 富集度
                last[2] = max(last[2], e)
                last[3] = max(last[3], l2)
            else:
                merged.append([chrom, s, e, l2])

        return [(m[0], m[1], m[2], m[3]) for m in merged]

    def call_cnv(
        self,
        bam_file: Path,
        sample_name: str,
        threads: int = 8,
        allowed_chroms: Optional[set] = None,
    ) -> Path:
        """
        微尺度滑动窗口局部深度扫描，产出与 CNVkit .call.cns 兼容的 TSV 格式。
        输出格式: chromosome\tstart\tend\tlog2 (1-based 闭区间)
        """
        bam_file = Path(bam_file)
        if not bam_file.exists():
            raise FileNotFoundError(f"BAM file not found: {bam_file}")

        ensure_bam_index(bam_file, SAMTOOLS)

        call_out = self.output_dir / f"{sample_name}.call.cns"
        logger.info(
            f"[{sample_name}] Scanning micro-scale coverage "
            f"(window={self.window_size}bp, fold_change={self.fold_change}, "
            f"flank_bins={self.flank_bins}, cluster_max={self.cluster_max_len}bp)..."
        )

        raw_candidates: List[Tuple[str, int, int, float]] = []

        with pysam.AlignmentFile(str(bam_file), "rb") as bam:
            references = list(bam.references)
            lengths = list(bam.lengths)

            for chrom, chrom_len in zip(references, lengths):
                if allowed_chroms is not None and chrom not in allowed_chroms:
                    continue
                if chrom_len < self.window_size:
                    continue

                num_bins = int(math.ceil(chrom_len / self.window_size))
                counts = np.zeros(num_bins, dtype=np.uint32)

                try:
                    for read in bam.fetch(chrom):
                        if (
                            read.is_unmapped
                            or read.is_duplicate
                            or read.is_secondary
                            or read.mapping_quality < 20
                            or read.reference_start is None
                            or read.reference_end is None
                        ):
                            continue

                        read_span = read.reference_end - read.reference_start
                        if read_span <= 0 or read_span > 1000:
                            continue

                        s_bin = max(0, min(read.reference_start // self.window_size, num_bins - 1))
                        e_bin = max(0, min((read.reference_end - 1) // self.window_size, num_bins - 1))
                        counts[s_bin : e_bin + 1] += 1
                except Exception as e:
                    logger.warning(f"Error fetching reads on {chrom}: {e}")
                    continue

                non_zero = counts[counts > 0]
                if len(non_zero) == 0:
                    continue

                baseline = float(np.median(non_zero))
                baseline = max(baseline, 1.0)
                cutoff = baseline * self.fold_change
                min_reads_diff = max(3, int(round(0.1 * baseline)))

                enriched_mask = (counts >= cutoff) & ((counts.astype(np.float32) - baseline) >= min_reads_diff)
                enriched_indices = np.where(enriched_mask)[0]
                if len(enriched_indices) == 0:
                    continue

                # 聚类缝合
                cur_start_bin = enriched_indices[0]
                cur_end_bin = enriched_indices[0] + 1

                for idx in enriched_indices[1:]:
                    if idx <= cur_end_bin + 1:
                        new_span = (idx + 1 - cur_start_bin) * self.window_size
                        if new_span <= self.cluster_max_len:
                            cur_end_bin = idx + 1
                            continue

                    # 扩展前后 flank_bins
                    ext_s_bin = max(0, cur_start_bin - self.flank_bins)
                    ext_e_bin = min(num_bins, cur_end_bin + self.flank_bins)

                    reg_start = ext_s_bin * self.window_size
                    reg_end = min(ext_e_bin * self.window_size, chrom_len)
                    span = reg_end - reg_start

                    if span >= self.min_region_len:
                        local_mean = float(np.mean(counts[cur_start_bin:cur_end_bin]))
                        log2_val = math.log2((local_mean + 1e-4) / baseline)
                        raw_candidates.append((chrom, reg_start + 1, reg_end, log2_val))

                    cur_start_bin = idx
                    cur_end_bin = idx + 1

                # 封闭最后一个区间
                ext_s_bin = max(0, cur_start_bin - self.flank_bins)
                ext_e_bin = min(num_bins, cur_end_bin + self.flank_bins)

                reg_start = ext_s_bin * self.window_size
                reg_end = min(ext_e_bin * self.window_size, chrom_len)
                span = reg_end - reg_start

                if span >= self.min_region_len:
                    local_mean = float(np.mean(counts[cur_start_bin:cur_end_bin]))
                    log2_val = math.log2((local_mean + 1e-4) / baseline)
                    raw_candidates.append((chrom, reg_start + 1, reg_end, log2_val))

        # 执行二次区间合并，剔除重叠冗余
        final_candidates = self._merge_candidate_regions(raw_candidates)

        with open(call_out, "w", encoding="utf-8") as f:
            f.write("chromosome\tstart\tend\tlog2\n")
            for c, s, e, l2 in final_candidates:
                f.write(f"{c}\t{s}\t{e}\t{l2:.4f}\n")

        logger.info(
            f"[{sample_name}] Micro-coverage profiling finished. "
            f"Identified {len(final_candidates)} candidate segment(s) after flank expansion -> {call_out.name}"
        )
        return call_out

    def align_and_call(
        self,
        fastq1: Path,
        fastq2: Path,
        sample_name: str,
        threads: int = 8,
        allowed_chroms: Optional[set] = None,
    ) -> Path:
        """端到端完整流程: 比对 + 快速微覆盖度变异调用"""
        bam = self.align(fastq1, fastq2, sample_name, threads=threads)
        return self.call_cnv(bam, sample_name, threads=threads, allowed_chroms=allowed_chroms)

    def extract_candidates(
        self,
        call_cns: Path,
        min_log2: Optional[float] = None,
        min_size: int = 0,
        max_size: Optional[int] = None,
        sample_name: Optional[str] = None,
    ) -> List[Dict]:
        """从 .call.cns 读取候选区域字典列表"""
        call_cns = Path(call_cns)
        if not call_cns.exists():
            raise FileNotFoundError(f"Call file not found: {call_cns}")

        df = pd.read_csv(call_cns, sep="\t", comment=None, dtype=str)
        missing = [c for c in CALL_CNS_REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{call_cns.name} lacks required columns {missing}")

        df["start"] = pd.to_numeric(df["start"], errors="coerce")
        df["end"] = pd.to_numeric(df["end"], errors="coerce")
        df["log2"] = pd.to_numeric(df["log2"], errors="coerce")
        df = df.dropna(subset=["start", "end"])
        df["size"] = df["end"] - df["start"]

        n0 = len(df)
        if min_log2 is not None:
            df = df[df["log2"] >= float(min_log2)]
        if min_size:
            df = df[df["size"] >= int(min_size)]
        if max_size:
            df = df[df["size"] <= int(max_size)]

        logger.info(
            f"[{call_cns.name}] Segments: {n0} total -> {len(df)} after filtering "
            f"(min_log2={min_log2}, min_size={min_size}, max_size={max_size})"
        )

        prefix = f"{sample_name}_" if sample_name else ""
        candidates = []
        for i, (_, row) in enumerate(df.iterrows()):
            chrom = str(row["chromosome"]).strip()
            start, end = int(row["start"]), int(row["end"])
            candidates.append({
                "name": f"{prefix}mcv{i}",
                "chrom": chrom,
                "start": start,
                "end": end,
                "log2": float(row["log2"]) if pd.notna(row["log2"]) else float("nan"),
                "size": end - start,
            })
        return candidates

    def extract_sequences(self, candidates: List[Dict], output_fa: Path) -> int:
        """批量提取候选区域序列写入合并 FASTA (Header: >name|chrom:start-end)"""
        output_fa = Path(output_fa)
        if not candidates:
            logger.warning("No candidates to extract; writing empty FASTA.")
            output_fa.write_text("")
            return 0

        ensure_faidx(self.ref_genome, SAMTOOLS)

        uniq, seen = [], set()
        for c in candidates:
            key = (c["chrom"], c["start"], c["end"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)

        regions = [(c["chrom"], c["start"], c["end"]) for c in uniq]
        seq_map, skipped = extract_regions(self.ref_genome, regions, SAMTOOLS)
        for region, reason in skipped:
            logger.warning(f"Skipped {region[0]}:{region[1]}-{region[2]} -> {reason}")

        written = 0
        with open(output_fa, "w", encoding="utf-8") as f:
            for c in uniq:
                seq = seq_map.get((c["chrom"], c["start"], c["end"]))
                if not seq:
                    continue
                seq = seq.upper()
                f.write(f">{c['name']}|{c['chrom']}:{c['start']}-{c['end']}\n")
                for i in range(0, len(seq), 70):
                    f.write(seq[i : i + 70] + "\n")
                written += 1

        logger.info(f"Wrote {written}/{len(candidates)} sequences to {output_fa.name}")
        return written

    @staticmethod
    def candidates_to_bed_rows(candidates: List[Dict]) -> List[Tuple[str, int, int]]:
        """候选区域 -> BED 行 (0-based half-open)"""
        rows = []
        for c in candidates:
            rows.append((c["chrom"], max(0, int(c["start"]) - 1), int(c["end"])))
        return rows

    def cleanup_sample(self, sample_name: str, keep_bam: bool = False) -> int:
        """删除单个样本的中间文件"""
        patterns = [
            f"{sample_name}.sam",
            f"{sample_name}.call.cns",
            f"{sample_name}_candidates.fa",
        ]
        if not keep_bam:
            patterns += [f"{sample_name}.bam", f"{sample_name}.bam.bai"]

        removed = 0
        for pat in patterns:
            for p in self.output_dir.glob(pat):
                try:
                    p.unlink()
                    removed += 1
                except OSError as e:
                    logger.warning(f"Failed to remove {p}: {e}")
        logger.info(f"[{sample_name}] Removed {removed} intermediate files.")
        return removed