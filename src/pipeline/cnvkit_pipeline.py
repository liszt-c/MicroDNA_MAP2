"""
src/pipeline/cnvkit_pipeline.py - CNVkit 分析流程封装
"""
from pathlib import Path

import pandas as pd

from config import (HG19_FA, CNVKIT_REF_CNN, CNVKIT_TEMP_DIR,
                    BOWTIE2, BOWTIE2_BUILD, SAMTOOLS, CNVKIT)
from ..utils import run_command, setup_logger, ensure_faidx, extract_regions

logger = setup_logger('cnvkit_pipeline')

CALL_CNS_REQUIRED_COLS = ('chromosome', 'start', 'end', 'log2')


class CNVKitPipeline:
    def __init__(self, output_dir: Path = None, ref_genome: Path = None):
        self.output_dir = Path(output_dir) if output_dir else CNVKIT_TEMP_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ref_genome = Path(ref_genome) if ref_genome else HG19_FA
        if not self.ref_genome.exists():
            raise FileNotFoundError(
                f"Reference genome not found: {self.ref_genome}\n"
                f"Please place hg19.fa into {self.ref_genome.parent}/")
        self.index_prefix = self.ref_genome.parent / self.ref_genome.stem   # refs/hg19

    # ------------------------------------------------------------------ #
    def build_index(self):
        """构建 Bowtie2 索引 (若缺失)"""
        if list(self.ref_genome.parent.glob(f"{self.ref_genome.stem}.*.bt2")):
            logger.debug("Bowtie2 index already exists, skip building.")
            return
        logger.info(f"Building Bowtie2 index: {self.ref_genome} -> {self.index_prefix}")
        run_command([BOWTIE2_BUILD, "-f", str(self.ref_genome), str(self.index_prefix)])

    # ------------------------------------------------------------------ #
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
        ])

        logger.info(f"[{sample_name}] samtools sort -> {bam_file.name} ...")
        run_command([SAMTOOLS, "sort", f"-@{threads}", "-o", str(bam_file), str(sam_file)])
        try:
            sam_file.unlink()          # 中间 SAM 通常数 GB, 立即删除
        except OSError as e:
            logger.warning(f"Failed to remove {sam_file}: {e}")

        run_command([SAMTOOLS, "index", str(bam_file)])
        logger.info(f"[{sample_name}] BAM ready: {bam_file}")
        return bam_file

    # ------------------------------------------------------------------ #
    def call_cnv(self, bam_file: Path, sample_name: str, threads: int = 8) -> Path:
        """
        cnvkit.py batch (-m wgs)  -> {sample}.cnr / {sample}.cns
        cnvkit.py call            -> {sample}.call.cns
        """
        bam_file = Path(bam_file)
        batch_cmd = [CNVKIT, "batch", "-m", "wgs", "-p", str(threads),
                     "-d", str(self.output_dir)]
        if CNVKIT_REF_CNN.exists():
            batch_cmd += ["-r", str(CNVKIT_REF_CNN)]
            logger.info(f"[{sample_name}] Using CNVkit reference profile: {CNVKIT_REF_CNN.name}")
        else:
            logger.info(f"[{sample_name}] No reference .cnn found, CNVkit will build a flat reference.")
        batch_cmd.append(str(bam_file))

        logger.info(f"[{sample_name}] cnvkit batch ...")
        run_command(batch_cmd)

        cns_file = self.output_dir / f"{sample_name}.cns"
        if not cns_file.exists():
            raise FileNotFoundError(
                f"cnvkit batch did not produce {cns_file}. See stderr above.")

        # 关键: batch 不产生 .call.cns, 必须显式 call
        call_out = self.output_dir / f"{sample_name}.call.cns"
        logger.info(f"[{sample_name}] cnvkit call ...")
        run_command([CNVKIT, "call", str(cns_file), "-o", str(call_out)])

        if not call_out.exists():
            raise FileNotFoundError(f"cnvkit call did not produce {call_out}")
        return call_out

    def align_and_call(self, fastq1, fastq2, sample_name: str, threads: int = 8) -> Path:
        """完整流程: 比对 + CNV 调用, 返回 .call.cns 路径"""
        bam = self.align(fastq1, fastq2, sample_name, threads=threads)
        return self.call_cnv(bam, sample_name, threads=threads)

    # ------------------------------------------------------------------ #
    def extract_candidates(self, call_cns: Path,
                           min_log2: float = None,
                           min_size: int = 0,
                           max_size: int = None,
                           sample_name: str = None) -> list:
        """
        从 .call.cns 读取候选区域

        :param min_log2: 仅保留 log2 >= 该值的区域; None = 不过滤 (与原版行为一致)
        :param min_size / max_size: 区域长度过滤 (bp)
        :return: [{'name','chrom','start','end','log2','size'}, ...]  start/end 为 1-based 闭区间
        """
        call_cns = Path(call_cns)
        if not call_cns.exists():
            raise FileNotFoundError(f"CNV call file not found: {call_cns}")

        df = pd.read_csv(call_cns, sep='\t', comment=None, dtype=str)
        missing = [c for c in CALL_CNS_REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{call_cns.name} lacks required columns {missing}; "
                             f"found {list(df.columns)}")

        df['start'] = pd.to_numeric(df['start'], errors='coerce')
        df['end'] = pd.to_numeric(df['end'], errors='coerce')
        df['log2'] = pd.to_numeric(df['log2'], errors='coerce')
        df = df.dropna(subset=['start', 'end'])
        df['size'] = df['end'] - df['start']

        n0 = len(df)
        if min_log2 is not None:
            df = df[df['log2'] >= float(min_log2)]
        if min_size:
            df = df[df['size'] >= int(min_size)]
        if max_size:
            df = df[df['size'] <= int(max_size)]
        logger.info(f"[{call_cns.name}] segments: {n0} total -> {len(df)} after filtering "
                    f"(min_log2={min_log2}, min_size={min_size}, max_size={max_size})")

        prefix = f"{sample_name}_" if sample_name else ""
        candidates = []
        for i, (_, row) in enumerate(df.iterrows()):
            chrom = str(row['chromosome']).strip()
            start, end = int(row['start']), int(row['end'])
            candidates.append({
                'name': f"{prefix}cnv{i}",
                'chrom': chrom,
                'start': start,
                'end': end,
                'log2': float(row['log2']) if pd.notna(row['log2']) else float('nan'),
                'size': end - start,
            })
        return candidates

    # ------------------------------------------------------------------ #
    def extract_sequences(self, candidates: list, output_fa: Path) -> int:
        """
        批量提取候选区域序列, 写入单个合并 FASTA
        Header: >{name}|{chrom}:{start}-{end}
        :return: 成功写入的序列数
        """
        output_fa = Path(output_fa)
        if not candidates:
            logger.warning("No candidates to extract; writing empty FASTA.")
            output_fa.write_text("")
            return 0

        ensure_faidx(self.ref_genome, SAMTOOLS)

        # 去重 (相同区域只提取一次)
        uniq, seen = [], set()
        for c in candidates:
            key = (c['chrom'], c['start'], c['end'])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)

        regions = [(c['chrom'], c['start'], c['end']) for c in uniq]
        seq_map, skipped = extract_regions(self.ref_genome, regions, SAMTOOLS)
        for region, reason in skipped:
            logger.warning(f"Skipped {region[0]}:{region[1]}-{region[2]} -> {reason}")

        written = 0
        with open(output_fa, 'w', encoding='utf-8') as f:
            for c in uniq:
                seq = seq_map.get((c['chrom'], c['start'], c['end']))
                if not seq:
                    continue
                seq = seq.upper()
                f.write(f">{c['name']}|{c['chrom']}:{c['start']}-{c['end']}\n")
                for i in range(0, len(seq), 70):
                    f.write(seq[i:i + 70] + "\n")
                written += 1

        logger.info(f"Wrote {written}/{len(candidates)} sequences to {output_fa}")
        return written

    # ------------------------------------------------------------------ #
    @staticmethod
    def candidates_to_bed_rows(candidates: list) -> list:
        """
        候选区域 -> BED 行 (0-based half-open)
        .cns 的 start 为 1-based, 故 BED start = start - 1
        """
        rows = []
        for c in candidates:
            rows.append((c['chrom'], max(0, int(c['start']) - 1), int(c['end'])))
        return rows

    # ------------------------------------------------------------------ #
    def cleanup_sample(self, sample_name: str, keep_bam: bool = False):
        """删除单个样本的中间文件"""
        patterns = [f"{sample_name}.sam", f"{sample_name}.cnr", f"{sample_name}.cns",
                    f"{sample_name}.call.cns", f"{sample_name}_candidates.fa",
                    f"{sample_name}.antitargetcoverage.cnn", f"{sample_name}.targetcoverage.cnn"]
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