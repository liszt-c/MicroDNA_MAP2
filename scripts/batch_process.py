"""
scripts/batch_process.py - 从 FASTQ 到 MicroDNA 鉴定的全流程

等价于原版 MicroDNA_Map_batch.py, 但修复了:
  * scripts/ 不是包, 改用 sys.path + 直接导入 (解决 ImportError)
  * CNV 片段用 predict_long (滑窗) 而非 predict_short (截断) (修复语义错误)
  * BED 坐标从 .call.cns 的 1-based 转为 0-based (修复坐标偏移)
  * 不再产生零散 cnvkit_N.fa, 使用单个合并 FASTA
  * 清理逻辑可配置 (--cleanup)
"""
import argparse
import os
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中 (scripts/ 不是 Python 包)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from config import (RAW_DATA_DIR, PREDICTIONS_DIR, MODEL_DIR, CNVKIT_TEMP_DIR)
from src.pipeline.cnvkit_pipeline import CNVKitPipeline
from src.utils import setup_logger, parse_fasta_file

# 延迟导入 predict 模块中的函数 (避免顶层 import 时序问题)
_predict_module = None

def _get_predict_funcs():
    global _predict_module
    if _predict_module is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "predict", str(Path(__file__).resolve().parent / "predict.py"))
        _predict_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_predict_module)
    return _predict_module.load_model, _predict_module.predict_long_single


logger = setup_logger('batch_process')


def find_fastq_pairs(directory: Path):
    """查找配对的 fastq 文件 (支持 _1/_2, R1/R2 命名)"""
    pairs = []
    seen_bases = set()

    for pattern in ["*_1.fastq", "*_1.fq", "*_R1.fastq", "*_R1.fq"]:
        for r1 in sorted(directory.glob(pattern)):
            name = r1.name
            for suffix in ["_1.fastq", "_1.fq", "_R1.fastq", "_R1.fq"]:
                if name.endswith(suffix):
                    base = name[:-len(suffix)]
                    break
            else:
                continue

            if base in seen_bases:
                continue
            seen_bases.add(base)

            # 尝试多种 R2 命名
            r2_candidates = [
                directory / f"{base}_2.fastq",
                directory / f"{base}_2.fq",
                directory / f"{base}_R2.fastq",
                directory / f"{base}_R2.fq",
            ]
            r2 = next((c for c in r2_candidates if c.exists()), None)
            if r2:
                pairs.append((base, r1, r2))
            else:
                logger.warning(f"Missing R2 pair for {r1.name}")

    return pairs


def main():
    p = argparse.ArgumentParser(description="Batch process FASTQ -> MicroDNA",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input-dir', default=str(RAW_DATA_DIR), help="FASTQ 所在目录")
    p.add_argument('--threads', type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument('--limit', type=float, default=0.99, help="预测阈值")
    p.add_argument('--min-log2', type=float, default=None,
                   help="CNV log2 过滤阈值 (None=不过滤)")
    p.add_argument('--min-cnv-size', type=int, default=0, help="CNV 最小长度 (bp)")
    p.add_argument('--max-cnv-size', type=int, default=None, help="CNV 最大长度 (bp)")
    p.add_argument('--batch-size', type=int, default=256, help="滑窗推理批次大小")
    p.add_argument('--min-region-len', type=int, default=150, help="最小候选区域长度")
    p.add_argument('--cleanup', action='store_true', help="完成后删除中间文件")
    p.add_argument('--keep-bam', action='store_true', help="cleanup 时保留 BAM")
    p.add_argument('--model', default=None, help="模型路径")
    p.add_argument('--output-dir', default=str(PREDICTIONS_DIR))
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_fastq_pairs(input_dir)
    if not pairs:
        logger.error(f"No paired FASTQ files found in {input_dir}")
        sys.exit(1)
    logger.info(f"Found {len(pairs)} sample pair(s).")

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model) if args.model else MODEL_DIR / "best_model.pth"
    load_model_fn, predict_long_fn = _get_predict_funcs()
    model, ecc_class_index = load_model_fn(model_path, device)

    pipeline = CNVKitPipeline(output_dir=CNVKIT_TEMP_DIR)

    for base, r1, r2 in pairs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing sample: {base}")
        logger.info(f"{'='*60}")
        try:
            # 1. CNVkit: align + call
            call_cns = pipeline.align_and_call(r1, r2, base, threads=args.threads)

            # 2. 提取候选 CNV 区域
            candidates = pipeline.extract_candidates(
                call_cns, min_log2=args.min_log2,
                min_size=args.min_cnv_size, max_size=args.max_cnv_size,
                sample_name=base)

            if not candidates:
                logger.info(f"No CNV candidates for {base}, skipping.")
                if args.cleanup:
                    pipeline.cleanup_sample(base, keep_bam=args.keep_bam)
                continue

            logger.info(f"{len(candidates)} CNV candidate(s) for {base}")

            # 3. 批量提取序列到合并 FASTA
            temp_fa = CNVKIT_TEMP_DIR / f"{base}_candidates.fa"
            n_written = pipeline.extract_sequences(candidates, temp_fa)
            if n_written == 0:
                logger.warning(f"No sequences extracted for {base}")
                if args.cleanup:
                    pipeline.cleanup_sample(base, keep_bam=args.keep_bam)
                continue

            # 4. 对每个候选区域执行长序列滑窗预测
            records = parse_fasta_file(temp_fa)
            passed_bed = []
            passed_fasta = []

            for rec_header, rec_seq in records:
                regions, seq_clean, chrom_info = predict_long_fn(
                    rec_header, rec_seq, model, device, ecc_class_index,
                    limit=args.limit, batch_size=args.batch_size,
                    min_region_len=args.min_region_len)

                chrom = chrom_info.get('chrom', '')
                base_offset = chrom_info.get('start', 0) if chrom_info.get('has_position', False) else 0

                for rs, re in regions:
                    abs_s = base_offset + rs
                    abs_e = base_offset + re
                    # BED: 0-based half-open
                    passed_bed.append((chrom, max(0, abs_s), abs_e))
                    passed_fasta.append(
                        (f">{chrom}:{abs_s}-{abs_e}|prob>=limit", seq_clean[rs:re]))

            # 5. 写入最终结果
            final_bed = output_dir / f"{base}_microDNA.bed"
            final_fa = output_dir / f"{base}_microDNA.fasta"

            with open(final_bed, 'w', encoding='utf-8') as bf:
                for chrom, s, e in passed_bed:
                    bf.write(f"{chrom}\t{s}\t{e}\n")

            with open(final_fa, 'w', encoding='utf-8') as ff:
                for hdr, seq in passed_fasta:
                    ff.write(hdr + "\n")
                    for i in range(0, len(seq), 70):
                        ff.write(seq[i:i + 70] + "\n")

            logger.info(f"Sample {base}: {len(passed_bed)} MicroDNA(s) identified.")
            logger.info(f"  BED  -> {final_bed}")
            logger.info(f"  FASTA-> {final_fa}")

            # 6. 清理
            if args.cleanup:
                pipeline.cleanup_sample(base, keep_bam=args.keep_bam)

        except Exception as e:
            logger.error(f"Failed processing {base}: {e}", exc_info=True)

    logger.info("\nAll samples processed.")


if __name__ == '__main__':
    main()