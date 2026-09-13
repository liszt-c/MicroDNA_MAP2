"""
scripts/predict.py - 模型推理

模式:
  --mode short : 每条 FASTA 记录整体归一化到 400bp 后单次分类, 输出概率 TSV
  --mode long  : 两级滑动窗口扫描长序列, 输出候选区域 BED + FASTA

关键修复 (相对旧版 run.py / 重构 predict.py):
  * 多序列 FASTA 逐条处理 (旧版只读第一个 header, 后续序列全部拼接错误)
  * 坐标映射公式修正: seq_end = win_end_idx * STEP1 + SEQUENCE_LENGTH - STEP1
    (旧版多算一个步长, 末端系统性偏移)
  * BED/FASTA 写入时 abs_start/abs_end 在循环内计算 (旧版使用循环外泄漏变量)
  * 兼容原版权重 (eccDNA=类别0) 与新权重 (eccDNA=类别1), 通过 checkpoint meta 自动判断
  * 对已知旧版权重文件名 (如 6.pth) 自动回退到 eccDNA=class0
  * 长序列模式下默认合并重叠检测区域, 可通过 --no-merge 关闭

流式输出改造 (long 模式):
  * 输入侧: iter_fasta_file 惰性逐条读取, 内存与输入文件大小解耦
  * 输出侧: BED / FASTA 边处理边追加写入并逐条 flush, 不再在内存中累积全部结果;
    进程中断时已 flush 的部分结果保留在最终文件中
  * 每个输入文件额外生成 {stem}_progress.tsv 记录逐条处理进度 (可用 --no-progress-log 关闭)
  * 全部记录处理完仍无任何候选时, 删除空的 BED/FASTA (与旧行为一致: 无结果不产生文件)
"""
import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    MODEL_DIR, PREDICTIONS_DIR, SEQUENCE_LENGTH,
    SLIDE_STEP1, SLIDE_WINDOW2, SLIDE_STEP2
)
from src.model import ResNetSelfAttention
from src.dataprocess import encode_sequence, clean_sequence, parse_fasta_header
from src.utils import setup_logger, parse_fasta_file, iter_fasta_file

logger = setup_logger('predict')

STEP1 = SLIDE_STEP1
WINDOW2 = SLIDE_WINDOW2
STEP2 = SLIDE_STEP2

# 默认使用 eccDNA=class1
# 已知使用 eccDNA=class0 约定的旧版权重文件名
_LEGACY_ECC0_MODELS = {''}  # .pth


def torch_load_compat(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(model_path: Path, device):
    """
    加载模型, 自动检测标签约定。
    返回 (model, ecc_class_index)
    ecc_class_index: softmax 输出中 eccDNA 对应的列索引 (0 或 1)
    """
    model = ResNetSelfAttention()
    ckpt = torch_load_compat(model_path, device)

    ecc_class_index = 1  # 默认新约定

    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
        meta = {k: v for k, v in ckpt.items() if k != 'state_dict'}
        conv = meta.get('label_convention', '')
        if isinstance(conv, str) and 'eccdna=0' in conv.lower():
            ecc_class_index = 0
            logger.info("Detected legacy label convention: eccDNA=class0")
        elif isinstance(conv, str) and 'eccdna=1' in conv.lower():
            ecc_class_index = 1
            logger.info("Detected new label convention: eccDNA=class1")
        else:
            model_name = Path(model_path).name
            if model_name in _LEGACY_ECC0_MODELS:
                ecc_class_index = 0
                logger.info(f"Detected legacy model '{model_name}': assuming eccDNA=class0")
            else:
                logger.warning(f"No label_convention in checkpoint meta ({conv}), "
                               f"defaulting to eccDNA=class1. Verify against training config!")
    elif isinstance(ckpt, dict):
        state = ckpt
        model_name = Path(model_path).name
        if model_name in _LEGACY_ECC0_MODELS:
            ecc_class_index = 0
            logger.info(f"Detected legacy model '{model_name}' (bare state_dict): assuming eccDNA=class0")
        else:
            logger.warning("Bare state_dict without meta, defaulting to eccDNA=class1")
    else:
        logger.warning("Legacy full-model checkpoint; assuming eccDNA=class0 (original convention)")
        ecc_class_index = 0
        ckpt.to(device)
        ckpt.eval()
        return ckpt, ecc_class_index

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    model.to(device)
    model.eval()
    logger.info(f"Model loaded from {model_path} (ecc_class_index={ecc_class_index})")
    return model, ecc_class_index


def predict_short(seq: str, model, device, ecc_class_index: int) -> float:
    """短序列预测, 返回 eccDNA 概率"""
    encoded = encode_sequence(seq)
    tensor = torch.from_numpy(encoded).float().unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
    prob = torch.softmax(output, dim=1)[0, ecc_class_index].item()
    return prob


def merge_overlapping_regions(regions: List[Tuple[int, int]],
                              gap_tolerance: int = 0) -> List[Tuple[int, int]]:
    """
    合并重叠或相邻的区间。

    :param regions: [(start, end), ...] 已按 start 排序的 0-based 半开区间
    :param gap_tolerance: 允许的最大间隔 (bp); 0 = 仅合并真正重叠的区间
    :return: 合并后的区间列表
    """
    if not regions:
        return []
    sorted_regions = sorted(regions, key=lambda r: r[0])
    merged = [sorted_regions[0]]
    for start, end in sorted_regions[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + gap_tolerance:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def predict_long_single(header: str, seq_raw: str, model, device, ecc_class_index: int,
                        limit: float, batch_size: int, min_region_len: int,
                        do_merge: bool = True):
    """
    对单条长序列执行两级滑动窗口识别。
    返回: (regions, seq_clean, chrom_info)
    regions: [(start_bp, end_bp), ...] 相对于序列起始的 0-based 坐标
    """
    seq_clean = clean_sequence(seq_raw)
    n = len(seq_clean)
    chrom_info = parse_fasta_header(header)

    if n < 800:
        logger.debug(f"Sequence length {n} < 800bp, falling back to short mode.")
        prob = predict_short(seq_clean, model, device, ecc_class_index)
        if prob >= limit:
            return [(0, n)], seq_clean, chrom_info
        return [], seq_clean, chrom_info

    num_windows = max(0, (n - SEQUENCE_LENGTH) // STEP1 + 1)
    if num_windows == 0:
        return [], seq_clean, chrom_info

    logger.info(f"Sliding window: {num_windows} windows for sequence of length {n}")

    probs = []
    for i in tqdm(range(0, num_windows, batch_size), desc="Window inference", leave=False):
        cur_bs = min(batch_size, num_windows - i)
        batch_list = []
        for j in range(cur_bs):
            win_idx = i + j
            start = win_idx * STEP1
            end = start + SEQUENCE_LENGTH
            sub_seq = seq_clean[start:end]
            batch_list.append(encode_sequence(sub_seq))
        batch_tensor = torch.from_numpy(np.array(batch_list)).float().to(device)
        with torch.no_grad():
            outputs = model(batch_tensor)
        bp = torch.softmax(outputs, dim=1)[:, ecc_class_index].cpu().numpy()
        probs.extend(bp)

    probs = np.array(probs, dtype=np.float32)

    # 第二次滑动窗口平滑
    if len(probs) < WINDOW2:
        smoothed = probs.copy()
    else:
        num_win2 = (len(probs) - WINDOW2) // STEP2 + 1
        smoothed = np.zeros(num_win2, dtype=np.float32)
        for j in range(num_win2):
            s = j * STEP2
            e = s + WINDOW2
            smoothed[j] = np.mean(probs[s:e])

    binary = (smoothed >= limit).astype(np.int8)

    # 提取连续区域 (smoothed 索引)
    raw_regions = []
    in_region = False
    left = 0
    for idx, val in enumerate(binary):
        if val == 1 and not in_region:
            in_region = True
            left = idx
        elif val == 0 and in_region:
            in_region = False
            raw_regions.append((left, idx))
    if in_region:
        raw_regions.append((left, len(binary)))

    # 映射回原始序列坐标
    final_regions = []
    for l, r in raw_regions:
        win_start_idx = l * STEP2
        win_end_idx = (r - 1) * STEP2
        seq_start = win_start_idx * STEP1
        seq_end = win_end_idx * STEP1 + SEQUENCE_LENGTH
        seq_start = max(0, seq_start)
        seq_end = min(n, seq_end)
        if (seq_end - seq_start) >= min_region_len:
            final_regions.append((seq_start, seq_end))

    # 合并重叠区域 (默认开启)
    if do_merge:
        n_before = len(final_regions)
        final_regions = merge_overlapping_regions(final_regions, gap_tolerance=0)
        if len(final_regions) < n_before:
            logger.info(f"Merged {n_before} overlapping regions -> {len(final_regions)}")

    logger.info(f"Found {len(final_regions)} candidate regions (min_len={min_region_len}).")
    return final_regions, seq_clean, chrom_info


def _write_fasta_record(ff, hdr: str, seq: str, line_width: int = 70):
    """向已打开的文件句柄写入一条 FASTA 记录"""
    ff.write(hdr + "\n")
    for i in range(0, len(seq), line_width):
        ff.write(seq[i:i + line_width] + "\n")


def process_long_streaming(fa_file: Path, output_dir: Path, model, device,
                           ecc_class_index: int, args) -> int:
    """
    流式处理单个 FASTA 文件的 long 模式推理。

    - 输入: iter_fasta_file 逐条惰性读取
    - 输出: BED / FASTA / progress.tsv 边处理边追加写入, 每条记录后 flush
    - 返回累计检出的候选区域总数 (0 时删除空的 BED/FASTA)
    """
    bed_path = output_dir / f"{fa_file.stem}.bed"
    fasta_path = output_dir / f"{fa_file.stem}_candidates.fasta"
    progress_path = output_dir / f"{fa_file.stem}_progress.tsv"

    do_merge = not args.no_merge
    cand_counter = 0
    rec_idx = 0
    file_start = time.time()

    progress_f = None
    if not args.no_progress_log:
        progress_f = open(progress_path, 'w', encoding='utf-8')
        progress_f.write("record_idx\theader\tseq_len\tn_regions\tcumulative_regions\telapsed_sec\n")

    # BED / FASTA 以追加写方式打开, 每条记录处理完立即 flush 落盘
    with open(bed_path, 'w', encoding='utf-8') as bf, \
         open(fasta_path, 'w', encoding='utf-8') as ff:

        for header, seq_raw in iter_fasta_file(fa_file):
            rec_start = time.time()
            regions, seq_clean, chrom_info = predict_long_single(
                header, seq_raw, model, device, ecc_class_index,
                limit=args.limit, batch_size=args.batch_size,
                min_region_len=args.min_region_len, do_merge=do_merge)

            chrom = chrom_info.get('chrom', '') or f"seq{rec_idx}"
            base_offset = (chrom_info.get('start', 0)
                           if chrom_info.get('has_position', False) else 0)

            # ---- 当前记录的结果立即写入磁盘 ----
            for rs, re in regions:
                cand_counter += 1
                abs_s = base_offset + rs
                abs_e = base_offset + re
                bed_start = max(0, abs_s)
                bed_end = abs_e
                bf.write(f"{chrom}\t{bed_start}\t{bed_end}\n")

                fasta_hdr = f">candidate_{cand_counter}_{chrom}:{abs_s}-{abs_e}"
                _write_fasta_record(ff, fasta_hdr, seq_clean[rs:re])

            bf.flush()
            ff.flush()

            # ---- 进度记录 ----
            rec_elapsed = time.time() - rec_start
            if progress_f is not None:
                hdr_short = header[:120].replace('\t', ' ')
                progress_f.write(f"{rec_idx}\t{hdr_short}\t{len(seq_clean)}\t"
                                 f"{len(regions)}\t{cand_counter}\t{rec_elapsed:.2f}\n")
                progress_f.flush()

            logger.info(f"[{fa_file.name}] record {rec_idx} done: "
                        f"{len(regions)} region(s), cumulative {cand_counter}, "
                        f"{rec_elapsed:.2f}s")
            rec_idx += 1

    if progress_f is not None:
        progress_f.close()

    total_elapsed = time.time() - file_start

    # 无结果时删除空文件 (保持旧行为: run_microdna_map.py 依赖 glob("*.bed") 判断)
    if cand_counter == 0:
        bed_path.unlink(missing_ok=True)
        fasta_path.unlink(missing_ok=True)
        logger.info(f"No candidates found in {fa_file.name} "
                    f"({rec_idx} records, {total_elapsed:.2f}s)")
    else:
        logger.info(f"{fa_file.name}: {cand_counter} candidate(s) from {rec_idx} records "
                    f"({total_elapsed:.2f}s)")
        logger.info(f"  BED   -> {bed_path}")
        logger.info(f"  FASTA -> {fasta_path}")
        if progress_f is not None:
            logger.info(f"  Progress -> {progress_path}")

    return cand_counter


def main():
    p = argparse.ArgumentParser(description="MicroDNA Prediction",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input', required=True, help="FASTA 文件或目录")
    p.add_argument('--mode', choices=['short', 'long'], default='long')
    p.add_argument('--model', default=None, help="模型路径 (默认 models/best_model.pth)")
    p.add_argument('--limit', type=float, default=0.99, help="长序列阈值")
    p.add_argument('--batch-size', type=int, default=256, help="滑窗推理批次大小")
    p.add_argument('--min-region-len', type=int, default=150, help="最小候选区域长度 (bp)")
    p.add_argument('--no-merge', action='store_true',
                   help="禁用重叠区域合并 (默认自动合并滑窗产生的重叠检测)")
    p.add_argument('--no-progress-log', action='store_true',
                   help="不生成 {stem}_progress.tsv 逐记录进度文件")
    p.add_argument('--output-dir', default=str(PREDICTIONS_DIR))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model) if args.model else MODEL_DIR / "best_model.pth"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    model, ecc_class_index = load_model(model_path, device)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        fa_files = [input_path]
    else:
        fa_files = sorted(list(input_path.glob("*.fa")) + list(input_path.glob("*.fasta")))

    if not fa_files:
        logger.error(f"No FASTA files found in {input_path}")
        sys.exit(1)

    for fa_file in fa_files:
        logger.info(f"Processing {fa_file.name} ...")

        if args.mode == 'short':
            # short 模式本身就是逐条写 TSV, 但输入侧同样改为惰性读取
            tsv_path = output_dir / f"{fa_file.stem}_short_results.tsv"
            n_rec = 0
            with open(tsv_path, 'w', encoding='utf-8') as tf:
                tf.write("header\teccDNA_prob\n")
                for header, seq in iter_fasta_file(fa_file):
                    prob = predict_short(seq, model, device, ecc_class_index)
                    tf.write(f"{header}\t{prob:.6f}\n")
                    n_rec += 1
            if n_rec == 0:
                logger.warning(f"No records in {fa_file.name}, skipping.")
                tsv_path.unlink(missing_ok=True)
                continue
            logger.info(f"Short results ({n_rec} records) -> {tsv_path}")
        else:
            # long 模式: 全流式 (惰性读入 + 逐记录落盘)
            process_long_streaming(fa_file, output_dir, model, device,
                                   ecc_class_index, args)


if __name__ == '__main__':
    main()