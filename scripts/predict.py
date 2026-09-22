"""
scripts/predict.py - 高吞吐量模型推理

模式:
  --mode short : 批量归一化到 400bp 后以大 Batch 进行单次分类, 输出概率 TSV
  --mode long  : 跨序列滑动窗口聚合推理 (支持 Batch Size >= 512 满载运行), 输出候选区域 BED + FASTA
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
from src.utils import setup_logger, iter_fasta_file

logger = setup_logger('predict')

STEP1 = SLIDE_STEP1
WINDOW2 = SLIDE_WINDOW2
STEP2 = SLIDE_STEP2

_LEGACY_ECC0_MODELS = {''}


def torch_load_compat(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(model_path: Path, device):
    """加载模型, 自动检测标签约定及通道尺寸"""
    model = ResNetSelfAttention()
    ckpt = torch_load_compat(model_path, device)

    ecc_class_index = 1

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
    elif isinstance(ckpt, dict):
        state = ckpt
        if Path(model_path).name in _LEGACY_ECC0_MODELS:
            ecc_class_index = 0
    else:
        logger.warning("Legacy full-model checkpoint; assuming eccDNA=class0")
        ckpt.to(device)
        ckpt.eval()
        return ckpt, 0

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    logger.info(f"Model loaded from {model_path} (ecc_class_index={ecc_class_index})")
    return model, ecc_class_index


def merge_overlapping_regions(regions: List[Tuple[int, int]], gap_tolerance: int = 0) -> List[Tuple[int, int]]:
    """合并重叠或相邻的区间"""
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


def _write_fasta_record(ff, hdr: str, seq: str, line_width: int = 70):
    ff.write(hdr + "\n")
    for i in range(0, len(seq), line_width):
        ff.write(seq[i : i + line_width] + "\n")


# =====================================================================
# 高性能模式 1: Short 批量流式处理
# =====================================================================

def process_short_batched(fa_file: Path, output_dir: Path, model, device, ecc_class_index: int, batch_size: int):
    tsv_path = output_dir / f"{fa_file.stem}_short_results.tsv"
    n_rec = 0
    batch_headers, batch_seqs = [], []

    with open(tsv_path, 'w', encoding='utf-8') as tf:
        tf.write("header\teccDNA_prob\n")

        for header, seq in iter_fasta_file(fa_file):
            batch_headers.append(header)
            batch_seqs.append(encode_sequence(seq))

            if len(batch_seqs) >= batch_size:
                tensor = torch.from_numpy(np.array(batch_seqs)).float().to(device)
                with torch.no_grad():
                    outputs = model(tensor)
                    probs = torch.softmax(outputs, dim=1)[:, ecc_class_index].cpu().numpy()
                for h, p in zip(batch_headers, probs):
                    tf.write(f"{h}\t{p:.6f}\n")
                n_rec += len(batch_headers)
                batch_headers, batch_seqs = [], []

        # 处理末尾残余
        if batch_seqs:
            tensor = torch.from_numpy(np.array(batch_seqs)).float().to(device)
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)[:, ecc_class_index].cpu().numpy()
            for h, p in zip(batch_headers, probs):
                tf.write(f"{h}\t{p:.6f}\n")
            n_rec += len(batch_headers)

    if n_rec == 0:
        tsv_path.unlink(missing_ok=True)
        logger.warning(f"No records found in {fa_file.name}")
    else:
        logger.info(f"Short results ({n_rec} records) -> {tsv_path}")


# =====================================================================
# 高性能模式 2: Long 跨序列窗口聚合流式推理
# =====================================================================

class SequenceTracker:
    """维护单条长序列及其所有滑动窗口的状态"""
    def __init__(self, rec_idx: int, header: str, seq_raw: str):
        self.rec_idx = rec_idx
        self.header = header
        self.seq_clean = clean_sequence(seq_raw)
        self.n = len(self.seq_clean)
        self.chrom_info = parse_fasta_header(header)
        
        # 只要长度 >= SEQUENCE_LENGTH (400bp)，即可进行滑窗
        if self.n >= SEQUENCE_LENGTH:
            self.num_windows = (self.n - SEQUENCE_LENGTH) // STEP1 + 1
        else:
            self.num_windows = 1  # 退化为单次编码

        self.probs = np.zeros(self.num_windows, dtype=np.float32)
        self.filled_windows = 0

    def get_window_seq(self, win_idx: int) -> np.ndarray:
        if self.n >= SEQUENCE_LENGTH:
            start = win_idx * STEP1
            end = start + SEQUENCE_LENGTH
            return encode_sequence(self.seq_clean[start:end])
        else:
            return encode_sequence(self.seq_clean)

    def is_complete(self) -> bool:
        return self.filled_windows >= self.num_windows


def finalize_sequence_regions(tracker: SequenceTracker, limit: float, min_region_len: int, do_merge: bool):
    """当序列所有窗口推理完成后，执行两级平滑与区域合并"""
    n = tracker.n
    probs = tracker.probs

    if n < SEQUENCE_LENGTH:
        if probs[0] >= limit:
            return [(0, n)], tracker.seq_clean, tracker.chrom_info
        return [], tracker.seq_clean, tracker.chrom_info

    # 第二级平滑
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

    final_regions = []
    for l, r in raw_regions:
        win_start_idx = l * STEP2
        win_end_idx = (r - 1) * STEP2
        seq_start = max(0, win_start_idx * STEP1)
        seq_end = min(n, win_end_idx * STEP1 + SEQUENCE_LENGTH)
        if (seq_end - seq_start) >= min_region_len:
            final_regions.append((seq_start, seq_end))

    if do_merge:
        final_regions = merge_overlapping_regions(final_regions, gap_tolerance=0)

    return final_regions, tracker.seq_clean, tracker.chrom_info


def process_long_batched(fa_file: Path, output_dir: Path, model, device, ecc_class_index: int, args):
    bed_path = output_dir / f"{fa_file.stem}.bed"
    fasta_path = output_dir / f"{fa_file.stem}_candidates.fasta"
    progress_path = output_dir / f"{fa_file.stem}_progress.tsv"

    do_merge = not args.no_merge
    cand_counter = 0
    file_start = time.time()

    progress_f = None
    if not args.no_progress_log:
        progress_f = open(progress_path, 'w', encoding='utf-8')
        progress_f.write("record_idx\theader\tseq_len\tn_regions\tcumulative_regions\telapsed_sec\n")

    # 跨序列窗口缓冲队列
    window_batch_tensors = []
    window_batch_routes = []  # (tracker_id, win_idx)
    active_trackers = {}
    rec_counter = 0

    with open(bed_path, 'w', encoding='utf-8') as bf, open(fasta_path, 'w', encoding='utf-8') as ff:

        def flush_batch():
            nonlocal cand_counter
            if not window_batch_tensors:
                return

            tensor = torch.from_numpy(np.array(window_batch_tensors)).float().to(device)
            with torch.no_grad():
                outputs = model(tensor)
                batch_probs = torch.softmax(outputs, dim=1)[:, ecc_class_index].cpu().numpy()

            for (t_id, w_idx), prob in zip(window_batch_routes, batch_probs):
                tracker = active_trackers[t_id]
                tracker.probs[w_idx] = prob
                tracker.filled_windows += 1

            window_batch_tensors.clear()
            window_batch_routes.clear()

            # 结算并释放已完成所有窗口推理的序列
            completed_ids = [t_id for t_id, tr in active_trackers.items() if tr.is_complete()]
            for t_id in completed_ids:
                tracker = active_trackers.pop(t_id)
                regions, seq_clean, chrom_info = finalize_sequence_regions(
                    tracker, limit=args.limit, min_region_len=args.min_region_len, do_merge=do_merge
                )

                chrom = chrom_info.get('chrom', '') or f"seq{tracker.rec_idx}"
                base_offset = max(0, chrom_info.get('start', 1) - 1) if chrom_info.get('has_position', False) else 0

                for rs, re in regions:
                    cand_counter += 1
                    abs_s = base_offset + rs
                    abs_e = base_offset + re
                    bf.write(f"{chrom}\t{max(0, abs_s)}\t{abs_e}\n")
                    fasta_hdr = f">candidate_{cand_counter}_{chrom}:{abs_s}-{abs_e}"
                    _write_fasta_record(ff, fasta_hdr, seq_clean[rs:re])

                bf.flush()
                ff.flush()

                if progress_f is not None:
                    hdr_short = tracker.header[:120].replace('\t', ' ')
                    progress_f.write(
                        f"{tracker.rec_idx}\t{hdr_short}\t{tracker.n}\t"
                        f"{len(regions)}\t{cand_counter}\t{time.time() - file_start:.2f}\n"
                    )
                    progress_f.flush()

        # 流式读取序列生成窗口切片
        pbar = tqdm(desc=f"Streaming {fa_file.name}", unit="seq")
        for header, seq_raw in iter_fasta_file(fa_file):
            tracker = SequenceTracker(rec_counter, header, seq_raw)
            active_trackers[rec_counter] = tracker

            for w in range(tracker.num_windows):
                window_batch_tensors.append(tracker.get_window_seq(w))
                window_batch_routes.append((rec_counter, w))

                if len(window_batch_tensors) >= args.batch_size:
                    flush_batch()

            rec_counter += 1
            pbar.update(1)

        pbar.close()

        # 刷新末尾缓冲
        while active_trackers:
            flush_batch()

    if progress_f is not None:
        progress_f.close()

    total_elapsed = time.time() - file_start

    if cand_counter == 0:
        bed_path.unlink(missing_ok=True)
        fasta_path.unlink(missing_ok=True)
        logger.info(f"No candidates found in {fa_file.name} ({rec_counter} records, {total_elapsed:.2f}s)")
    else:
        logger.info(f"{fa_file.name}: {cand_counter} candidate(s) from {rec_counter} records in {total_elapsed:.2f}s")
        logger.info(f"  BED   -> {bed_path}")
        logger.info(f"  FASTA -> {fasta_path}")


def main():
    p = argparse.ArgumentParser(description="MicroDNA High-Throughput Prediction", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input', required=True, help="FASTA 文件或目录")
    p.add_argument('--mode', choices=['short', 'long'], default='long')
    p.add_argument('--model', default=None, help="模型路径 (默认 models/best_model.pth)")
    p.add_argument('--limit', type=float, default=0.75, help="判定阈值")
    p.add_argument('--batch-size', type=int, default=512, help="全局推理批次大小 (推荐 512 或 1024)")
    p.add_argument('--min-region-len', type=int, default=150, help="最小候选区域长度 (bp)")
    p.add_argument('--no-merge', action='store_true', help="禁用重叠区域合并")
    p.add_argument('--no-progress-log', action='store_true', help="不生成逐记录进度文件")
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

    fa_files = [input_path] if input_path.is_file() else sorted(list(input_path.glob("*.fa")) + list(input_path.glob("*.fasta")))
    if not fa_files:
        logger.error(f"No FASTA files found in {input_path}")
        sys.exit(1)

    for fa_file in fa_files:
        logger.info(f"Processing {fa_file.name} [Mode: {args.mode}, BatchSize: {args.batch_size}] ...")
        if args.mode == 'short':
            process_short_batched(fa_file, output_dir, model, device, ecc_class_index, args.batch_size)
        else:
            process_long_batched(fa_file, output_dir, model, device, ecc_class_index, args)


if __name__ == '__main__':
    main()