"""
src/utils.py - 通用工具函数
"""
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator, Tuple


def setup_logger(name: str, log_file: Path = None, level=logging.INFO) -> logging.Logger:
    """设置日志记录器 (防止重复添加 handler)"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def run_command(cmd: list, check: bool = True, cwd: Path = None) -> subprocess.CompletedProcess:
    """安全地执行子进程命令 (列表形式, 无 shell 注入风险)"""
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd else None
        )
        return result
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"[ERROR] Command failed: {' '.join(map(str, cmd))}\n")
        sys.stderr.write(f"[ERROR] returncode={e.returncode}\n")
        if e.stdout:
            sys.stderr.write(f"[ERROR] stdout: {e.stdout[-2000:]}\n")
        if e.stderr:
            sys.stderr.write(f"[ERROR] stderr: {e.stderr[-2000:]}\n")
        raise
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Executable not found: '{cmd[0]}'. "
            f"Please install it or set the full path in config.py"
        )


def parse_fasta_text(text: str):
    """解析 FASTA 格式文本, 返回 [(header_line, sequence), ...]"""
    records = []
    header = None
    parts = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if header is not None:
                records.append((header, ''.join(parts)))
            header = line
            parts = []
        else:
            parts.append(line)
    if header is not None:
        records.append((header, ''.join(parts)))
    return records


def parse_fasta_file(path: Path):
    """解析 FASTA 文件 (整读入内存, 仅适用于小文件)"""
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return parse_fasta_text(f.read())


def iter_fasta_file(path: Path) -> Iterator[Tuple[str, str]]:
    """惰性解析 FASTA 文件, 逐条 yield (header_line, sequence)"""
    path = Path(path)
    header = None
    parts = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(parts)
                header = line
                parts = []
            else:
                if header is not None:
                    parts.append(line)
    if header is not None:
        yield header, ''.join(parts)


def read_fai(ref_fa: Path) -> dict:
    """读取 .fai 索引, 返回 {'lengths': {chrom: length}, 'order': {chrom: index}}"""
    fai_path = Path(str(ref_fa) + '.fai')
    lengths, order = {}, {}
    if fai_path.exists():
        with open(fai_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                parts = line.rstrip('\n').split('\t')
                if len(parts) >= 2:
                    lengths[parts[0]] = int(parts[1])
                    order[parts[0]] = i
    return {'lengths': lengths, 'order': order}


def ensure_faidx(ref_fa: Path, samtools_bin: str = 'samtools') -> None:
    """确保参考基因组已建立 samtools faidx 索引"""
    ref_fa = Path(ref_fa)
    if not ref_fa.exists():
        raise FileNotFoundError(f"Reference genome not found: {ref_fa}")
    fai_path = Path(str(ref_fa) + '.fai')
    if not fai_path.exists():
        run_command([samtools_bin, 'faidx', str(ref_fa)])


def ensure_bam_index(bam_path: Path, samtools_bin: str = 'samtools') -> None:
    """确保 BAM 文件已建立 .bai 索引"""
    bam_path = Path(bam_path)
    if not bam_path.exists():
        raise FileNotFoundError(f"BAM file not found: {bam_path}")
    bai1 = Path(str(bam_path) + ".bai")
    bai2 = bam_path.with_suffix(".bai")
    if not bai1.exists() and not bai2.exists():
        run_command([samtools_bin, "index", str(bam_path)])


def extract_regions(ref_fa: Path, regions: list, samtools_bin: str = 'samtools'):
    """
    批量提取基因组区域序列 (单次调用 samtools faidx -r, 避免逐条产生子进程)
    """
    ref_fa = Path(ref_fa)
    fai = read_fai(ref_fa)
    lengths, order = fai['lengths'], fai['order']
    if not lengths:
        raise FileNotFoundError(f".fai index missing for {ref_fa}, run ensure_faidx first")

    valid = []
    skipped = []
    for region in regions:
        chrom, start, end = region
        if chrom not in lengths:
            skipped.append((region, f"chrom '{chrom}' not in reference index"))
            continue
        start = max(1, int(start))
        end = min(int(end), lengths[chrom])
        if start > end:
            skipped.append((region, f"invalid coordinate range after clamping ({start}>{end})"))
            continue
        valid.append((region, chrom, start, end))

    if not valid:
        return {}, skipped

    valid.sort(key=lambda t: (order[t[1]], t[2]))

    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False, encoding='utf-8') as tf:
        for _, chrom, start, end in valid:
            tf.write(f"{chrom}:{start}-{end}\n")
        tf_path = tf.name

    try:
        result = run_command([samtools_bin, 'faidx', str(ref_fa), '-r', tf_path])
    finally:
        os.unlink(tf_path)

    records = parse_fasta_text(result.stdout)
    seq_map = {}
    for (orig_key, _c, _s, _e), (_hdr, seq) in zip(valid, records):
        seq_map[orig_key] = seq

    if len(records) != len(valid):
        sys.stderr.write(f"[WARN] samtools returned {len(records)} records for {len(valid)} regions\n")

    return seq_map, skipped