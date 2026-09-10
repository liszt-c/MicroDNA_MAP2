"""
src/dataprocess.py - 序列预处理与编码

关键修复 (相对旧版):
1. 字符串清洗正确赋值 (原版 seq.replace() 结果被丢弃)
2. One-hot 编码向量化实现 (原版为纯 Python 循环, 慢 ~50 倍)
3. parse_fasta_header 正确处理 '>SampleID|chr1:100-200|eccDNA' 格式
   (原实现中 end 字段带 label 后缀导致 int() 失败, start/end 恒为 0)
4. 长度使用完整 len(seq) (原版 n=len-1 是 off-by-one bug)
"""
import random
import numpy as np

from config import SEQUENCE_LENGTH

# 碱基 -> One-hot 行号 (与原版一致: A=0, T=1, C=2, G=3)
_BASE_TO_ROW = {ord('A'): 0, ord('T'): 1, ord('C'): 2, ord('G'): 3}


def clean_sequence(seq: str) -> str:
    """清理序列: 移除空白字符, 统一大写"""
    seq = seq.replace(" ", "").replace("\t", "") \
             .replace("\n", "").replace("\r", "")
    return seq.upper()


def encode_sequence(seq: str, length: int = SEQUENCE_LENGTH) -> np.ndarray:
    """
    DNA 序列 -> One-hot 矩阵 (4, length), dtype=float32

    - 序列 <= length: 居中放置, 两侧填充随机单碱基噪声 (与原版训练数据一致)
    - 序列 >  length: 截取正中间 length 个碱基
    - 非 ACGT 字符 (如 N) 对应位置全为 0
    """
    seq = clean_sequence(seq)
    n = len(seq)
    data = np.zeros((4, length), dtype=np.float32)

    if n == 0:
        # 全噪声填充
        rows = np.random.randint(0, 4, size=length)
        data[rows, np.arange(length)] = 1.0
        return data

    codes = np.frombuffer(seq.encode('ascii', errors='replace'), dtype=np.uint8).astype(np.int16)
    rows = np.full(n, -1, dtype=np.int16)
    for code, r in _BASE_TO_ROW.items():
        rows[codes == code] = r

    if n <= length:
        fund = (length - n) // 2

        # 左侧随机噪声
        if fund > 0:
            noise = np.random.randint(0, 4, size=fund)
            data[noise, np.arange(fund)] = 1.0

        # 真实序列 (居中)
        valid = rows >= 0
        if valid.any():
            cols = fund + np.nonzero(valid)[0]
            data[rows[valid], cols] = 1.0

        # 右侧随机噪声
        remaining = length - n - fund
        if remaining > 0:
            start = fund + n
            noise = np.random.randint(0, 4, size=remaining)
            data[noise, start + np.arange(remaining)] = 1.0
    else:
        # 截取正中间
        fund = (n - length) // 2
        sub_rows = rows[fund:fund + length]
        valid = sub_rows >= 0
        if valid.any():
            data[sub_rows[valid], np.nonzero(valid)[0]] = 1.0

    return data


def parse_fasta_header(header: str) -> dict:
    """
    解析 FASTA Header, 提取 ID / 染色体位置 / 标签

    支持格式:
      >SampleID|chr1:100-200|eccDNA      (process_data.py 生成的标准格式)
      >chr1:100-200                       (samtools 输出格式)
      >candidate_0_chr1:100-200           (cnvkit_pipeline 生成的格式)
      >seq_name description text          (普通 FASTA)

    返回 dict:
      id: str            序列 ID
      chrom: str         染色体名 (未解析到则为 '')
      start: int         起始坐标 (未解析到则为 0)
      end: int           终止坐标 (未解析到则为 0)
      has_position: bool 是否成功解析到坐标
      label: int|None    1=eccDNA, 0=otherDNA, None=未标注
    """
    info = {"id": "", "chrom": "", "start": 0, "end": 0,
            "has_position": False, "label": None}

    h = header.lstrip('>').strip()
    if not h:
        return info

    segments = [s.strip() for s in h.split('|') if s.strip()]

    # --- 解析位置: 在任意 segment 中找 'chrom:start-end' 模式 ---
    for seg in segments:
        if ':' not in seg:
            continue
        chrom_part, pos_part = seg.split(':', 1)
        # chrom_part 可能带前缀 (如 candidate_0_chr1), 取最后一段作为染色体名
        # 但保留完整 seg 供上游按需使用
        if '-' not in pos_part:
            continue
        start_str, end_str = pos_part.split('-', 1)
        # end_str 可能带尾部杂质, 取第一个非数字前
        end_clean = ''
        for ch in end_str:
            if ch.isdigit():
                end_clean += ch
            else:
                break
        try:
            start_val = int(start_str)
            end_val = int(end_clean) if end_clean else 0
        except ValueError:
            continue
        # 染色体名: 若含 '_', 取最后一段 (candidate_0_chr1 -> chr1)
        chrom_name = chrom_part.strip()
        if '_' in chrom_name:
            chrom_name = chrom_name.rsplit('_', 1)[-1]
        info['chrom'] = chrom_name
        info['start'] = start_val
        info['end'] = end_val
        info['has_position'] = True
        break

    # --- 解析标签 ---
    for seg in segments[1:] if len(segments) > 1 else []:
        sl = seg.lower()
        if 'ecc' in sl:
            info['label'] = 1
            break
        if 'other' in sl or 'genomic' in sl:
            info['label'] = 0
            break

    # --- 解析 ID ---
    if len(segments) > 1:
        info['id'] = segments[0].split(':')[0].split()[0]
    else:
        info['id'] = h.split()[0].split(':')[0]

    return info