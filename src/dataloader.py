"""
src/dataloader.py - PyTorch Dataset 实现

设计要点:
- 启动时以二进制模式扫描 FASTA, 仅记录 (header, 字节偏移, label), 内存 O(记录数)
- __getitem__ 时 seek 到偏移量按需读取单条序列 (惰性加载, 百万级序列不爆内存)
- 文件句柄按进程缓存并在 fork/spawn 后自动重建, 兼容 DataLoader 多 worker
- worker_init_fn 隔离 numpy/random 种子, 避免各 worker 产生相同的随机噪声填充
- 不依赖任何第三方 FASTA 库
"""
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split

from config import (PROCESSED_DATA_DIR, DEFAULT_BATCH_SIZE, NUM_WORKERS, RANDOM_SEED)
from .dataprocess import encode_sequence, parse_fasta_header
from .utils import setup_logger

logger = setup_logger('dataloader')

ECC_LABEL = 1        # eccDNA
OTHER_LABEL = 0      # otherDNA / genomic background


def seed_worker(worker_id: int):
    """DataLoader worker 初始化: 为每个 worker 设置独立的 numpy/random 种子"""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class MicroDNADataset(Dataset):
    """从单个合并 FASTA 文件加载序列的 Dataset"""

    def __init__(self, fasta_path, fallback_label: int = None):
        """
        :param fasta_path: FASTA 文件路径 (如 data/processed/eccDNA.fa)
        :param fallback_label: header 中无标签时使用的默认标签;
                               None 时按文件名推断 (含 'eccdna' -> 1, 否则 0)
        """
        self.fasta_path = Path(fasta_path)
        if not self.fasta_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {self.fasta_path}")

        if fallback_label is None:
            fallback_label = ECC_LABEL if 'eccdna' in self.fasta_path.name.lower() else OTHER_LABEL
        self.fallback_label = int(fallback_label)

        self.headers = []
        self.offsets = []     # 每条记录 '>' 行的字节偏移
        self.labels = []

        # 惰性文件句柄 (不参与序列化)
        self._fh = None
        self._fh_pid = None

        self._build_index()
        self.size = len(self.offsets)
        if self.size == 0:
            raise ValueError(f"No FASTA records found in {self.fasta_path}")

        n_ecc = sum(1 for l in self.labels if l == ECC_LABEL)
        logger.info(f"Indexed {self.size} records from {self.fasta_path.name} "
                    f"(ecc={n_ecc}, other={self.size - n_ecc})")

    # ------------------------------------------------------------------ #
    def _build_index(self):
        """二进制扫描, 记录每条记录 header 行的精确字节偏移"""
        with open(self.fasta_path, 'rb') as f:
            pos = f.tell()
            line = f.readline()
            while line:
                if line.startswith(b'>'):
                    header = line.decode('utf-8', errors='replace').strip()
                    info = parse_fasta_header(header)
                    label = info['label'] if info['label'] is not None else self.fallback_label
                    self.headers.append(header)
                    self.offsets.append(pos)
                    self.labels.append(int(label))
                pos = f.tell()
                line = f.readline()

    def _get_fh(self):
        """获取当前进程的文件句柄 (fork/spawn 后自动重开)"""
        pid = os.getpid()
        if self._fh is None or self._fh_pid != pid:
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception:
                    pass
            self._fh = open(self.fasta_path, 'rb')
            self._fh_pid = pid
        return self._fh

    def _read_sequence(self, idx: int) -> str:
        """seek 到指定偏移, 读取单条序列 (支持多行折叠格式)"""
        fh = self._get_fh()
        fh.seek(self.offsets[idx])
        fh.readline()                       # 跳过 header 行
        parts = []
        for line in fh:
            if line.startswith(b'>'):
                break
            parts.append(line.strip())
        return b''.join(parts).decode('utf-8', errors='replace')

    # ------------------------------------------------------------------ #
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.size:
            raise IndexError(f"index {idx} out of range [0, {self.size})")
        seq = self._read_sequence(idx)
        data = torch.from_numpy(encode_sequence(seq)).float()       # (4, SEQUENCE_LENGTH)
        label = torch.tensor(self.labels[idx], dtype=torch.long)    # CrossEntropyLoss 要求 long
        return data, label

    def get_header_info(self, idx: int) -> dict:
        """返回指定记录的 header 解析结果 (调试/追溯用)"""
        return parse_fasta_header(self.headers[idx])

    # ------------------------------------------------------------------ #
    def __getstate__(self):
        """序列化时剔除文件句柄 (Windows spawn 模式必需)"""
        state = self.__dict__.copy()
        state['_fh'] = None
        state['_fh_pid'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __del__(self):
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass


class CombinedMicroDNADataset(Dataset):
    """合并 eccDNA.fa (label=1) 与 otherDNA.fa (label=0)"""

    def __init__(self, ecc_path=None, other_path=None):
        ecc_path = Path(ecc_path) if ecc_path else PROCESSED_DATA_DIR / "eccDNA.fa"
        other_path = Path(other_path) if other_path else PROCESSED_DATA_DIR / "otherDNA.fa"
        if not ecc_path.exists() or not other_path.exists():
            raise FileNotFoundError(
                f"Processed FASTA not found (expected {ecc_path} and {other_path}). "
                f"Run scripts/process_data.py first.")
        self.ecc_ds = MicroDNADataset(ecc_path, fallback_label=ECC_LABEL)
        self.other_ds = MicroDNADataset(other_path, fallback_label=OTHER_LABEL)
        self.n_ecc = len(self.ecc_ds)

    def __len__(self):
        return self.n_ecc + len(self.other_ds)

    def __getitem__(self, idx):
        if idx < self.n_ecc:
            return self.ecc_ds[idx]
        return self.other_ds[idx - self.n_ecc]

    def label_of(self, idx: int) -> int:
        return ECC_LABEL if idx < self.n_ecc else OTHER_LABEL


def subset_labels(full_ds: CombinedMicroDNADataset, subset: Subset) -> np.ndarray:
    """获取 random_split 子集的标签数组 (用于类别平衡采样/统计)"""
    return np.array([full_ds.label_of(i) for i in subset.indices], dtype=np.int64)


def build_train_val_loaders(batch_size: int = DEFAULT_BATCH_SIZE,
                            val_split: float = 0.2,
                            seed: int = RANDOM_SEED,
                            num_workers: int = NUM_WORKERS,
                            balanced: bool = False,
                            ecc_path=None, other_path=None):
    """
    构建训练/验证 DataLoader (固定种子划分, 可复现)
    :param balanced: True 时对训练集使用 WeightedRandomSampler 平衡正负样本
    :return: (train_loader, val_loader, full_dataset, train_subset, val_subset)
    """
    dataset = CombinedMicroDNADataset(ecc_path, other_path)

    total = len(dataset)
    val_n = int(round(total * val_split))
    train_n = total - val_n
    if train_n == 0 or val_n == 0:
        raise ValueError(f"Dataset too small to split: total={total}, val_split={val_split}")

    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_n, val_n], generator=g)

    sampler = None
    if balanced:
        labels = subset_labels(dataset, train_ds)
        counts = np.bincount(labels, minlength=2).astype(np.float64)
        if (counts == 0).any():
            raise ValueError(f"Balanced sampling impossible, class counts = {counts.tolist()}")
        weight_per_class = 1.0 / counts
        weights = weight_per_class[labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights), replacement=True)
        logger.info(f"Balanced sampling enabled. class counts = {counts.tolist()}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=(sampler is None),
                              sampler=sampler, num_workers=num_workers,
                              worker_init_fn=seed_worker, generator=g, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, worker_init_fn=seed_worker,
                            generator=g, drop_last=False)

    logger.info(f"Split: train={train_n}, val={val_n} (seed={seed})")
    return train_loader, val_loader, dataset, train_ds, val_ds


def build_full_loader(batch_size: int = DEFAULT_BATCH_SIZE,
                      num_workers: int = NUM_WORKERS,
                      ecc_path=None, other_path=None):
    """构建覆盖全部数据的 DataLoader (评估用, 不打乱)"""
    dataset = CombinedMicroDNADataset(ecc_path, other_path)
    g = torch.Generator().manual_seed(RANDOM_SEED)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    logger.info(f"Full evaluation set: {len(dataset)} records")
    return loader, dataset