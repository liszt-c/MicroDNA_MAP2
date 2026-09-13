"""
src/hnm.py - 困难负例挖掘 (Hard Negative Mining) 模块
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .dataloader import seed_worker, subset_labels
from .utils import setup_logger

logger = setup_logger('hnm')


def perform_hnm(model, full_dataset, current_train_subset, batch_size, num_workers, device,
                threshold=0.1, keep_easy_ratio=0.1, balanced=False, seed=42):
    """
    在训练中途执行困难负例挖掘
    
    :param threshold: 负例的预测概率 >= 此阈值时，被定义为困难负例 (Hard Negative)
    :param keep_easy_ratio: 保留多少比例的简单负例，以防止灾难性遗忘
    :return: (new_train_loader, new_train_subset)
    """
    logger.info(f"Starting Hard Negative Mining evaluation on {len(current_train_subset)} samples...")
    model.eval()

    # 必须使用 shuffle=False，以保证预测结果与 subset indices 一一对应
    eval_loader = DataLoader(current_train_subset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in eval_loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            # 提取正例 (eccDNA=1) 的预测概率
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    # 获取当前训练集在全量 Dataset 中的绝对索引
    original_indices = np.array(current_train_subset.indices)

    # 1. 划分样本掩码 (Masks)
    pos_mask = (all_labels == 1)
    neg_mask = (all_labels == 0)

    hard_neg_mask = neg_mask & (all_probs >= threshold)
    easy_neg_mask = neg_mask & (all_probs < threshold)

    pos_indices = original_indices[pos_mask]
    hard_neg_indices = original_indices[hard_neg_mask]
    easy_neg_indices = original_indices[easy_neg_mask]

    logger.info("=== HNM Statistics before resampling ===")
    logger.info(f"  Positives (eccDNA):      {len(pos_indices)}")
    logger.info(f"  Hard Negatives (Prob >= {threshold}): {len(hard_neg_indices)}")
    logger.info(f"  Easy Negatives (Prob <  {threshold}): {len(easy_neg_indices)}")

    # 2. 采样策略：保留所有正例 + 保留所有困难负例 + 随机保留部分简单负例
    rng = np.random.default_rng(seed)
    
    # 保证简单负例的数量至少不低于困难负例，维持背景知识
    target_easy_count = int(len(easy_neg_indices) * keep_easy_ratio)
    target_easy_count = max(target_easy_count, len(hard_neg_indices))
    target_easy_count = min(target_easy_count, len(easy_neg_indices))  # 防止越界

    sampled_easy_neg_indices = rng.choice(easy_neg_indices, size=target_easy_count, replace=False)

    # 3. 合并新索引并打乱
    new_indices = np.concatenate([pos_indices, hard_neg_indices, sampled_easy_neg_indices])
    rng.shuffle(new_indices)
    new_train_subset = Subset(full_dataset, new_indices.tolist())

    # 4. 重建 DataLoader (如果使用了类别平衡采样，需要重新计算权重)
    sampler = None
    if balanced:
        labels = subset_labels(full_dataset, new_train_subset)
        counts = np.bincount(labels, minlength=2).astype(np.float64)
        weight_per_class = 1.0 / np.maximum(counts, 1)
        weights = weight_per_class[labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights), replacement=True
        )

    g = torch.Generator().manual_seed(seed)
    new_loader = DataLoader(new_train_subset, batch_size=batch_size, shuffle=(sampler is None),
                            sampler=sampler, num_workers=num_workers,
                            worker_init_fn=seed_worker, generator=g, drop_last=False)

    logger.info("=== HNM Dataset Rebuilt ===")
    logger.info(f"  Total samples now: {len(new_train_subset)}")
    if balanced:
        logger.info(f"  Balanced weights updated for new class counts: {counts.tolist()}")

    return new_loader, new_train_subset