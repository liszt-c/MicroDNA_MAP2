#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark/models_comparison/train_eval.py

用于统一训练和评估论文 3.1 节中的对比模型。
强制复用 src.dataloader 以确保同环境对比公平性。
加入了自适应 AMP (Automatic Mixed Precision) 加速逻辑。
"""

import argparse
import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm

# 将项目根目录加入环境变量以导入核心模块
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, RANDOM_SEED, PROCESSED_DATA_DIR
from src.dataloader import build_train_val_loaders, ECC_LABEL
from src.utils import setup_logger
from benchmark.models_comparison.models import ResNet50, ResNetNoAttention, TransformerClassifier

logger = setup_logger('models_comparison')


def evaluate(model, loader, device):
    """验证阶段保持全精度 (FP32) 推理，以获得最准确的指标评估"""
    model.eval()
    all_probs, all_labels, all_preds = [], [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            
            probs = torch.softmax(outputs, dim=1)[:, ECC_LABEL]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = np.array(all_preds)

    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else 0.0
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0

    return {'acc': acc, 'auc': auc, 'f1': f1}


def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate Baseline Models for Comparison")
    parser.add_argument('--model', type=str, required=True, 
                        choices=['resnet50', 'resnet_no_att', 'transformer'],
                        help="选择要评估的对比模型架构")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / "benchmark/results/models_comparison")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # === 智能 AMP 检测机制 ===
    # 保护旧架构(如 P40/P100)，在 Volta(V100) 及以上架构自动开启 Tensor Core 加速
    if torch.cuda.is_available():
        gpu_cap = torch.cuda.get_device_capability()
        use_amp = (gpu_cap[0] >= 7)
    else:
        use_amp = False
        
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # 强制固定种子以保证严格公平的对比环境
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    ecc_fa = PROCESSED_DATA_DIR / "eccDNA.fa"
    other_fa = PROCESSED_DATA_DIR / "otherDNA.fa"
    if not ecc_fa.exists() or not other_fa.exists():
        logger.error(f"Cannot find processed FASTA files in {PROCESSED_DATA_DIR}")
        sys.exit(1)

    logger.info("Building dataloaders (using core pipeline for fairness)...")
    train_loader, val_loader, _, _, _ = build_train_val_loaders(
        batch_size=args.batch_size, seed=RANDOM_SEED, balanced=True, 
        ecc_path=ecc_fa, other_path=other_fa
    )

    if args.model == 'resnet50':
        model = ResNet50().to(device)
    elif args.model == 'resnet_no_att':
        model = ResNetNoAttention().to(device)
    elif args.model == 'transformer':
        # Transformer 需要较小学习率，防止梯度爆炸
        args.lr = args.lr * 0.1 
        model = TransformerClassifier().to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    
    logger.info(f"Starting training for {args.model} on {device} | AMP Enabled: {use_amp}")
    
    best_metrics = {'epoch': 0, 'acc': 0, 'auc': 0, 'f1': 0}

    for epoch in range(1, args.epochs + 1):
        model.train()
        
        # 加入 tqdm 进度条，设置 leave=False 使其结束后自动清除，不扰乱 log 记录
        pbar = tqdm(train_loader, desc=f"{args.model.upper()} Epoch [{epoch}/{args.epochs}]", leave=False)
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # --- AMP 前向传播 ---
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # --- AMP 梯度缩放与反向传播 ---
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 实时更新进度条后缀
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        metrics = evaluate(model, val_loader, device)
        logger.info(f"Epoch [{epoch}/{args.epochs}] - ACC: {metrics['acc']:.4f}, AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")

        # 使用 AUC 决定最佳模型
        if metrics['auc'] > best_metrics['auc']:
            best_metrics = {'epoch': epoch, **metrics}
            torch.save(model.state_dict(), args.output_dir / f"best_{args.model}.pth")

    logger.info(f"Finished {args.model}. Best Epoch: {best_metrics['epoch']}, ACC: {best_metrics['acc']:.4f}, AUC: {best_metrics['auc']:.4f}")
    
    # 记录汇总结果
    csv_path = args.output_dir / "comparison_summary.csv"
    record = pd.DataFrame([{
        'Model': args.model, 
        'Best_Epoch': best_metrics['epoch'], 
        'Accuracy': best_metrics['acc'], 
        'AUC': best_metrics['auc'], 
        'F1_Score': best_metrics['f1']
    }])
    if csv_path.exists():
        record.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        record.to_csv(csv_path, index=False)

if __name__ == "__main__":
    main()