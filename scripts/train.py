"""
scripts/train.py - 模型训练入口

集成了多阶段在线困难负例挖掘 (Multi-round Online Hard Negative Mining) 机制。
逻辑：
  1. Base Stage: 训练 base-epochs 轮次，产出初步的最佳模型。
  2. HNM Stages: 进行 hnm-rounds 轮。每轮开始前，加载上一阶段的 best_model.pth，
     对训练集进行重采样挖掘，重置优化器和学习率，训练 hnm-epochs 轮次。
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (MODEL_DIR, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE,
                    DEFAULT_WEIGHT_DECAY, DEFAULT_STEP_SIZE, DEFAULT_GAMMA,
                    DEFAULT_FLOODING_B, RANDOM_SEED, NUM_WORKERS, PROCESSED_DATA_DIR,
                    LAYER_SIZE, DEFAULT_BASE_EPOCHS, DEFAULT_HNM_ROUNDS,
                    DEFAULT_HNM_EPOCHS, DEFAULT_HNM_THRESHOLD, DEFAULT_HNM_KEEP_EASY)
from src.model import ResNetSelfAttention
from src.dataloader import build_train_val_loaders, subset_labels, ECC_LABEL
from src.hnm import perform_hnm
from src.utils import setup_logger

logger = setup_logger('train')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device, flooding_b: float):
    """在验证集上评估, 返回指标字典。验证集全生命周期不变。"""
    model.eval()
    total_loss, n_batches, n_samples = 0.0, 0, 0
    all_probs, all_labels, all_preds = [], [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if flooding_b > 0:
                loss = (loss - flooding_b).abs() + flooding_b

            total_loss += float(loss.item())
            n_batches += 1
            n_samples += labels.size(0)

            probs = torch.softmax(outputs, dim=1)[:, ECC_LABEL]
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            all_preds.append(outputs.argmax(dim=1).detach().cpu().numpy())

    y_true = np.concatenate(all_labels) if all_labels else np.array([])
    y_prob = np.concatenate(all_probs) if all_probs else np.array([])
    y_pred = np.concatenate(all_preds) if all_preds else np.array([])

    metrics = {
        'loss': total_loss / max(n_batches, 1),
        'n': n_samples,
        'acc': float(accuracy_score(y_true, y_pred)) if n_samples else 0.0,
        'f1': float(f1_score(y_true, y_pred, zero_division=0)) if n_samples else 0.0,
        'auc': 0.0,
    }
    if n_samples and len(np.unique(y_true)) > 1:
        try:
            metrics['auc'] = float(roc_auc_score(y_true, y_prob))
        except ValueError as e:
            logger.warning(f"AUC computation failed: {e}")
    return metrics


def main():
    p = argparse.ArgumentParser(description='Train MicroDNA classifier with Multi-round HNM',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--base-epochs', type=int, default=DEFAULT_BASE_EPOCHS, help="基础训练轮数")
    p.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE)
    p.add_argument('--weight-decay', type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument('--step-size', type=int, default=DEFAULT_STEP_SIZE, help="StepLR 衰减间隔 (epoch)")
    p.add_argument('--gamma', type=float, default=DEFAULT_GAMMA, help="StepLR 衰减率")
    p.add_argument('--flooding-b', type=float, default=DEFAULT_FLOODING_B, help="Flooding 正则化参数 b")
    p.add_argument('--val-split', type=float, default=0.2)
    p.add_argument('--balanced', action='store_true', help="训练集使用类别平衡采样")
    p.add_argument('--num-workers', type=int, default=NUM_WORKERS)
    p.add_argument('--seed', type=int, default=RANDOM_SEED)
    p.add_argument('--output-dir', type=str, default=str(MODEL_DIR))
    p.add_argument('--ecc-excel-fasta', dest='ecc_fa', default=None)
    p.add_argument('--other-fasta', dest='other_fa', default=None)
    
    p.add_argument('--layer-size', type=int, default=LAYER_SIZE, help=f"ResNet 基础通道数")
    
    # ---- 困难负例挖掘参数 ----
    p.add_argument('--hnm-rounds', type=int, default=DEFAULT_HNM_ROUNDS, help="HNM 挖掘的轮次，0表示不挖掘")
    p.add_argument('--hnm-epochs', type=int, default=DEFAULT_HNM_EPOCHS, help="每一轮 HNM 附加训练的 epoch 数")
    p.add_argument('--hnm-threshold', type=float, default=DEFAULT_HNM_THRESHOLD, help="困难负例的概率下限")
    p.add_argument('--hnm-keep-easy', type=float, default=DEFAULT_HNM_KEEP_EASY, help="保留的简单负例比例")

    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / 'logs'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | seed={args.seed} | base_epochs={args.base_epochs} | "
                f"hnm_rounds={args.hnm_rounds} | hnm_epochs={args.hnm_epochs} | "
                f"bs={args.batch_size} | lr={args.lr} | layer_size={args.layer_size}")

    ecc_fa = Path(args.ecc_fa) if args.ecc_fa else PROCESSED_DATA_DIR / "eccDNA.fa"
    other_fa = Path(args.other_fa) if args.other_fa else PROCESSED_DATA_DIR / "otherDNA.fa"

    # 初始化全量数据集与第一次的 Dataloader
    train_loader, val_loader, full_ds, train_sub, val_sub = build_train_val_loaders(
        batch_size=args.batch_size, val_split=args.val_split, seed=args.seed,
        num_workers=args.num_workers, balanced=args.balanced,
        ecc_path=ecc_fa, other_path=other_fa)

    tr_lab = subset_labels(full_ds, train_sub)
    va_lab = subset_labels(full_ds, val_sub)
    logger.info(f"Initial Train class counts: {np.bincount(tr_lab, minlength=2).tolist()}")
    logger.info(f"Validation class counts (Invariant): {np.bincount(va_lab, minlength=2).tolist()}")

    model = ResNetSelfAttention(layer_size=args.layer_size).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    best_auc_overall = -1.0
    best_acc_overall = -1.0
    global_epoch_counter = 0
    global_step = 0

    total_stages = args.hnm_rounds + 1

    for stage in range(total_stages):
        is_hnm_stage = (stage > 0)
        epochs_for_stage = args.hnm_epochs if is_hnm_stage else args.base_epochs
        stage_name = f"HNM Round {stage}" if is_hnm_stage else "Base Training"

        logger.info(f"\n{'='*70}")
        logger.info(f"--- Starting Stage {stage + 1}/{total_stages}: {stage_name} ---")
        logger.info(f"{'='*70}")

        # 如果是 HNM 阶段，加载最佳模型，进行数据重采样
        if is_hnm_stage:
            best_model_path = out_dir / "best_model.pth"
            if best_model_path.exists():
                logger.info(f"Loading best model from previous stage: {best_model_path.name}")
                ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt['state_dict'])
            else:
                logger.warning("Best model not found! Proceeding with current weights.")
            
            # 使用最强模型执行困难负例挖掘
            train_loader, train_sub = perform_hnm(
                model=model, full_dataset=full_ds, current_train_subset=train_sub,
                batch_size=args.batch_size, num_workers=args.num_workers, device=device,
                threshold=args.hnm_threshold, keep_easy_ratio=args.hnm_keep_easy,
                balanced=args.balanced, seed=args.seed + stage
            )

        # 每个阶段开始时，重新初始化优化器和调度器
        # 原因：HNM 彻底改变了 Loss 景观，网络需要原始的学习率动力来调整分类边界
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma) \
            if args.step_size > 0 else None

        for epoch in range(1, epochs_for_stage + 1):
            global_epoch_counter += 1
            model.train()
            running_loss, correct, total = 0.0, 0, 0

            for inputs, labels in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                raw_loss = float(loss.item())

                if args.flooding_b > 0:
                    loss = (loss - args.flooding_b).abs() + args.flooding_b

                loss.backward()
                optimizer.step()

                running_loss += raw_loss
                total += labels.size(0)
                correct += int((outputs.argmax(1) == labels).sum().item())

                global_step += 1
                if global_step % 100 == 0:
                    writer.add_scalar('Loss/train_step', raw_loss, global_step)

            if scheduler is not None:
                scheduler.step()

            train_acc = correct / max(total, 1)
            train_loss = running_loss / max(len(train_loader), 1)
            vm = evaluate(model, val_loader, criterion, device, flooding_b=0.0)

            cur_lr = optimizer.param_groups[0]['lr']
            logger.info(f"[{stage_name}] Epoch [{epoch}/{epochs_for_stage}] (Global {global_epoch_counter}) "
                        f"lr={cur_lr:.2e} | train loss={train_loss:.4f} acc={train_acc:.4f} | "
                        f"val loss={vm['loss']:.4f} acc={vm['acc']:.4f} auc={vm['auc']:.4f} f1={vm['f1']:.4f}")

            writer.add_scalar('Loss/train', train_loss, global_epoch_counter)
            writer.add_scalar('Accuracy/train', train_acc, global_epoch_counter)
            writer.add_scalar('Loss/val', vm['loss'], global_epoch_counter)
            writer.add_scalar('Accuracy/val', vm['acc'], global_epoch_counter)
            writer.add_scalar('AUC/val', vm['auc'], global_epoch_counter)
            writer.add_scalar('LR', cur_lr, global_epoch_counter)

            torch.save({'epoch': global_epoch_counter, 'state_dict': model.state_dict(),
                        'val_acc': vm['acc'], 'val_auc': vm['auc'],
                        'label_convention': 'eccDNA=1, otherDNA=0',
                        'layer_size': args.layer_size},
                       out_dir / "last_model.pth")

            # 跨阶段维护全局最优指标，保证最终输出的始终是在恒定验证集上最好的模型
            score = vm['auc'] if vm['auc'] > 0 else vm['acc']
            if score > best_auc_overall:
                best_auc_overall = score
                best_acc_overall = vm['acc']
                torch.save({'epoch': global_epoch_counter, 'state_dict': model.state_dict(),
                            'val_acc': vm['acc'], 'val_auc': vm['auc'],
                            'label_convention': 'eccDNA=1, otherDNA=0',
                            'layer_size': args.layer_size},
                           out_dir / "best_model.pth")
                logger.info(f"  -> Saved NEW GLOBAL BEST model (score={score:.4f}, acc={vm['acc']:.4f})")

    writer.close()
    logger.info(f"\nAll Training Stages Finished. "
                f"Global Best Score (AUC) = {best_auc_overall:.4f}, Acc = {best_acc_overall:.4f}")
    logger.info(f"Models saved in {out_dir}")

    return {
        'best_auc': best_auc_overall,
        'best_acc': best_acc_overall,
    }


if __name__ == '__main__':
    main()