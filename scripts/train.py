"""
scripts/train.py - 模型训练入口

相对原版 train.py 的变更:
  * 标签统一为 torch.long + CrossEntropyLoss (原版混用 .float()/.long())
  * ROC/AUC 改用 sklearn (原版手写 50 档滑块统计, 精度低且 O(batch*50) 慢)
  * 保留 Flooding 正则化与 StepLR 衰减, 参数全部可通过 CLI/config 配置
  * 数据来自 data/processed/{eccDNA,otherDNA}.fa, 固定种子划分
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

from config import (MODEL_DIR, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE,
                    DEFAULT_WEIGHT_DECAY, DEFAULT_STEP_SIZE, DEFAULT_GAMMA,
                    DEFAULT_FLOODING_B, RANDOM_SEED, NUM_WORKERS, PROCESSED_DATA_DIR)
from src.model import ResNetSelfAttention
from src.dataloader import build_train_val_loaders, subset_labels, ECC_LABEL
from src.utils import setup_logger

logger = setup_logger('train')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device, flooding_b: float):
    """在验证集上评估, 返回指标字典"""
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
    p = argparse.ArgumentParser(description='Train MicroDNA classifier',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    p.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE)
    p.add_argument('--weight-decay', type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument('--step-size', type=int, default=DEFAULT_STEP_SIZE, help="StepLR 衰减间隔 (epoch)")
    p.add_argument('--gamma', type=float, default=DEFAULT_GAMMA, help="StepLR 衰减率")
    p.add_argument('--flooding-b', type=float, default=DEFAULT_FLOODING_B,
                   help="Flooding 正则化参数 b; 0 表示关闭")
    p.add_argument('--val-split', type=float, default=0.2)
    p.add_argument('--balanced', action='store_true', help="训练集使用类别平衡采样")
    p.add_argument('--num-workers', type=int, default=NUM_WORKERS)
    p.add_argument('--seed', type=int, default=RANDOM_SEED)
    p.add_argument('--output-dir', type=str, default=str(MODEL_DIR))
    p.add_argument('--ecc-excel-fasta', dest='ecc_fa', default=None)
    p.add_argument('--other-fasta', dest='other_fa', default=None)
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / 'logs'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | seed={args.seed} | epochs={args.epochs} | "
                f"bs={args.batch_size} | lr={args.lr} | flooding_b={args.flooding_b}")

    ecc_fa = Path(args.ecc_fa) if args.ecc_fa else PROCESSED_DATA_DIR / "eccDNA.fa"
    other_fa = Path(args.other_fa) if args.other_fa else PROCESSED_DATA_DIR / "otherDNA.fa"

    train_loader, val_loader, full_ds, train_sub, val_sub = build_train_val_loaders(
        batch_size=args.batch_size, val_split=args.val_split, seed=args.seed,
        num_workers=args.num_workers, balanced=args.balanced,
        ecc_path=ecc_fa, other_path=other_fa)

    tr_lab = subset_labels(full_ds, train_sub)
    va_lab = subset_labels(full_ds, val_sub)
    logger.info(f"Train class counts (other/ecc): {np.bincount(tr_lab, minlength=2).tolist()}")
    logger.info(f"Val   class counts (other/ecc): {np.bincount(va_lab, minlength=2).tolist()}")

    model = ResNetSelfAttention().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma) \
        if args.step_size > 0 else None

    best_auc, best_acc, best_epoch = -1.0, -1.0, -1
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            raw_loss = float(loss.item())

            if args.flooding_b > 0:                       # Flooding: |loss - b| + b
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
        logger.info(f"Epoch [{epoch}/{args.epochs}] lr={cur_lr:.2e} | "
                    f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
                    f"val loss={vm['loss']:.4f} acc={vm['acc']:.4f} "
                    f"auc={vm['auc']:.4f} f1={vm['f1']:.4f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', vm['loss'], epoch)
        writer.add_scalar('Accuracy/val', vm['acc'], epoch)
        writer.add_scalar('AUC/val', vm['auc'], epoch)
        writer.add_scalar('F1/val', vm['f1'], epoch)
        writer.add_scalar('LR', cur_lr, epoch)

        # 保存最新权重 (state_dict, 推荐格式)
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                    'val_acc': vm['acc'], 'val_auc': vm['auc'],
                    'label_convention': 'eccDNA=1, otherDNA=0'},
                   out_dir / "last_model.pth")

        # 以 AUC 为主指标保存 best (AUC 不可用时退回 acc)
        score = vm['auc'] if vm['auc'] > 0 else vm['acc']
        if score > best_auc:
            best_auc, best_acc, best_epoch = score, vm['acc'], epoch
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                        'val_acc': vm['acc'], 'val_auc': vm['auc'],
                        'label_convention': 'eccDNA=1, otherDNA=0'},
                       out_dir / "best_model.pth")
            logger.info(f"  -> Saved best model (score={score:.4f}, acc={vm['acc']:.4f})")

    writer.close()
    logger.info(f"Training finished. Best epoch={best_epoch}, "
                f"score={best_auc:.4f}, val_acc={best_acc:.4f}")
    logger.info(f"Models saved in {out_dir}")


if __name__ == '__main__':
    main()