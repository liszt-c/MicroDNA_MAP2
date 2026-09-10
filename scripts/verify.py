"""
scripts/verify.py - 模型评估: 准确率 / AUC / PR-AUC / 混淆矩阵 / 分类报告 + 图表

整合并替代原版 verification.py 与 ROC_draw.py。
标签约定: 0 = otherDNA, 1 = eccDNA (与训练一致)。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve, auc, precision_recall_curve,
                             f1_score, precision_score, recall_score, accuracy_score,
                             confusion_matrix, classification_report)

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (MODEL_DIR, METRICS_DIR, DEFAULT_BATCH_SIZE, NUM_WORKERS, PROCESSED_DATA_DIR)
from src.model import ResNetSelfAttention
from src.dataloader import build_full_loader, ECC_LABEL
from src.utils import setup_logger

logger = setup_logger('verify')
CLASS_NAMES = ['otherDNA', 'eccDNA']     # 索引 0 / 1


def torch_load_compat(path: Path, device):
    """兼容 torch>=2.6 (weights_only 默认 True) 与旧版整模型保存格式"""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(model_path: Path, device):
    model = ResNetSelfAttention()
    ckpt = torch_load_compat(model_path, device)

    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
        meta = {k: v for k, v in ckpt.items() if k != 'state_dict'}
        if meta:
            logger.info(f"Checkpoint meta: {meta}")
    elif isinstance(ckpt, dict):
        state, meta = ckpt, {}
    else:
        # 旧版 torch.save(model) 保存了整个模型对象
        logger.warning("Checkpoint contains a full model object (legacy format); using it directly.")
        ckpt.to(device)
        ckpt.eval()
        return ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Missing keys when loading state_dict: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys when loading state_dict: {unexpected}")
    model.to(device)
    model.eval()
    logger.info(f"Model loaded from {model_path}")
    return model


def collect_predictions(model, loader, device):
    probs, labels = [], []
    with torch.no_grad():
        for i, (inputs, lbs) in enumerate(loader):
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            probs.append(torch.softmax(outputs, dim=1)[:, ECC_LABEL].cpu().numpy())
            labels.append(lbs.numpy())
            if (i + 1) % 20 == 0:
                logger.info(f"  ... {i + 1}/{len(loader)} batches")
    y_prob = np.concatenate(probs) if probs else np.array([], dtype=np.float32)
    y_true = np.concatenate(labels) if labels else np.array([], dtype=np.int64)
    return y_true, y_prob


def plot_metrics(y_true, y_prob, y_pred, cm, report_txt, threshold, auc_roc, pr_auc, out_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec_c, rec_c, _ = precision_recall_curve(y_true, y_prob)

    fig = plt.figure(figsize=(14, 11))

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc_roc:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=1.5, ls='--')
    ax1.set_xlim([0.0, 1.0]); ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'ROC Curve (threshold={threshold})')
    ax1.legend(loc='lower right')

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(rec_c, prec_c, color='steelblue', lw=2, label=f'PR (AUC = {pr_auc:.4f})')
    ax2.set_xlim([0.0, 1.0]); ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve'); ax2.legend(loc='lower left')

    ax3 = plt.subplot(2, 2, 3)
    im = ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax3.set_title(f'Confusion Matrix (threshold={threshold})')
    plt.colorbar(im, ax=ax3, fraction=0.046)
    ticks = np.arange(len(CLASS_NAMES))
    ax3.set_xticks(ticks); ax3.set_yticks(ticks)
    ax3.set_xticklabels(CLASS_NAMES); ax3.set_yticklabels(CLASS_NAMES)
    ax3.set_xlabel('Predicted'); ax3.set_ylabel('True')
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax3.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')

    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    ax4.text(0.0, 0.95, report_txt, fontfamily='monospace', fontsize=9, va='top')
    ax4.set_title('Classification Report')

    plt.tight_layout()
    out_png = out_dir / f"evaluation_metrics_threshold{threshold}.png"
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Figure saved -> {out_png}")
    return out_png


def main():
    p = argparse.ArgumentParser(description="Evaluate MicroDNA model",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model', default=None, help="模型路径 (默认 models/best_model.pth)")
    p.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument('--threshold', type=float, default=0.5, help="eccDNA 概率判定阈值")
    p.add_argument('--num-workers', type=int, default=NUM_WORKERS)
    p.add_argument('--ecc-fasta', default=None)
    p.add_argument('--other-fasta', default=None)
    p.add_argument('--output-dir', default=str(METRICS_DIR))
    p.add_argument('--no-plot', action='store_true')
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model) if args.model else MODEL_DIR / "best_model.pth"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}. Train it first.")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(model_path, device)

    ecc_fa = Path(args.ecc_fasta) if args.ecc_fasta else PROCESSED_DATA_DIR / "eccDNA.fa"
    other_fa = Path(args.other_fasta) if args.other_fasta else PROCESSED_DATA_DIR / "otherDNA.fa"
    loader, dataset = build_full_loader(batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        ecc_path=ecc_fa, other_path=other_fa)

    logger.info(f"Evaluating on {len(dataset)} records (full dataset) ...")
    y_true, y_prob = collect_predictions(model, loader, device)
    if y_true.size == 0:
        logger.error("No samples evaluated.")
        sys.exit(1)

    y_pred = (y_prob >= args.threshold).astype(int)
    y_pred_argmax = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    acc_argmax = accuracy_score(y_true, y_pred_argmax)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    unique = np.unique(y_true)
    if len(unique) < 2:
        logger.warning(f"Only one class present in labels {unique.tolist()}; AUC/PR-AUC undefined.")
        auc_roc, pr_auc = float('nan'), float('nan')
    else:
        auc_roc = roc_auc_score(y_true, y_prob)
        _p, _r, _t = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(_r, _p)

    report_txt = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                       labels=[0, 1], zero_division=0)
    report_argmax_txt = classification_report(y_true, y_pred_argmax, target_names=CLASS_NAMES,
                                              labels=[0, 1], zero_division=0)

    lines = [
        "=" * 64,
        f"Model              : {model_path}",
        f"Device             : {device}",
        f"Threshold          : {args.threshold}",
        f"Label convention   : 0=otherDNA, 1=eccDNA",
        "=" * 64,
        f"Total samples      : {y_true.size}",
        f"  eccDNA           : {int((y_true == 1).sum())}",
        f"  otherDNA         : {int((y_true == 0).sum())}",
        "-" * 64,
        f"Accuracy (thresh)  : {acc:.4f}",
        f"Accuracy (argmax)  : {acc_argmax:.4f}",
        f"Precision          : {prec:.4f}",
        f"Recall             : {rec:.4f}",
        f"F1                 : {f1:.4f}",
        f"Specificity        : {spec:.4f}",
        f"AUC-ROC            : {auc_roc:.4f}",
        f"AUC-PR             : {pr_auc:.4f}",
        "-" * 64,
        f"Confusion matrix   : TN={tn} FP={fp} FN={fn} TP={tp}",
        "",
        "Classification Report (threshold):",
        report_txt,
        "",
        "Classification Report (argmax):",
        report_argmax_txt,
    ]
    text = "\n".join(lines)
    print(text)

    report_path = out_dir / f"evaluation_report_threshold{args.threshold}.txt"
    report_path.write_text(text + "\n", encoding='utf-8')
    logger.info(f"Report saved -> {report_path}")

    np.savez_compressed(out_dir / f"predictions_threshold{args.threshold}.npz",
                        y_true=y_true, y_prob=y_prob, y_pred=y_pred)

    if not args.no_plot and len(unique) >= 2:
        plot_metrics(y_true, y_prob, y_pred, cm, report_txt, args.threshold,
                     auc_roc, pr_auc, out_dir)
    elif args.no_plot:
        logger.info("Plotting skipped (--no-plot).")


if __name__ == '__main__':
    main()