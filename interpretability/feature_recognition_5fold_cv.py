#!/usr/bin/env python3
"""
5-fold sub-CV pipeline on train_fold1 (1552 cases).

Steps:
  1. Create 5 stratified sub-folds from train_fold1.txt
  2. Fine-tune Image-only model on each sub-train (~1242 cases), evaluate on sub-val (~310)
  3. Aggregate OOF predictions (covering all 1552 cases)
  4. Compute per-feature accuracy table

Usage:
  # Step 1: Create sub-fold splits
  python shap_outputs/train_5fold_subcv.py --mode split

  # Step 2: Train + predict (all 5 folds)
  python shap_outputs/train_5fold_subcv.py --mode train_predict --epochs 50

  # Step 3: Compute accuracy from OOF predictions
  python shap_outputs/train_5fold_subcv.py --mode accuracy

  # Or run all steps:
  python shap_outputs/train_5fold_subcv.py --mode all --epochs 50
"""

import os
import sys
import argparse
import warnings
import subprocess
import shutil
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, 'classification_models/main')
warnings.filterwarnings('ignore')

# === Constants ===
TRAIN_ANNO = 'classification_models/data/labels/train_fold1.txt'
LABELS_DIR = 'classification_models/data/labels'
DATA_DIR = 'classification_models/data/images'
SUBCV_DIR = 'classification_models/data/labels/subcv'
OUTPUT_DIR = 'outputs/feature_recognition_cv'
BASE_CHECKPOINT = 'checkpoints/image_only/model_best.pth.tar'
NUM_FOLDS = 5
SEED = 42

IMAGING_FEATURE_NAMES_EN = [
    'Nonrim arterial phase hyperenhancement',
    'Rim APHE',
    'Nonperipheral washout',
    'Peripheral "Washout"',
    'Corona Enhancement',
    'Enhancing capsule',
    'Nonenhancing capsule',
    'Peripheral Discontinuous Nodular Enhancement',
    'Progressive Enhancement',
    'Centripetal Enhancement',
    'Parallels blood pool enhancement',
    'Uniform AP Enhancement',
    'Uniform PVP Enhancement',
    'Uniform DP Enhancement',
    'Necrosis or severe ischemia',
    'Blood Products in Mass',
    'Nodule-in-nodule architecture',
    'Mosaic Architecture',
    'Delayed Central Enhancement',
    'Infiltrative appearance',
    'Portal venous phase peritumoral hypoenhancement',
    'Fat in Mass, more than Liver',
    'Fat Sparing in Solid Mass',
    'Intratumoral artery',
]
NUM_FEATURES = 24


def read_anno_lines(anno_path):
    """Read annotation file, return list of (case_name, full_line) tuples."""
    lines = []
    with open(anno_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            case_name = parts[0].strip()
            if case_name and case_name != 'casename':
                lines.append((case_name, line))
    return lines


def get_label(case_line):
    """Extract pathology label from annotation line (column 2)."""
    parts = case_line.split('\t')
    if len(parts) >= 2:
        return parts[1].strip()
    return 'unknown'


def create_splits():
    """Create 5 stratified sub-fold splits from train_fold1.txt."""
    from sklearn.model_selection import StratifiedKFold

    os.makedirs(SUBCV_DIR, exist_ok=True)
    entries = read_anno_lines(TRAIN_ANNO)
    print(f"Total entries in {TRAIN_ANNO}: {len(entries)}")

    case_names = [e[0] for e in entries]
    labels = [get_label(e[1]) for e in entries]

    # Count class distribution
    from collections import Counter
    label_counts = Counter(labels)
    print(f"Class distribution: {dict(label_counts)}")

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(case_names, labels), 1):
        train_file = os.path.join(SUBCV_DIR, f'sub_train_fold{fold_idx}.txt')
        val_file = os.path.join(SUBCV_DIR, f'sub_val_fold{fold_idx}.txt')

        with open(train_file, 'w') as f:
            for i in train_indices:
                f.write(entries[i][1] + '\n')
        with open(val_file, 'w') as f:
            for i in val_indices:
                f.write(entries[i][1] + '\n')

        train_labels = [labels[i] for i in train_indices]
        val_labels = [labels[i] for i in val_indices]
        print(f"  Fold {fold_idx}: train={len(train_indices)}, val={len(val_indices)}")
        print(f"    Train dist: {dict(Counter(train_labels))}")
        print(f"    Val dist:   {dict(Counter(val_labels))}")

    print(f"\nSplits saved to {SUBCV_DIR}/")


def train_and_predict_fold(fold, epochs, batch_size, lr):
    """Fine-tune and predict for a single sub-fold."""
    train_file = os.path.join(SUBCV_DIR, f'sub_train_fold{fold}.txt')
    val_file = os.path.join(SUBCV_DIR, f'sub_val_fold{fold}.txt')
    ckpt_dir = os.path.join(OUTPUT_DIR, f'subcv_fold{fold}')
    pred_file = os.path.join(ckpt_dir, 'oof_feat_pred.npy')

    if os.path.exists(pred_file):
        print(f"  Fold {fold}: OOF predictions already exist, skipping")
        return np.load(pred_file), np.load(pred_file.replace('feat_pred', 'feat_true'))

    os.makedirs(ckpt_dir, exist_ok=True)

    # === Step A: Fine-tune from Image-only model checkpoint ===
    print(f"\n{'='*60}")
    print(f"Fold {fold}: Training")
    print(f"  Train: {train_file}")
    print(f"  Val:   {val_file}")
    print(f"  Output: {ckpt_dir}")
    print(f"{'='*60}")

    train_cmd = [
        sys.executable, 'classification_models/main/train.py',
        '--data_dir', DATA_DIR,
        '--train_anno_file', train_file,
        '--val_anno_file', val_file,
        '--model', 'uniformer_small_IL_features',
        '--num-classes', '3',
        '--num-feature-classes', '24',
        '--feature-fusion', 'hierarchical_simple',
        '--label-mode', 'original',
        '--lr', str(lr),
        '--lr-loss-weight', '0.1',
        '--warmup-epochs', '5',
        '--batch-size', str(batch_size),
        '--epochs', str(epochs),
        '--initial-checkpoint', BASE_CHECKPOINT,
        '--output', ckpt_dir,
        '--workers', '8',
        '--img_size', '20', '96', '96',
        '--crop_size', '10', '80', '80',
        '--flip_prob', '0.5',
        '--reprob', '0.25',
        '--rcprob', '0.25',
        '--angle', '45',
        '--seed', str(SEED),
    ]

    print(f"  CMD: {' '.join(train_cmd)}")
    result = subprocess.run(train_cmd, cwd=os.getcwd())
    if result.returncode != 0:
        print(f"  ERROR: Training failed for fold {fold}")
        return None, None

    # === Step B: Run OOF prediction on sub-val ===
    best_ckpt = os.path.join(ckpt_dir, 'model_best.pth.tar')
    if not os.path.exists(best_ckpt):
        # Checkpoint may be in a model-name subdirectory
        for sub in os.listdir(ckpt_dir):
            sub_path = os.path.join(ckpt_dir, sub, 'model_best.pth.tar')
            if os.path.isfile(sub_path):
                best_ckpt = sub_path
                break
        else:
            # Fallback: find latest .pth.tar
            for root, dirs, files in os.walk(ckpt_dir):
                for f in sorted(files):
                    if f.endswith('.pth.tar'):
                        best_ckpt = os.path.join(root, f)

    print(f"\n  Fold {fold}: OOF prediction with {best_ckpt}")

    from datasets.mp_liver_dataset import MultiPhaseLiverDataset
    from torch.utils.data import DataLoader
    import copy

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import models  # noqa: F401  -- register custom model
    from timm.models import create_model
    model = create_model('uniformer_small_IL_features',
                         pretrained=False, num_classes=3,
                         num_feature_classes=24,
                         feature_fusion='hierarchical_simple')
    ckpt = torch.load(best_ckpt, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    print(f"  Loaded: {best_ckpt}")

    # Create dataset for sub-val
    args = argparse.Namespace(
        val_anno_file=val_file,
        data_dir=DATA_DIR,
        skip_cases_file='classification_models/data/skip_cases.txt',
        data1_file='classification_models/data/data1.xlsx',
        case_mapping_file='',
        label_mode='original',
        img_size=[20, 96, 96],
        crop_size=[10, 80, 80],
        val_transform_list=['center_crop'],
    )
    dataset = MultiPhaseLiverDataset(args, is_training=False)
    print(f"  Sub-val dataset: {len(dataset)} samples")

    # Extract predictions
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=False)
    all_feat, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(device)
            x = model.forward_features(inputs)
            gap = x.flatten(2).mean(-1)
            feat_probs = torch.sigmoid(model.feature_fc_head(gap))
            all_feat.append(feat_probs.cpu())
            all_labels.append(batch[1])
    data = {
        'feat': torch.cat(all_feat).numpy(),
        'labels': torch.cat(all_labels).numpy(),
    }

    # Extract feature ground truth
    case_names = dataset.get_case_names()
    feat_gt = np.zeros((len(dataset), NUM_FEATURES), dtype=np.float32)
    for i in range(len(dataset)):
        features = dataset.case_to_features.get(case_names[i])
        if features is not None:
            feat_len = min(len(features), NUM_FEATURES)
            feat_gt[i, :feat_len] = features[:feat_len].numpy()

    # Save
    np.save(pred_file, data['feat'])
    np.save(pred_file.replace('feat_pred', 'feat_true'), feat_gt)
    np.save(pred_file.replace('feat_pred', 'case_names'), np.array(case_names))
    print(f"  Saved OOF predictions: {pred_file}")

    del model, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return data['feat'], feat_gt


def compute_accuracy():
    """Aggregate OOF predictions and compute accuracy table."""
    all_feat_pred = []
    all_feat_true = []

    for fold in range(1, NUM_FOLDS + 1):
        ckpt_dir = os.path.join(OUTPUT_DIR, f'subcv_fold{fold}')
        pred_file = os.path.join(ckpt_dir, 'oof_feat_pred.npy')
        true_file = os.path.join(ckpt_dir, 'oof_feat_true.npy')

        if not os.path.exists(pred_file):
            print(f"  WARNING: Fold {fold} predictions not found: {pred_file}")
            continue

        feat_pred = np.load(pred_file)
        feat_true = np.load(true_file)
        print(f"  Fold {fold}: {feat_pred.shape[0]} samples")
        all_feat_pred.append(feat_pred)
        all_feat_true.append(feat_true)

    if not all_feat_pred:
        print("ERROR: No OOF predictions found! Run --mode train_predict first.")
        return

    all_feat_pred = np.concatenate(all_feat_pred, axis=0)
    all_feat_true = np.concatenate(all_feat_true, axis=0)
    total = all_feat_pred.shape[0]
    print(f"\nTotal OOF samples: {total}")

    # Compute accuracy
    pred_binary = (all_feat_pred >= 0.5).astype(int)
    true_binary = all_feat_true.astype(int)

    rows = []
    for i, name in enumerate(IMAGING_FEATURE_NAMES_EN):
        acc = np.mean(pred_binary[:, i] == true_binary[:, i]) * 100
        neg_count = int(np.sum(true_binary[:, i] == 0))
        pos_count = int(np.sum(true_binary[:, i] == 1))
        rows.append({
            'Imaging Feature': name,
            'Accuracy (%)': round(acc, 2),
            'No. of Lesions without Feature Present': neg_count,
            'No. of Lesions with Feature Present': pos_count,
        })

    df_acc = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, 'train_5fold_feature_accuracy.csv')
    df_acc.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {csv_path}")

    # Save Markdown
    md_path = os.path.join(OUTPUT_DIR, 'train_5fold_feature_accuracy.md')
    threshold = 80.0
    md_lines = [
        "# 训练集5折交叉验证 — 24个成像特征预测准确率\n",
        f"总样本数: {total} 例 (5-fold CV on train_fold1)\n",
        "| No. | Imaging Feature | Accuracy (%) | No. of Lesions without Feature Present | No. of Lesions with Feature Present |",
        "|---|---|---|---|---|",
    ]
    for idx, row in df_acc.iterrows():
        mark = '✓' if row['Accuracy (%)'] > threshold else '✗'
        md_lines.append(
            f"| {idx+1} | {row['Imaging Feature']} | {row['Accuracy (%)']:.2f} | "
            f"{row['No. of Lesions without Feature Present']} | "
            f"{row['No. of Lesions with Feature Present']} | {mark} |"
        )
    n_selected = int((df_acc['Accuracy (%)'] > threshold).sum())
    md_lines.append(f"\n**保留特征数 (Accuracy > {threshold}%):** {n_selected}")
    md_lines.append(f"**过滤特征数:** {NUM_FEATURES - n_selected}")

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print(f"Saved: {md_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"5-Fold Sub-CV Feature Accuracy (threshold={threshold}%)")
    print(f"{'='*60}")
    print(f"  Total OOF samples: {total}")
    print(f"\n  RETAINED (>{threshold}%):")
    for _, row in df_acc.iterrows():
        if row['Accuracy (%)'] > threshold:
            print(f"    {row['Accuracy (%)']:6.2f}%  {row['Imaging Feature']}")
    print(f"\n  EXCLUDED (≤{threshold}%):")
    for _, row in df_acc.iterrows():
        if row['Accuracy (%)'] <= threshold:
            print(f"    {row['Accuracy (%)']:6.2f}%  {row['Imaging Feature']}")
    print(f"\n  Retained: {n_selected} / {NUM_FEATURES}")


def main():
    parser = argparse.ArgumentParser(description='5-fold sub-CV pipeline on train_fold1')
    parser.add_argument('--mode', choices=['split', 'train_predict', 'accuracy', 'all'],
                        default='all', help='Pipeline mode')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Fine-tuning epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Fine-tuning learning rate (default: 5e-5)')
    args = parser.parse_args()

    if args.mode in ('split', 'all'):
        print("=" * 60)
        print("Step 1: Creating 5-fold stratified sub-splits")
        print("=" * 60)
        create_splits()

    if args.mode in ('train_predict', 'all'):
        print("\n" + "=" * 60)
        print("Step 2: Training + OOF prediction")
        print("=" * 60)
        for fold in range(1, NUM_FOLDS + 1):
            train_and_predict_fold(fold, args.epochs, args.batch_size, args.lr)

    if args.mode in ('accuracy', 'all'):
        print("\n" + "=" * 60)
        print("Step 3: Computing OOF accuracy table")
        print("=" * 60)
        compute_accuracy()


if __name__ == '__main__':
    main()
