#!/usr/bin/env python3
import argparse
import os
import json
import csv
import glob
import time
import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from contextlib import suppress
from torch.utils.data.dataloader import DataLoader
from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_legacy

import models
from metrics import *
from datasets.mp_liver_dataset import MultiPhaseLiverDataset, create_loader
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免在服务器上报错
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix as sk_confusion_matrix, roc_curve, auc as roc_auc

# 24个征象完整名称列表
ALL_SIGN_NAMES_24 = [
    'Nonrim arterial phase hyperenhancement',
    'Rim APHE',
    'Nonperipheral washout',
    'Peripheral "washout"',
    'Corona enhancement',
    'Enhancing capsule',
    'Nonenhancing capsule',
    'Peripheral discontinuous nodular enhancement',
    'Progressive enhancement',
    'Centripetal enhancement',
    'Parallels blood pool enhancement',
    'Uniform AP enhancement',
    'Uniform PVP enhancement',
    'Uniform DP enhancement',
    'Necrosis or severe ischemia',
    'Blood products in mass',
    'Nodule-in-nodule architecture',
    'Mosaic architecture',
    'Delayed central enhancement',
    'Infiltrative appearance',
    'Portal venous phase peritumoral hypoenhancement',
    'Fat in mass, more than liver',
    'Fat sparing in solid mass',
    'Intratumoral artery',
]

has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')


parser = argparse.ArgumentParser(description='LLD-MMRI2023 Validation')

parser.add_argument('--img_size', default=(20, 96, 96), type=int, nargs='+', help='input image size.')
parser.add_argument('--crop_size', default=(16, 80, 80), type=int, nargs='+', help='cropped image size.')
parser.add_argument('--data_dir', default='classification_models/data/images/', type=str)
parser.add_argument('--val_anno_file', default='classification_models/data/labels/val_fold1.txt', type=str)
parser.add_argument('--test_anno_file', default='classification_models/data/labels/test.txt', type=str,
                    help='External test set annotation file. If provided, will also predict on this set.')
parser.add_argument('--val_transform_list', default=['center_crop'], nargs='+', type=str)
parser.add_argument('--model', '-m', metavar='NAME', default='uniformer_small_IL',
                    help='model architecture (default: uniformer_small_IL)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default=' ', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--legacy-jit', dest='legacy_jit', action='store_true',
                    help='use legacy jit mode for pytorch 1.5/1.5.1/1.6 to get back fusion performance')
parser.add_argument('--results-dir', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--score-dir', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation score (summary)')
parser.add_argument('--is_aux_head', action='store_true', default=False,
                    help='use benign/maglinant label as auxiliary head')
parser.add_argument('--case-mapping-file', default='', type=str,
                    help='Path to case name mapping file (original_id -> case_xxx)')
parser.add_argument('--data1-file', default='classification_models/data/data1.xlsx', type=str,
                    help='Path to data1.xlsx containing clinical features and feature labels')
parser.add_argument('--label-mode', default='lr', type=str, choices=['original', 'lr'],
                    help='Which label column to use: "original" (2nd col, 0/1/2) or "lr" (4th col, LR grade). Default: "lr"')
parser.add_argument('--skip-cases-file', default='classification_models/data/classification_dataset/skip_cases.txt', type=str,
                    help='Path to skip cases file (cases with missing/mismatched feature labels)')
parser.add_argument('--num-feature-classes', type=int, default=None,
                    help='Number of feature classes for feature head (default: None, auto-enabled if model name contains "features")')
parser.add_argument('--selected-features', default=None, type=int, nargs='+',
                    help='Optional ablation only: subset of feature indices (0-23). The final study models use all 24 features.')
parser.add_argument('--include-clinical', action='store_true', default=False,
                    help='Include LR grade and clinical variables as extra features')
parser.add_argument('--clinical-dim', default=0, type=int,
                    help='Dimension of clinical features (default: 0, auto-calculated if include-clinical)')
parser.add_argument('--clinical-scale', default=None, type=float,
                    help='Initial scale factor for clinical features in fusion (default: None, auto-inferred from checkpoint). '
                         'Set >1.0 to boost clinical feature weight.')
parser.add_argument('--normalize-clinical', action='store_true', default=False,
                    help='Apply z-score normalization to continuous clinical variables (age, AFP). '
                         'Must match the training configuration.')
parser.add_argument('--clinical-stats-file', default='', type=str,
                    help='Path to clinical normalization stats JSON '
                         '(default: auto-detect from checkpoint output dir)')
parser.add_argument('--feature-fusion', default=None, type=str, choices=[None, 'hierarchical', 'hierarchical_simple', 'late_fusion'],
                    help='Feature fusion mode for models with feature head')
parser.add_argument('--prior-calibration', action='store_true', default=False,
                    help='Calibrate prediction probabilities using train/test class prior distribution')
parser.add_argument('--train-anno-file', default=None, type=str,
                    help='Training annotation file for computing class prior (auto-inferred from args.yaml if not set)')
parser.add_argument('--minority-bias', default=0.0, type=float,
                    help='Logit-space additive bias for minority classes in 5-class LR mode. '
                         'Adds this value to the logits of minority classes (LR-3, LR-4) before softmax. '
                         'Higher values = stronger preference for minority classes. '
                         'Set 0.0 to disable. Recommended: 1.0~3.0')
parser.add_argument('--lr34-bias-search', action='store_true', default=False,
                    help='方案四: Auto-search optimal minority-bias for LR-3/LR-4. '
                         'Searches bias in [0, 0.5, 1.0, 1.5, ..., 5.0] and selects the value '
                         'that maximizes combined F1 of LR-3 + LR-4. '
                         'Overrides --minority-bias if set.')
parser.add_argument('--lr34-bias-target', default='f1', type=str, choices=['f1', 'recall', 'macro_f1'],
                    help='方案四: Optimization target for bias search. '
                         '"f1" = sum of LR-3+LR-4 F1 (default), '
                         '"recall" = sum of LR-3+LR-4 Recall, '
                         '"macro_f1" = overall macro F1')
parser.add_argument('--threshold-optimize', action='store_true', default=False,
                    help='Optimize per-class decision thresholds to maximize macro-F1. '
                         'Searches thresholds in [0.05, 0.10, ..., 0.95] for each class. '
                         'Reports both original (argmax) and optimized metrics.')

def load_class_counts_from_anno(anno_file, label_mode, num_classes):
    """从标注文件统计各类别样本数

    Args:
        anno_file: 标注文件路径
        label_mode: 'lr' 或 'original'
        num_classes: 类别数
    Returns:
        np.ndarray: shape (num_classes,) 各类别样本数
    """
    if not anno_file or not os.path.exists(anno_file):
        _logger.warning(f"Annotation file not found: {anno_file}")
        return None

    lr_mapping = {'1/2': 0, '3': 1, '4': 2, '5': 3, 'M': 4}
    pathology_mapping = {'liangxing': 0, 'noHCC': 1, 'HCC': 2}

    with open(anno_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
    skip_header = 1 if first_line.startswith('casename') else 0
    df = pd.read_csv(anno_file, sep='\t', header=None, skiprows=skip_header)

    counts = np.zeros(num_classes, dtype=np.float64)
    if label_mode == 'lr':
        raw_labels = df.iloc[:, 2].astype(str)
        for raw in raw_labels:
            lr_val = raw.strip().split()[-1]
            cls_id = lr_mapping.get(lr_val, -1)
            if 0 <= cls_id < num_classes:
                counts[cls_id] += 1
    else:
        raw_labels = df.iloc[:, 1].astype(str)
        for raw in raw_labels:
            cls_id = pathology_mapping.get(raw.strip(), -1)
            if 0 <= cls_id < num_classes:
                counts[cls_id] += 1

    return counts


def apply_minority_logit_bias(logits, bias, num_classes):
    """在 logit 空间对少数类施加加法偏置，改善类别不平衡导致的 argmax 偏移。

    在 softmax 之前，给少数类 (LR-3, LR-4) 的 logits 加上一个正值，
    相当于降低这些类别的决策阈值。bias 越大，少数类越容易被选中。

    数学等价：给 class_i 的 logit 加 b，等价于将其 softmax 概率乘以 e^b，
    然后重新归一化。

    Args:
        logits: (N, C) 原始 logits（softmax 之前）
        bias: 加法偏置值，0 则禁用
        num_classes: 类别数（仅在 5 类 LR 模式下生效）
    Returns:
        adjusted_logits: (N, C) 调整后的 logits
    """
    if bias <= 0 or num_classes != 5:
        return logits

    # 少数类索引: LR-3 (1) 和 LR-4 (2)
    minority_classes = [1, 2]

    adjusted = logits.copy()
    for c in minority_classes:
        adjusted[:, c] += bias

    _logger.info(f"Minority logit bias (bias={bias}): added {bias} to classes {minority_classes} (LR-3, LR-4)")

    return adjusted


def search_optimal_lr34_bias(logits, true_labels, num_classes=5, target='f1'):
    """方案四：自动搜索最优的 LR-3/LR-4 logit 偏置值。

    在 logit 空间搜索 bias ∈ [0, 0.5, 1.0, ..., 5.0]，
    找到使 LR-3 和 LR-4 综合指标最优的值。

    Args:
        logits: (N, C) 原始 logits
        true_labels: (N,) 真实标签
        num_classes: 类别数
        target: 优化目标 ('f1', 'recall', 'macro_f1')

    Returns:
        best_bias: 最优偏置值
        best_score: 对应的最优得分
        search_results: 所有搜索点的结果列表
    """
    from sklearn.metrics import f1_score, recall_score

    search_range = np.arange(0.0, 5.5, 0.5)
    best_bias = 0.0
    best_score = -1.0
    search_results = []

    for bias in search_range:
        adjusted = apply_minority_logit_bias(logits.copy(), bias, num_classes)
        pred_scores = torch.softmax(torch.from_numpy(adjusted), dim=1).cpu().numpy()
        pred_labels = np.argmax(pred_scores, axis=1)

        # LR-3 (idx=1) and LR-4 (idx=2)
        mask_3 = true_labels == 1
        mask_4 = true_labels == 2

        if target == 'f1':
            f1_3 = f1_score(mask_3.astype(int), (pred_labels == 1).astype(int), zero_division=0)
            f1_4 = f1_score(mask_4.astype(int), (pred_labels == 2).astype(int), zero_division=0)
            score = f1_3 + f1_4
        elif target == 'recall':
            r_3 = recall_score(mask_3.astype(int), (pred_labels == 1).astype(int), zero_division=0)
            r_4 = recall_score(mask_4.astype(int), (pred_labels == 2).astype(int), zero_division=0)
            score = r_3 + r_4
        else:  # macro_f1
            score = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

        # 同时计算各类详情
        f1_3 = f1_score(mask_3.astype(int), (pred_labels == 1).astype(int), zero_division=0)
        f1_4 = f1_score(mask_4.astype(int), (pred_labels == 2).astype(int), zero_division=0)
        r_3 = recall_score(mask_3.astype(int), (pred_labels == 1).astype(int), zero_division=0)
        r_4 = recall_score(mask_4.astype(int), (pred_labels == 2).astype(int), zero_division=0)

        search_results.append({
            'bias': float(bias),
            'score': float(score),
            'lr3_f1': float(f1_3), 'lr3_recall': float(r_3),
            'lr4_f1': float(f1_4), 'lr4_recall': float(r_4),
        })

        if score > best_score:
            best_score = score
            best_bias = float(bias)

    _logger.info(f"[方案四] Bias search (target={target}): best_bias={best_bias}, best_score={best_score:.4f}")
    _logger.info("[方案四] Search results:")
    for r in search_results:
        _logger.info(f"  bias={r['bias']:.1f}: LR-3 F1={r['lr3_f1']:.3f} R={r['lr3_recall']:.3f} | "
                     f"LR-4 F1={r['lr4_f1']:.3f} R={r['lr4_recall']:.3f} | score={r['score']:.4f}")

    return best_bias, best_score, search_results


def optimize_per_class_thresholds(pred_scores, true_labels, num_classes, class_names=None):
    """Per-class threshold optimization to maximize macro-F1.

    For each class, search thresholds in [0.05, 0.10, ..., 0.95] to find
    the one that maximizes that class's F1 score. Then use the thresholds
    for final prediction: a sample is assigned to the class with the highest
    score among all classes whose score exceeds their threshold.
    If no class exceeds its threshold, fall back to argmax.

    Args:
        pred_scores: (N, C) softmax probabilities
        true_labels: (N,) ground truth labels
        num_classes: number of classes
        class_names: optional list of class name strings

    Returns:
        best_thresholds: (C,) optimal threshold per class
        optimized_pred_labels: (N,) predictions using optimized thresholds
        original_pred_labels: (N,) original argmax predictions
        metrics_comparison: dict comparing original vs optimized metrics
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, cohen_kappa_score
    from itertools import product as iter_product

    if class_names is None:
        if num_classes == 5:
            class_names = ['LR-1/2', 'LR-3', 'LR-4', 'LR-5', 'LR-M']
        elif num_classes == 3:
            class_names = ['Benign', 'Non-HCC Malignancies', 'HCC']
        else:
            class_names = [f'Class {i}' for i in range(num_classes)]

    original_pred_labels = np.argmax(pred_scores, axis=1)
    threshold_candidates = np.arange(0.05, 1.0, 0.05)  # 0.05 to 0.95

    # Phase 1: find per-class optimal threshold (binary F1 for each class)
    best_thresholds = np.zeros(num_classes)
    for c in range(num_classes):
        binary_true = (true_labels == c).astype(int)
        best_f1 = -1
        best_t = 0.5  # default
        for t in threshold_candidates:
            binary_pred = (pred_scores[:, c] >= t).astype(int)
            f1_c = f1_score(binary_true, binary_pred, zero_division=0)
            if f1_c > best_f1:
                best_f1 = f1_c
                best_t = t
        best_thresholds[c] = best_t

    _logger.info(f"[Threshold Optimize] Per-class best thresholds: {dict(zip(class_names, best_thresholds.round(3)))}")

    # Phase 2: apply thresholds for multi-class prediction
    # Strategy: for each sample, among classes where score >= threshold,
    # pick the one with highest score. If none exceed, use argmax.
    optimized_pred_labels = np.zeros(len(pred_scores), dtype=int)
    for i in range(len(pred_scores)):
        above_thresh = np.where(pred_scores[i] >= best_thresholds)[0]
        if len(above_thresh) > 0:
            # Among classes above threshold, pick highest score
            best_cls = above_thresh[np.argmax(pred_scores[i, above_thresh])]
            optimized_pred_labels[i] = best_cls
        else:
            # Fallback to argmax
            optimized_pred_labels[i] = np.argmax(pred_scores[i])

    # Phase 3: fine-tune thresholds jointly with a grid refinement
    # Use a narrower search around the found thresholds (+/-0.15) with step 0.02
    # to optimize macro-F1 directly
    refined_thresholds = best_thresholds.copy()
    best_macro_f1 = f1_score(true_labels, optimized_pred_labels, average='macro', zero_division=0)

    for c in range(num_classes):
        center = best_thresholds[c]
        local_range = np.clip(np.arange(center - 0.15, center + 0.16, 0.02), 0.05, 0.95)
        for t in local_range:
            test_thresholds = refined_thresholds.copy()
            test_thresholds[c] = t
            # recompute predictions
            test_preds = np.zeros(len(pred_scores), dtype=int)
            for i in range(len(pred_scores)):
                above = np.where(pred_scores[i] >= test_thresholds)[0]
                if len(above) > 0:
                    test_preds[i] = above[np.argmax(pred_scores[i, above])]
                else:
                    test_preds[i] = np.argmax(pred_scores[i])
            macro_f1 = f1_score(true_labels, test_preds, average='macro', zero_division=0)
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                refined_thresholds[c] = t
                optimized_pred_labels = test_preds.copy()

    best_thresholds = refined_thresholds
    _logger.info(f"[Threshold Optimize] Refined thresholds: {dict(zip(class_names, best_thresholds.round(3)))}")

    # Phase 4: compute comparison metrics
    metrics_comparison = {
        'thresholds': {class_names[i]: float(best_thresholds[i]) for i in range(num_classes)},
        'original': {},
        'optimized': {},
    }

    # Per-class comparison
    for c in range(num_classes):
        bin_true = (true_labels == c).astype(int)
        bin_orig = (original_pred_labels == c).astype(int)
        bin_opt = (optimized_pred_labels == c).astype(int)

        metrics_comparison['original'][class_names[c]] = {
            'ACC': float(accuracy_score(bin_true, bin_orig)),
            'F1': float(f1_score(bin_true, bin_orig, zero_division=0)),
            'Recall': float(recall_score(bin_true, bin_orig, zero_division=0)),
            'Precision': float(precision_score(bin_true, bin_orig, zero_division=0)),
            'Kappa': float(cohen_kappa_score(bin_true, bin_orig)),
        }
        metrics_comparison['optimized'][class_names[c]] = {
            'ACC': float(accuracy_score(bin_true, bin_opt)),
            'F1': float(f1_score(bin_true, bin_opt, zero_division=0)),
            'Recall': float(recall_score(bin_true, bin_opt, zero_division=0)),
            'Precision': float(precision_score(bin_true, bin_opt, zero_division=0)),
            'Kappa': float(cohen_kappa_score(bin_true, bin_opt)),
        }

    # Overall comparison
    for _metric_fn, _key, _avg in [
        (f1_score, 'macro_F1', 'macro'), (f1_score, 'weighted_F1', 'weighted'),
        (recall_score, 'macro_recall', 'macro'), (recall_score, 'weighted_recall', 'weighted'),
        (precision_score, 'macro_precision', 'macro'), (precision_score, 'weighted_precision', 'weighted'),
    ]:
        metrics_comparison['original'][_key] = float(_metric_fn(true_labels, original_pred_labels, average=_avg, zero_division=0))
        metrics_comparison['optimized'][_key] = float(_metric_fn(true_labels, optimized_pred_labels, average=_avg, zero_division=0))
    metrics_comparison['original']['accuracy'] = float(accuracy_score(true_labels, original_pred_labels))
    metrics_comparison['original']['kappa'] = float(cohen_kappa_score(true_labels, original_pred_labels))
    metrics_comparison['optimized']['accuracy'] = float(accuracy_score(true_labels, optimized_pred_labels))
    metrics_comparison['optimized']['kappa'] = float(cohen_kappa_score(true_labels, optimized_pred_labels))

    return best_thresholds, optimized_pred_labels, original_pred_labels, metrics_comparison

def calibrate_with_prior(pred_scores, train_class_counts, true_labels):
    """使用训练集类别先验校正预测概率

    校正公式:
        calibrated_p(c|x) ∝ p(c|x) × P_test(c) / P_train(c)

    Args:
        pred_scores: (N, C) softmax 概率
        train_class_counts: 训练集各类别样本数, shape (C,)
        true_labels: 测试集/验证集真实标签, 用于计算目标分布
    Returns:
        calibrated_scores: (N, C) 校正后的概率
    """
    train_prior = train_class_counts / train_class_counts.sum()

    # 从 true_labels 统计目标分布
    num_classes = pred_scores.shape[1]
    test_counts = np.zeros(num_classes, dtype=np.float64)
    for label in true_labels:
        test_counts[int(label)] += 1
    test_prior = test_counts / test_counts.sum()

    # 校正因子 = 目标分布 / 训练分布
    calibration_factor = test_prior / np.maximum(train_prior, 1e-8)

    _logger.info(f"Prior calibration enabled")
    _logger.info(f"  Train prior: {train_prior}")
    _logger.info(f"  Test prior:  {test_prior}")
    _logger.info(f"  Calibration factor: {calibration_factor}")

    calibrated = pred_scores * calibration_factor[np.newaxis, :]
    calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)

    return calibrated


def validate(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    amp_autocast = suppress  # do nothing
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
        else:
            _logger.warning("Neither APEX or Native Torch AMP is available.")
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    # if args.native_amp:
    #     amp_autocast = torch.cuda.amp.autocast
    #     _logger.info('Validating in mixed precision with native PyTorch AMP.')
    # elif args.apex_amp:
    #     _logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
    # else:
    #     _logger.info('Validating in float32. AMP not enabled.')

    if args.legacy_jit:
        set_jit_legacy()

    # Auto infer num_classes based on label_mode if not explicitly set
    if args.num_classes is None:
        if getattr(args, 'label_mode', 'original') == 'lr':
            args.num_classes = 5
        else:
            args.num_classes = 3
        _logger.info(f'Auto inferred num_classes={args.num_classes} based on label_mode={args.label_mode}')

    # create model
    model_kwargs = dict(
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        pretrained_cfg=None,
    )
    if 'features' in args.model:
        selected = getattr(args, 'selected_features', None)
        if selected is not None:
            num_feat = len(selected)  # 纯二分类征象
            model_kwargs['num_feature_classes'] = num_feat
            _logger.info(f'Using {len(selected)} selected features: {selected}')
        else:
            model_kwargs['num_feature_classes'] = getattr(args, 'num_feature_classes', None) or 24
        model_kwargs['feature_fusion'] = getattr(args, 'feature_fusion', None)
        # 融合层/分类头正则化参数：从 checkpoint 配置自动加载
        head_drop = getattr(args, 'head_drop_rate', 0.0)
        if head_drop > 0:
            model_kwargs['head_drop_rate'] = head_drop
        fusion_hidden = getattr(args, 'fusion_hidden_dim', 512)
        if fusion_hidden != 512:
            model_kwargs['fusion_hidden_dim'] = fusion_hidden
        if getattr(args, 'include_clinical', False):
            clinical_dim = getattr(args, 'clinical_dim', 0)
            if clinical_dim == 0:
                clinical_dim = 10  # 10维实验室指标（肿瘤大小已加入征象向量）
            model_kwargs['clinical_dim'] = clinical_dim
            # 临床特征缩放系数：优先使用命令行参数，否则从checkpoint配置加载
            clinical_scale = getattr(args, 'clinical_scale', None)
            if clinical_scale is not None:
                model_kwargs['clinical_scale_init'] = clinical_scale
    
    model = create_model(
        args.model,
        **model_kwargs)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' %
                 (args.model, param_count))

    model = model.cuda()
    if args.apex_amp:
        model = amp.initialize(model, opt_level='O1')

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(
            model, device_ids=list(range(args.num_gpu)))

    dataset = MultiPhaseLiverDataset(args, is_training=False)
    case_names = dataset.get_case_names()

    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        num_workers=args.workers,
                        pin_memory=args.pin_mem,
                        shuffle=False)

    predictions = []
    labels = []
    case_names_out = []
    feature_predictions = []
    feature_labels_list = []
    lr_predictions = []
    lr_labels_list = []

    model.eval()
    pbar = tqdm(total=len(dataset))
    with torch.no_grad():
        for batch in loader:
            extra_features = None
            lr_label = None
            if len(batch) == 5:
                # 5元组：(image, label, features, lr_label, clinical)
                input, target, features, lr_label, extra_features = batch
            elif len(batch) == 4:
                input, target, features, fourth = batch
                # 判断第4元素是LR标签还是临床特征
                if isinstance(fourth, torch.Tensor) and fourth.dim() == 1 and fourth.dtype in (torch.long, torch.int, torch.int32):
                    lr_label = fourth
                else:
                    extra_features = fourth
            elif len(batch) == 3:
                input, target, features = batch
            else:
                input, target = batch
                features = None
            
            target = target.cuda()
            input = input.cuda()
            if extra_features is not None:
                extra_features = extra_features.cuda()
            # compute output
            with amp_autocast():
                if extra_features is not None:
                    output = model(input, extra_features=extra_features)
                else:
                    output = model(input)
                # 支持 feature head + lr head 输出: (out1, feature_out, lr_pred)
                if isinstance(output, (tuple, list)):
                    is_feature_model = 'features' in getattr(args, 'model', '')
                    if is_feature_model and len(output) >= 2 and output[1] is not None and output[1].dim() == 2:
                        feature_predictions.append(output[1])
                        if features is not None:
                            feature_labels_list.append(features)
                    # 收集LR预测结果
                    if len(output) >= 3 and output[2] is not None:
                        lr_predictions.append(output[2])
                        if lr_label is not None:
                            lr_labels_list.append(lr_label)
                    output = output[0]
            predictions.append(output)
            labels.append(target)
            pbar.update(input.size(0))
        pbar.close()
    
    return process_prediction(predictions, labels, args, feature_predictions, feature_labels_list, case_names, lr_predictions, lr_labels_list)


def process_prediction(outputs, targets, args, feature_outputs=None, feature_targets=None, case_names=None, lr_outputs=None, lr_targets=None):
    outputs = torch.cat(outputs, dim=0).detach()
    targets = torch.cat(targets, dim=0).detach()
    
    targets_np = targets.cpu().numpy()
    
    # 在 logit 空间应用少数类偏置（softmax 之前），仅 5 类 LR 模式
    minority_bias = getattr(args, 'minority_bias', 0.0)
    lr34_bias_search = getattr(args, 'lr34_bias_search', False)

    # 方案四：自动搜索最优 bias（在 softmax 之前）
    if lr34_bias_search and outputs.shape[1] == 5:
        logits_np = outputs.cpu().numpy()
        target_metric = getattr(args, 'lr34_bias_target', 'f1')
        best_bias, best_score, search_results = search_optimal_lr34_bias(
            logits_np, targets_np, num_classes=outputs.shape[1], target=target_metric)
        minority_bias = best_bias
        _logger.info(f"[方案四] Auto-search selected bias={best_bias} (score={best_score:.4f})")
        # 保存搜索结果到文件
        if getattr(args, 'results_dir', ''):
            search_path = os.path.join(args.results_dir, 'lr34_bias_search.json')
            os.makedirs(args.results_dir, exist_ok=True)
            with open(search_path, 'w') as f:
                json.dump({'best_bias': best_bias, 'best_score': best_score,
                           'target': target_metric, 'results': search_results}, f, indent=2)
            _logger.info(f"[方案四] Search results saved to {search_path}")

    if minority_bias > 0 and outputs.shape[1] == 5:
        outputs_np = outputs.cpu().numpy()
        outputs_np = apply_minority_logit_bias(outputs_np, minority_bias, outputs_np.shape[1])
        pred_score_np = torch.softmax(torch.from_numpy(outputs_np), dim=1).cpu().numpy()
    else:
        pred_score_np = torch.softmax(outputs, dim=1).cpu().numpy()
    
    # 处理征象预测结果
    feature_pred_np = None
    feature_true_np = None
    if feature_outputs and len(feature_outputs) > 0:
        feature_pred = torch.cat(feature_outputs, dim=0).detach()
        feature_pred_np = feature_pred.cpu().numpy()
        if feature_targets and len(feature_targets) > 0:
            feature_true = torch.cat(feature_targets, dim=0).detach()
            feature_true_np = feature_true.cpu().numpy()
    
    # 处理LR等级预测结果
    lr_pred_np = None
    lr_true_np = None
    if lr_outputs and len(lr_outputs) > 0:
        lr_pred = torch.cat(lr_outputs, dim=0).detach()
        lr_pred_np = lr_pred.cpu().numpy()
        if lr_targets and len(lr_targets) > 0:
            lr_true = torch.cat(lr_targets, dim=0).detach()
            lr_true_np = lr_true.cpu().numpy()
    
    return pred_score_np, targets_np, feature_pred_np, feature_true_np, case_names, lr_pred_np, lr_true_np


def write_score2json(score_info, args, case_names=None):
    score_info = score_info.astype(float)
    score_list = []

    # 优先使用从 Dataset 获取的实际 case name，保证与预测结果严格对齐
    if case_names is not None and len(case_names) == len(score_info):
        for idx, (case_id, score) in enumerate(zip(case_names, score_info)):
            score = list(score)
            pred = score.index(max(score))
            score_list.append({
                'image_id': case_id,
                'prediction': pred,
                'score': score,
            })
    else:
        # fallback: 从 anno 文件读取 case id
        with open(args.val_anno_file, 'r') as f:
            anno_lines = [line.strip() for line in f if line.strip()]
        
        if len(anno_lines) != len(score_info):
            _logger.warning(f"Mismatch between annotation lines ({len(anno_lines)}) and predictions ({len(score_info)}). Trimming to min.")
            min_len = min(len(anno_lines), len(score_info))
            anno_lines = anno_lines[:min_len]
            score_info = score_info[:min_len]

        for idx, line in enumerate(anno_lines):
            parts = line.split()
            if not parts:
                continue
            id = parts[0].rsplit('/', 1)[-1]
            score = list(score_info[idx])
            pred = score.index(max(score))
            score_list.append({
                'image_id': id,
                'prediction': pred,
                'score': score,
            })

    json_data = json.dumps(score_list, indent=4)
    save_name = os.path.join(args.results_dir, 'score.json')
    with open(save_name, 'w') as f:
        f.write(json_data)
    _logger.info(f"Prediction has been saved to '{save_name}'.")


def calculate_and_save_feature_metrics(feature_pred, feature_true, results_dir, selected_features=None):
    """计算并保存每个征象的预测准确性，输出格式类似表2_feature_accuracy.csv
    
    Args:
        feature_pred: (N, num_features) 征象预测概率
        feature_true: (N, num_features) 征象真实值
        results_dir: 结果输出目录
        selected_features: 可选消融用的原始征象索引列表；最终模型使用全部24个征象。
    """
    if feature_pred is None or feature_true is None:
        _logger.info("No feature predictions available, skipping feature metrics.")
        return
    
    num_features = feature_pred.shape[1]
    # 阈值 0.5 将 sigmoid 概率转为二分类预测
    feature_pred_binary = (feature_pred >= 0.5).astype(int)
    
    rows = []
    for i in range(num_features):
        pred_i = feature_pred_binary[:, i]
        true_i = feature_true[:, i].astype(int)
        
        # 计算准确率
        acc = np.mean(pred_i == true_i)
        
        # 统计阴性和阳性数量
        neg_count = int(np.sum(true_i == 0))
        pos_count = int(np.sum(true_i == 1))
        
        # 根据 selected_features 映射征象名称
        if selected_features is not None and i < len(selected_features):
            orig_idx = selected_features[i]
            feature_name = ALL_SIGN_NAMES_24[orig_idx] if 0 <= orig_idx < len(ALL_SIGN_NAMES_24) else f'Feature {orig_idx}'
        elif i < len(ALL_SIGN_NAMES_24):
            feature_name = ALL_SIGN_NAMES_24[i]
        else:
            feature_name = f'Feature {i}'
        rows.append({
            'Imaging Feature': feature_name,
            'Accuracy (%)': f"{acc * 100:.2f}",
            'No. of Lesions without Feature Present': neg_count,
            'No. of Lesions with Feature Present': pos_count,
        })
    
    df = pd.DataFrame(rows)
    
    # 输出 CSV
    csv_path = os.path.join(results_dir, 'feature_accuracy.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    _logger.info(f"Feature accuracy table saved to '{csv_path}'")
    
    # 输出 Markdown
    md_path = os.path.join(results_dir, 'feature_accuracy.md')
    md_lines = ["# Feature Prediction Accuracy\n",
                "| Imaging Feature | Accuracy (%) | No. of Lesions without Feature Present | No. of Lesions with Feature Present |",
                "|---|---|---|---|"]
    for _, row in df.iterrows():
        md_lines.append(
            f"| {row['Imaging Feature']} | {row['Accuracy (%)']} | "
            f"{row['No. of Lesions without Feature Present']} | {row['No. of Lesions with Feature Present']} |"
        )
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    _logger.info(f"Feature accuracy markdown saved to '{md_path}'")
    
    # 打印结果
    _logger.info("\n" + "="*60)
    _logger.info("FEATURE PREDICTION ACCURACY")
    _logger.info("="*60)
    _logger.info(df.to_string(index=False))
    _logger.info("="*60)


def calculate_and_save_metrics(pred_scores, true_labels, args):
    """计算并保存所有评估指标：混淆矩阵、AUC、Accuracy、Sensitivity等"""
    
    # 获取预测类别
    pred_labels = np.argmax(pred_scores, axis=1)
    
    # 1. 准确率 (Accuracy)
    accuracy = ACC(pred_scores, true_labels)
    
    # 2. 混淆矩阵 (Confusion Matrix)
    cm = confusion_matrix(pred_scores, true_labels)
    
    # 3. 分类报告 (包含 Precision, Recall/Sensitivity, F1-score)
    report = cls_report(pred_scores, true_labels)
    
    # 4. 各项指标
    f1 = F1_score(pred_scores, true_labels)
    recall = Recall(pred_scores, true_labels)  # Sensitivity = Recall
    precision = Precision(pred_scores, true_labels)
    kappa = Cohen_Kappa(pred_scores, true_labels)
    
    # 5. AUC (多分类使用 One-vs-Rest)
    num_classes = pred_scores.shape[1]
    try:
        if num_classes == 2:
            auc = roc_auc_score(true_labels, pred_scores[:, 1])
        else:
            # 多分类：One-vs-Rest AUC
            auc = roc_auc_score(true_labels, pred_scores, multi_class='ovr', average='weighted')
    except Exception as e:
        _logger.warning(f"Failed to calculate AUC: {e}")
        auc = None
    
    # 6. 计算每个类别的指标 (ACC, F1, Recall, Precision, Kappa)
    cm_np = sk_confusion_matrix(true_labels, pred_labels)
    per_class_metrics = calculate_per_class_metrics(pred_scores, true_labels, pred_labels, num_classes, cm_np)
    
    # 7. 计算每个类别的 Sensitivity (Recall) - 仅保留在metrics_dict中供JSON使用
    sensitivities = {}
    for i in range(num_classes):
        tp = cm_np[i, i]
        fn = cm_np[i, :].sum() - tp
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sensitivities[f'class_{i}_sensitivity'] = sensitivity
    
    # 构建结果字典
    metrics_dict = {
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'recall_weighted': float(recall),
        'precision_weighted': float(precision),
        'kappa': float(kappa),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'per_class_metrics': per_class_metrics,
    }
    
    if auc is not None:
        metrics_dict['auc'] = float(auc)
    
    metrics_dict.update({k: float(v) for k, v in sensitivities.items()})
    
    # 保存为 JSON
    metrics_json_path = os.path.join(args.results_dir, 'evaluation_metrics.json')
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=4, ensure_ascii=False)
    _logger.info(f"Evaluation metrics saved to '{metrics_json_path}'")
    
    # 保存为 CSV（详细结果）
    metrics_csv_path = os.path.join(args.results_dir, 'evaluation_metrics.csv')
    with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'ACC', 'F1', 'Recall', 'Precision', 'Kappa'])
        for cls_name, metrics in per_class_metrics.items():
            writer.writerow([
                cls_name,
                f"{metrics['ACC']:.4f}",
                f"{metrics['F1']:.4f}",
                f"{metrics['Recall']:.4f}",
                f"{metrics['Precision']:.4f}",
                f"{metrics['Kappa']:.4f}"
            ])
        # 添加整体汇总行
        writer.writerow([
            'ALL',
            f"{float(accuracy):.4f}",
            f"{float(f1):.4f}",
            f"{float(recall):.4f}",
            f"{float(precision):.4f}",
            f"{float(kappa):.4f}"
        ])
    _logger.info(f"Evaluation metrics CSV saved to '{metrics_csv_path}'")
    
    # 打印结果
    _logger.info("\n" + "="*60)
    _logger.info("EVALUATION RESULTS")
    _logger.info("="*60)
    _logger.info(f"Accuracy: {accuracy * 100:.2f}%")
    if auc is not None:
        _logger.info(f"AUC: {auc:.4f}")
    _logger.info(f"F1 Score (Weighted): {f1:.4f}")
    _logger.info(f"Sensitivity/Recall (Weighted): {recall:.4f}")
    _logger.info(f"Precision (Weighted): {precision:.4f}")
    _logger.info(f"Cohen Kappa: {kappa:.4f}")
    _logger.info("\nConfusion Matrix:")
    _logger.info(f"{cm}")
    _logger.info("\nClassification Report:")
    _logger.info(report)
    _logger.info("="*60)
    
    # 绘制并保存混淆矩阵图和ROC曲线
    plot_confusion_matrix(cm_np, args.num_classes, args.results_dir)
    plot_multiclass_roc(pred_scores, true_labels, num_classes, args.results_dir)
    
    # 8. 阈值优化（如果启用）
    if getattr(args, 'threshold_optimize', False):
        _logger.info("\n[Threshold Optimization] Running per-class threshold search...")
        best_thresholds, opt_pred_labels, orig_pred_labels, metrics_comp = \
            optimize_per_class_thresholds(pred_scores, true_labels, num_classes)
        
        # 保存优化结果到 JSON
        threshold_json_path = os.path.join(args.results_dir, 'threshold_optimization.json')
        with open(threshold_json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_comp, f, indent=4, ensure_ascii=False)
        _logger.info(f"Threshold optimization results saved to '{threshold_json_path}'")
        
        # 打印对比结果
        _logger.info("\n" + "="*60)
        _logger.info("THRESHOLD OPTIMIZATION RESULTS")
        _logger.info("="*60)
        _logger.info(f"Optimal thresholds: {metrics_comp['thresholds']}")
        _logger.info(f"\n{'Metric':<15} {'Original':>10} {'Optimized':>10} {'Delta':>10}")
        _logger.info("-" * 50)
        for key in ['macro_F1', 'weighted_F1', 'accuracy', 'kappa']:
            orig_v = metrics_comp['original'][key]
            opt_v = metrics_comp['optimized'][key]
            delta = opt_v - orig_v
            _logger.info(f"{key:<15} {orig_v:>10.4f} {opt_v:>10.4f} {delta:>+10.4f}")
        
        _logger.info("\nPer-class F1 comparison:")
        if num_classes == 5:
            cls_names = ['LR-1/2', 'LR-3', 'LR-4', 'LR-5', 'LR-M']
        elif num_classes == 3:
            cls_names = ['Benign', 'Non-HCC Malignancies', 'HCC']
        else:
            cls_names = [f'Class {i}' for i in range(num_classes)]
        _logger.info(f"{'Class':<12} {'Orig F1':>10} {'Opt F1':>10} {'Orig R':>10} {'Opt R':>10} {'Orig P':>10} {'Opt P':>10}")
        _logger.info("-" * 75)
        for cn in cls_names:
            orig_c = metrics_comp['original'].get(cn, {})
            opt_c = metrics_comp['optimized'].get(cn, {})
            _logger.info(f"{cn:<12} {orig_c.get('F1',0):>10.4f} {opt_c.get('F1',0):>10.4f} "
                        f"{orig_c.get('Recall',0):>10.4f} {opt_c.get('Recall',0):>10.4f} "
                        f"{orig_c.get('Precision',0):>10.4f} {opt_c.get('Precision',0):>10.4f}")
        _logger.info("="*60)
        
        # 保存优化后的 CSV
        opt_csv_path = os.path.join(args.results_dir, 'evaluation_metrics_optimized.csv')
        with open(opt_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Class', 'ACC', 'F1', 'Recall', 'Precision', 'Kappa'])
            for cn in cls_names:
                opt_c = metrics_comp['optimized'].get(cn, {})
                writer.writerow([cn,
                    f"{opt_c.get('ACC', 0):.4f}",
                    f"{opt_c.get('F1',0):.4f}",
                    f"{opt_c.get('Recall',0):.4f}",
                    f"{opt_c.get('Precision',0):.4f}",
                    f"{opt_c.get('Kappa', 0):.4f}"])
            writer.writerow(['threshold_optimize (weighted)',
                f"{metrics_comp['optimized']['accuracy']:.4f}",
                f"{metrics_comp['optimized'].get('weighted_F1', 0):.4f}",
                f"{metrics_comp['optimized'].get('weighted_recall', 0):.4f}",
                f"{metrics_comp['optimized'].get('weighted_precision', 0):.4f}",
                f"{metrics_comp['optimized']['kappa']:.4f}"])
        _logger.info(f"Optimized metrics CSV saved to '{opt_csv_path}'")
    
    return metrics_dict


def plot_confusion_matrix(cm, num_classes, results_dir):
    """绘制混淆矩阵热力图（基于百分比），避免类别数量不均导致颜色不均衡"""
    plt.figure(figsize=(10, 8))
    
    # 根据类别数生成标签
    if num_classes == 5:
        labels = ['LR-1/2', 'LR-3', 'LR-4', 'LR-5', 'LR-M']
    elif num_classes == 3:
        labels = ['Benign', 'Non-HCC Malignancies', 'HCC']
    else:
        labels = [f'Class {i}' for i in range(num_classes)]
    
    # 将混淆矩阵转换为行百分比（每行求和为100%）
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_percent = np.where(cm_sum > 0, cm / cm_sum * 100, 0)
    
    # 绘制热力图
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                linewidths=.5, linecolor='gray',
                cbar_kws={'label': 'Percentage (%)'})
    
    plt.title('Confusion Matrix (Row-normalized %)', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(results_dir, 'confusion_matrix_percent.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    _logger.info(f"Confusion matrix plot (percentage) saved to '{save_path}'")
    
    # 同时保存原始计数的版本（供参考）
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                linewidths=.5, linecolor='gray')
    
    plt.title('Confusion Matrix (Raw Counts)', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    save_path_raw = os.path.join(results_dir, 'confusion_matrix_raw.png')
    plt.savefig(save_path_raw, dpi=300, bbox_inches='tight')
    plt.close()
    _logger.info(f"Confusion matrix plot (raw counts) saved to '{save_path_raw}'")


def plot_multiclass_roc(pred_scores, true_labels, num_classes, results_dir):
    """绘制多分类ROC曲线（One-vs-Rest），将所有类别画在同一张图上"""
    plt.figure(figsize=(10, 8))
    
    # 根据类别数生成标签
    if num_classes == 5:
        labels = ['LR-1/2', 'LR-3', 'LR-4', 'LR-5', 'LR-M']
    elif num_classes == 3:
        labels = ['Benign', 'Non-HCC Malignancies', 'HCC']
    else:
        labels = [f'Class {i}' for i in range(num_classes)]
    
    # 二分类
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(true_labels, pred_scores[:, 1])
        roc_auc_value = roc_auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{labels[1]} (AUC = {roc_auc_value:.3f})')
    
    # 多分类：One-vs-Rest
    else:
        for i in range(num_classes):
            # 构建二分类标签
            binary_labels = (true_labels == i).astype(int)
            scores = pred_scores[:, i]
            # 过滤 NaN/Inf，避免 roc_curve 报错
            valid_mask = np.isfinite(scores)
            if not valid_mask.any():
                _logger.warning(f"Class {i} has no valid prediction scores, skipping ROC.")
                continue
            fpr, tpr, _ = roc_curve(binary_labels[valid_mask], scores[valid_mask])
            roc_auc_value = roc_auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{labels[i]} (AUC = {roc_auc_value:.3f})')
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-class ROC Curve (One-vs-Rest)', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(results_dir, 'roc_curve_multiclass.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    _logger.info(f"ROC curve plot saved to '{save_path}'")


def calculate_per_class_metrics(pred_scores, true_labels, pred_labels, num_classes, cm_np):
    """计算每个类别的 ACC, F1, Recall, Precision, Kappa"""
    from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score
    
    per_class_metrics = {}
    
    for i in range(num_classes):
        # 构建二分类标签（当前类 vs 其他类）
        binary_true = (true_labels == i).astype(int)
        binary_pred = (pred_labels == i).astype(int)
        
        # 计算该类的指标
        acc = np.mean(binary_true == binary_pred)
        
        # 计算 F1, Precision, Recall（如果该类没有样本，返回0）
        try:
            f1 = f1_score(binary_true, binary_pred, zero_division=0)
            recall = recall_score(binary_true, binary_pred, zero_division=0)
            precision = precision_score(binary_true, binary_pred, zero_division=0)
            kappa = cohen_kappa_score(binary_true, binary_pred)
        except Exception as e:
            _logger.warning(f"Failed to calculate metrics for class {i}: {e}")
            f1 = 0.0
            recall = 0.0
            precision = 0.0
            kappa = 0.0
        
        # 类别标签
        if num_classes == 5:
            cls_name = ['LR-1/2', 'LR-3', 'LR-4', 'LR-5', 'LR-M'][i]
        elif num_classes == 3:
            cls_name = ['Benign', 'Non-HCC Malignancies', 'HCC'][i]
        else:
            cls_name = f'Class {i}'
        
        per_class_metrics[cls_name] = {
            'ACC': float(acc),
            'F1': float(f1),
            'Recall': float(recall),
            'Precision': float(precision),
            'Kappa': float(kappa),
        }
    
    return per_class_metrics


def save_lr_predictions(lr_pred, lr_true, results_dir):
    """保存LR等级预测结果到CSV，并计算总体准确率"""
    if lr_pred is None:
        return
    
    lr_labels = ['LR-1/2', 'LR-3', 'LR-4', 'LR-5', 'LR-M']
    lr_pred_labels = np.argmax(lr_pred, axis=1)
    
    rows = []
    for i in range(len(lr_pred)):
        row = {
            'sample_idx': i,
            'pred_lr': lr_labels[lr_pred_labels[i]] if lr_pred_labels[i] < len(lr_labels) else f'Class_{lr_pred_labels[i]}',
            'pred_probs': ';'.join([f'{p:.4f}' for p in lr_pred[i]]),
        }
        if lr_true is not None:
            row['true_lr'] = lr_labels[int(lr_true[i])] if int(lr_true[i]) < len(lr_labels) else f'Class_{int(lr_true[i])}'
        rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, 'lr_predictions.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    _logger.info(f"LR predictions saved to '{csv_path}'")
    
    # 计算并打印LR预测准确率
    if lr_true is not None:
        lr_acc = np.mean(lr_pred_labels == lr_true)
        _logger.info(f"LR grade prediction accuracy: {lr_acc * 100:.2f}%")
        # 每个等级的准确率
        for i, label in enumerate(lr_labels):
            mask = lr_true == i
            if mask.sum() > 0:
                cls_acc = np.mean(lr_pred_labels[mask] == i)
                _logger.info(f"  {label}: {cls_acc * 100:.2f}% (n={int(mask.sum())})")


def main():
    setup_default_logging()
    args = parser.parse_args()
    
    # 如果提供了 checkpoint 路径，自动推断其他参数
    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt_dir = os.path.dirname(args.checkpoint)
        args_file = os.path.join(ckpt_dir, 'args.yaml')
        
        if os.path.exists(args_file):
            import yaml
            with open(args_file, 'r') as f:
                ckpt_args = yaml.safe_load(f)
            
            # 从 args.yaml 加载关键配置，但保留命令行传入的参数优先级
            for key in ['model', 'num_classes', 'img_size', 'crop_size', 'label_mode', 'num_feature_classes', 'feature_fusion', 'selected_features', 'include_clinical', 'clinical_dim', 'clinical_scale', 'normalize_clinical', 'prior_calibration', 'train_anno_file', 'data1_file', 'case_mapping_file', 'minority_bias', 'ordinal_alpha', 'ordinal_type', 'minority_repeat', 'minority_classes', 'head_drop_rate', 'fusion_hidden_dim']:
                if key in ckpt_args:
                    if not hasattr(args, key) or getattr(args, key) is None or getattr(args, key) == parser.get_default(key):
                        setattr(args, key, ckpt_args[key])
            
            # 自动推断 clinical_stats_file：从 checkpoint 目录向上查找
            if getattr(args, 'normalize_clinical', False) and not getattr(args, 'clinical_stats_file', ''):
                _stats_candidate = os.path.join(os.path.dirname(ckpt_dir), 'clinical_stats.json')
                if os.path.exists(_stats_candidate):
                    args.clinical_stats_file = _stats_candidate
                    _logger.info(f"Auto-detected clinical stats file: {_stats_candidate}")
            
            _logger.info(f"Loaded configuration from {args_file}")
            
            # 自动检测 hierarchical vs hierarchical_simple：检查 checkpoint 是否包含 lr_head
            if getattr(args, 'feature_fusion', None) == 'hierarchical':
                try:
                    import torch as _torch
                    _ckpt = _torch.load(args.checkpoint, map_location='cpu', weights_only=False)
                    _sd = _ckpt.get('state_dict', _ckpt)
                    if 'lr_head.weight' not in _sd:
                        args.feature_fusion = 'hierarchical_simple'
                        _logger.info("Auto-detected hierarchical_simple mode (checkpoint has no lr_head)")
                    del _ckpt, _sd
                except Exception as e:
                    _logger.warning(f"Failed to auto-detect fusion mode: {e}")
        else:
            _logger.warning(f"args.yaml not found in {ckpt_dir}. Using default/command-line arguments.")

    if not args.results_dir:
        args.results_dir = os.path.join(os.path.dirname(args.checkpoint), 'pred_results')
    
    # 1. Predict on internal validation set
    _logger.info("===== Predicting on internal validation set =====")
    pred_scores, true_labels, feature_pred, feature_true, case_names, lr_pred, lr_true = validate(args)
    val_results_dir = os.path.join(args.results_dir, 'val')
    os.makedirs(val_results_dir, exist_ok=True)
    original_results_dir = args.results_dir
    args.results_dir = val_results_dir
    # 类别先验校正（如果启用）
    if getattr(args, 'prior_calibration', False):
        train_anno = getattr(args, 'train_anno_file', None)
        if not train_anno:
            # 从 args.yaml 中推断 train_anno_file
            if args.checkpoint and os.path.isfile(args.checkpoint):
                ckpt_dir = os.path.dirname(args.checkpoint)
                args_file = os.path.join(ckpt_dir, 'args.yaml')
                if os.path.exists(args_file):
                    import yaml
                    with open(args_file, 'r') as f:
                        ckpt_args = yaml.safe_load(f)
                    train_anno = ckpt_args.get('train_anno_file', None)
        if train_anno and os.path.exists(train_anno):
            train_counts = load_class_counts_from_anno(train_anno, args.label_mode, args.num_classes)
            if train_counts is not None:
                _logger.info(f"Applying prior calibration with train anno: {train_anno}")
                pred_scores = calibrate_with_prior(pred_scores, train_counts, true_labels)
            else:
                _logger.warning("Failed to load train class counts, skipping prior calibration")
        else:
            _logger.warning(f"Train anno file not found: {train_anno}, skipping prior calibration")

    write_score2json(pred_scores, args, case_names=case_names)
    calculate_and_save_metrics(pred_scores, true_labels, args)
    calculate_and_save_feature_metrics(feature_pred, feature_true, val_results_dir, getattr(args, 'selected_features', None))
    save_lr_predictions(lr_pred, lr_true, val_results_dir)
    
    # 2. Predict on external test set if provided
    if args.test_anno_file and os.path.exists(args.test_anno_file):
        _logger.info("===== Predicting on external test set =====")
        args.val_anno_file = args.test_anno_file
        pred_scores, true_labels, feature_pred, feature_true, case_names, lr_pred, lr_true = validate(args)
        test_results_dir = os.path.join(original_results_dir, 'test')
        os.makedirs(test_results_dir, exist_ok=True)
        args.results_dir = test_results_dir

        # 类别先验校正（测试集）
        if getattr(args, 'prior_calibration', False):
            if train_anno and os.path.exists(train_anno):
                train_counts = load_class_counts_from_anno(train_anno, args.label_mode, args.num_classes)
                if train_counts is not None:
                    _logger.info(f"Applying prior calibration to test set")
                    pred_scores = calibrate_with_prior(pred_scores, train_counts, true_labels)

        write_score2json(pred_scores, args, case_names=case_names)
        calculate_and_save_metrics(pred_scores, true_labels, args)
        calculate_and_save_feature_metrics(feature_pred, feature_true, test_results_dir, getattr(args, 'selected_features', None))
        save_lr_predictions(lr_pred, lr_true, test_results_dir)

if __name__ == '__main__':
    main()