#!/usr/bin/env python3
"""
SHAP Analysis for the CT-EAIC classification model (classification_models/ckpts/ct_eaic)
- Model: uniformer_small_IL_features with hierarchical_simple fusion + clinical features
- 24 imaging features (select 5-fold CV accuracy > 80% on training set) + 10 clinical features
- Only violin plots, all English text
- Output: outputs/shap/ct_eaic

Feature filtering:
  1. Run compute_5fold_feature_accuracy.py to get 5-fold CV accuracy on training set
  2. Imaging features with accuracy > 80% are retained for SHAP analysis
  3. SHAP is computed on val and test using the unified feature set

Run: python interpretability/shap_ct_eaic.py --checkpoint <CT_EAIC_CHECKPOINT>
"""

import os
import re
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'classification_models/main')
import models  # noqa: F401
warnings.filterwarnings('ignore')

# ===================== 24 Imaging Feature Names (matching ALL_SIGN_NAMES_24, data1.xlsx col 18~41) =====================
IMAGING_FEATURE_NAMES_EN = [
    'Nonrim arterial phase hyperenhancement',      # 0
    'Rim APHE',                                      # 1
    'Nonperipheral washout',                        # 2
    'Peripheral "Washout"',                           # 3
    'Corona Enhancement',                           # 4
    'Enhancing capsule',                            # 5
    'Nonenhancing capsule',                         # 6
    'Peripheral Discontinuous Nodular Enhancement',  # 7
    'Progressive Enhancement',                      # 8
    'Centripetal Enhancement',                      # 9
    'Parallels blood pool enhancement',             # 10
    'Uniform AP Enhancement',                       # 11
    'Uniform PVP Enhancement',                      # 12
    'Uniform DP Enhancement',                       # 13
    'Necrosis or severe ischemia',                  # 14
    'Blood Products in Mass',                       # 15
    'Nodule-in-nodule architecture',                # 16
    'Mosaic Architecture',                          # 17
    'Delayed Central Enhancement',                  # 18
    'Infiltrative appearance',                      # 19
    'Portal venous phase peritumoral hypoenhancement', # 20
    'Fat in Mass, more than Liver',                 # 21
    'Fat Sparing in Solid Mass',                    # 22
    'Intratumoral artery',                          # 23
]

NUM_IMAGING_FEATURES = len(IMAGING_FEATURE_NAMES_EN)  # 24

# ===================== 10 Clinical Feature Names =====================
CLINICAL_FEATURE_NAMES_EN = [
    'Sex',          # 0: 性别
    'Age',          # 1: 年龄
    'AFP',          # 2: 甲胎蛋白
    'PLT',          # 3: 血小板计数
    'ALB',          # 4: 白蛋白
    'ALT',          # 5: 谷丙转氨酶
    'AST',          # 6: 谷草转氨酶
    'ALP',          # 7: 碱性磷酸酶
    'TBIL',         # 8: 总胆红素
    'PT',           # 9: 凝血酶原时间
]

NUM_CLINICAL_FEATURES = len(CLINICAL_FEATURE_NAMES_EN)  # 10

CLASS_NAMES = ['Benign', 'Malignant non-HCC', 'HCC']


# ===================== Feature Filtering =====================
def load_and_filter_features(accuracy_csv, threshold=80.0):
    """Filter imaging features with accuracy > threshold%"""
    df = pd.read_csv(accuracy_csv, encoding='utf-8-sig')
    name_col = df.columns[0]
    acc_col = df.columns[1]

    selected_indices = []
    excluded_indices = []

    for i, en_name in enumerate(IMAGING_FEATURE_NAMES_EN):
        match = df[df[name_col].str.strip().str.lower() == en_name.strip().lower()]
        if len(match) == 0:
            # Fuzzy fallback
            for keyword in ['portal', 'fat in mass', 'mosaic', 'blood products', 'mild']:
                if keyword in en_name.lower():
                    match = df[df[name_col].str.contains(keyword, case=False, na=False)]
                    if len(match) > 0:
                        break

        if len(match) > 0:
            acc = float(match.iloc[0][acc_col])
            if acc > threshold:
                selected_indices.append(i)
            else:
                excluded_indices.append((i, acc))
        else:
            print(f"  [WARNING] Feature not found in CSV: {en_name}")
            excluded_indices.append((i, -1))

    return selected_indices, excluded_indices, df


# ===================== SHAP Wrapper =====================
class HierarchicalSimpleClinicalSHAPPredictor(nn.Module):
    """
    SHAP wrapper for hierarchical_simple model with clinical features.
    Fixed: GAP(512)
    Variable: selected imaging feature probabilities + clinical features (n_img + 10)
    Non-selected imaging features: filled with population mean
    Output: num_classes softmax probabilities
    
    Model forward: GAP(512) + features(24) + clinical*scale(10) → intermediate_fc → head(3)
    """

    def __init__(self, intermediate_fc, head, fixed_gap,
                 feature_mean_all, selected_img_indices, non_selected_img_indices,
                 clinical_scale=None):
        super().__init__()
        self.intermediate_fc = intermediate_fc
        self.head = head
        self.fixed_gap = fixed_gap
        self.feature_mean = feature_mean_all  # mean of 24 imaging features
        self.selected_img_idx = selected_img_indices
        self.non_selected_img_idx = non_selected_img_indices
        self.clinical_scale = clinical_scale  # learnable scale parameter (tensor or None)

    def forward(self, x_input):
        """
        x_input: (B, n_selected_img + n_clinical)
            First n_selected_img columns: selected imaging feature probs
            Last n_clinical columns: raw clinical feature values
        """
        if isinstance(x_input, np.ndarray):
            x_input = torch.from_numpy(x_input).float()
        device = self.fixed_gap.device
        if not x_input.is_cuda:
            x_input = x_input.to(device)

        B = x_input.size(0)
        n_sel_img = len(self.selected_img_idx)
        n_clinical = NUM_CLINICAL_FEATURES

        # Split input into imaging and clinical parts
        x_img = x_input[:, :n_sel_img]       # (B, n_sel_img)
        x_clinical = x_input[:, n_sel_img:]   # (B, n_clinical)

        # Build full 24-dim imaging feature vector
        full_features = torch.zeros(B, NUM_IMAGING_FEATURES, device=device, dtype=x_input.dtype)

        # Non-selected imaging features -> population mean
        if len(self.non_selected_img_idx) > 0:
            ns_mean = torch.tensor(
                [self.feature_mean[j] for j in self.non_selected_img_idx],
                device=device, dtype=x_input.dtype,
            )
            for k, j in enumerate(self.non_selected_img_idx):
                full_features[:, j] = ns_mean[k]

        # Selected imaging features -> variable input
        for k, j in enumerate(self.selected_img_idx):
            full_features[:, j] = x_img[:, k]

        # Apply clinical scale if available
        if self.clinical_scale is not None:
            scaled_clinical = x_clinical * self.clinical_scale
        else:
            scaled_clinical = x_clinical

        # Concatenate: gap(512) + features(24) + scaled_clinical(10)
        gap_exp = self.fixed_gap.expand(B, -1)
        combined = torch.cat([gap_exp, full_features, scaled_clinical], dim=-1)

        # intermediate_fc -> head -> softmax
        x = self.intermediate_fc(combined)
        logits = self.head(x)
        probs = torch.softmax(logits, dim=-1)
        return probs


# ===================== Feature Extraction =====================
def extract_features_from_model(model, dataset, device, batch_size=4, include_clinical=True):
    """Extract GAP(512), 24 feature probabilities, and clinical features from each sample"""
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=False)

    all_gap, all_feat, all_labels, all_clinical = [], [], [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            # Dataset returns 5-tuple: (image, label, features, lr, clinical)
            inputs = batch[0]
            labels = batch[1]

            inputs = inputs.to(device)
            x = model.forward_features(inputs)
            gap = x.flatten(2).mean(-1)
            feat_probs = torch.sigmoid(model.feature_fc_head(gap))

            all_gap.append(gap.cpu())
            all_feat.append(feat_probs.cpu())
            all_labels.append(labels)

            # Extract clinical features from batch
            if include_clinical and len(batch) >= 5:
                all_clinical.append(batch[4])
            else:
                # Fallback: zeros
                all_clinical.append(torch.zeros(inputs.size(0), NUM_CLINICAL_FEATURES))

    return {
        'gap': torch.cat(all_gap).numpy(),
        'feat': torch.cat(all_feat).numpy(),
        'labels': torch.cat(all_labels).numpy(),
        'clinical': torch.cat(all_clinical).numpy(),
    }


# ===================== SHAP Computation =====================
def compute_shap_for_data(data, model, device, selected_img_indices, non_selected_img_indices,
                          feature_mean_all, num_classes, nsamples=500,
                          max_eval_samples=200, background_size=50,
                          background_data=None):
    """Compute SHAP values for imaging + clinical features combined"""
    import shap

    N = len(data['labels'])

    if max_eval_samples > 0 and max_eval_samples < N:
        eval_indices = np.random.RandomState(42).choice(N, max_eval_samples, replace=False)
    else:
        eval_indices = np.arange(N)
    n_eval = len(eval_indices)

    n_sel_img = len(selected_img_indices)
    n_clinical = NUM_CLINICAL_FEATURES
    n_total = n_sel_img + n_clinical

    # Build background data: [selected_imaging_features, clinical_features]
    if background_data is not None:
        X_bg = background_data
    else:
        bg_size = min(background_size, N)
        bg_indices = np.random.RandomState(42).choice(N, bg_size, replace=False)
        bg_img = np.array([data['feat'][idx, selected_img_indices] for idx in bg_indices], dtype=np.float32)
        bg_clin = np.array([data['clinical'][idx, :] for idx in bg_indices], dtype=np.float32)
        X_bg = np.concatenate([bg_img, bg_clin], axis=1)

    # Build test data: [selected_imaging_features, clinical_features]
    test_img = np.array([data['feat'][idx, selected_img_indices] for idx in eval_indices], dtype=np.float32)
    test_clin = np.array([data['clinical'][idx, :] for idx in eval_indices], dtype=np.float32)
    X_test = np.concatenate([test_img, test_clin], axis=1)

    background = shap.kmeans(X_bg, min(len(X_bg), 30))

    print(f"  Evaluating SHAP for {n_eval} samples (nsamples={nsamples}, "
          f"background={len(X_bg)}, features={n_total})")
    all_shap = np.zeros((n_eval, n_total, num_classes), dtype=np.float64)

    clinical_scale = getattr(model, 'clinical_scale', None)

    for i, sample_idx in enumerate(eval_indices):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"    Sample {i+1}/{n_eval} (idx={sample_idx}, true_label={data['labels'][sample_idx]})")

        gap_t = torch.from_numpy(data['gap'][sample_idx:sample_idx+1]).float().to(device)

        predictor = HierarchicalSimpleClinicalSHAPPredictor(
            model.intermediate_fc, model.head,
            gap_t,
            feature_mean_all, selected_img_indices, non_selected_img_indices,
            clinical_scale=clinical_scale,
        )

        def predict_fn(x_np):
            with torch.no_grad():
                t = torch.from_numpy(x_np).float().to(device)
                p = predictor(t)
                return p.cpu().numpy()

        explainer = shap.KernelExplainer(predict_fn, background)
        sv = explainer.shap_values(X_test[i:i+1], nsamples=nsamples, silent=True)

        if isinstance(sv, list):
            for ci in range(num_classes):
                all_shap[i, :, ci] = sv[ci][0]
        else:
            all_shap[i] = sv[0]

    print(f"  SHAP computation complete. Shape: {all_shap.shape}")
    return all_shap, X_test, eval_indices


# ===================== Plotting =====================
def plot_shap_violin(shap_values, X_test, feature_names, output_dir, class_idx=2,
                     suffix='', tail_percent=5.0, symmetric_xlim=False,
                     symmetric_expansion=1.0):
    """SHAP violin plot with outlier clipping that preserves every violin body.

    The limits are derived per feature instead of from all SHAP values pooled
    together.  Pooling makes each feature contribute only 1/n_features of the
    values, so a dominant feature (for example AFP) can have much of its main
    density mistaken for global outliers.  The envelope of each feature's
    central 90% keeps the main violin areas visible while still clipping the
    extreme 5% tails on either side.
    """
    import shap

    sv = shap_values[:, :, class_idx]

    # Compute a central range for every feature first, then take their envelope.
    # This preserves 90% of each feature rather than 98% of the pooled matrix.
    feature_low, feature_high = np.percentile(
        sv, [tail_percent, 100.0 - tail_percent], axis=0
    )
    body_left = float(np.min(feature_low))
    body_right = float(np.max(feature_high))
    span = body_right - body_left
    # A modest margin prevents KDE bodies from touching the plot boundary.
    margin = span * 0.10 if span > 0 else max(abs(body_left) * 0.10, 1e-6)
    xlim_left = body_left - margin
    xlim_right = body_right + margin
    if symmetric_xlim:
        symmetric_limit = max(abs(xlim_left), abs(xlim_right)) * symmetric_expansion
        xlim_left = -symmetric_limit
        xlim_right = symmetric_limit

    plt.figure(figsize=(12, max(8, len(feature_names) * 0.45)))
    shap.summary_plot(
        sv, X_test,
        feature_names=feature_names,
        show=False, max_display=len(feature_names),
        plot_type='violin',
        plot_size=None,
    )
    plt.title(f'SHAP Value Distribution for {CLASS_NAMES[class_idx]}', fontsize=13)
    plt.xlim(xlim_left, xlim_right)
    plt.tight_layout()
    fname = f'shap_violin_{CLASS_NAMES[class_idx]}'
    if suffix:
        fname += f'_{suffix}'
    path = os.path.join(output_dir, f'{fname}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_shap_bar_top15(shap_values, feature_names, output_dir, class_idx=2, top_n=15):
    """Horizontal bar chart of top-N features by mean |SHAP|, all bars rightward, colored by sign."""
    sv = shap_values[:, :, class_idx]  # (N_samples, N_features)
    mean_abs = np.mean(np.abs(sv), axis=0)
    mean_shap = np.mean(sv, axis=0)

    # Sort by absolute importance descending
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    # Reverse so largest absolute value is at top after invert_yaxis
    top_idx = top_idx[::-1]

    top_names = [feature_names[i] for i in top_idx]
    top_mean = mean_shap[top_idx]

    # Colors: positive=blue, negative=orange (based on sign of mean SHAP)
    colors = ['#1f77b4' if v >= 0 else '#ff7f0e' for v in top_mean]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.5)))
    y_pos = np.arange(len(top_names))
    # All bars go right using absolute values
    ax.barh(y_pos, np.abs(top_mean), color=colors, edgecolor='white', height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=10)
    ax.set_xlabel('|Mean SHAP value| (impact magnitude)', fontsize=11)
    ax.set_title(f'Top-{top_n} Features for {CLASS_NAMES[class_idx]}', fontsize=13)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    path = os.path.join(output_dir, f'shap_bar_top{top_n}_{CLASS_NAMES[class_idx]}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ===================== Report =====================
def save_report(shap_values, X_test, feature_names, img_selected_indices,
                feature_acc, output_dir, num_classes=3, n_img_features=None,
                report_suffix=''):
    """Save SHAP contribution report CSV with separate imaging and clinical sections"""
    if n_img_features is None:
        n_img_features = len(img_selected_indices)

    rows = []
    for ci in range(num_classes):
        sv = shap_values[:, :, ci]
        mean_abs = np.mean(np.abs(sv), axis=0)
        mean_shap = np.mean(sv, axis=0)
        std_shap = np.std(sv, axis=0)

        for k in range(len(feature_names)):
            if k < n_img_features:
                # Imaging feature
                feat_name = feature_names[k]
                feat_acc = feature_acc.get(feat_name, '')
                feat_type = 'Imaging'
            else:
                # Clinical feature
                feat_name = feature_names[k]
                feat_acc = ''
                feat_type = 'Clinical'

            rows.append({
                'class_index': ci,
                'class_name': CLASS_NAMES[ci],
                'feature_type': feat_type,
                'feature_name': feat_name,
                'feature_accuracy_%': feat_acc,
                'mean_abs_shap': float(mean_abs[k]),
                'mean_shap': float(mean_shap[k]),
                'std_shap': float(std_shap[k]),
                'mean_feature_value': float(np.mean(X_test[:, k])),
            })

    df = pd.DataFrame(rows)
    fname = 'shap_feature_contribution_report'
    if report_suffix:
        fname += f'_{report_suffix}'
    path = os.path.join(output_dir, f'{fname}.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  Saved report: {path}")


# ===================== Accuracy Table =====================
def _save_accuracy_table(acc_df, selected_indices, excluded_indices, output_dir,
                         threshold=80.0):
    """Save supplementary accuracy table (24 imaging features) as CSV and Markdown.

    This table lists all 24 imaging features with their 5-fold CV accuracy,
    indicating which features are retained (> threshold) for SHAP analysis.
    """
    name_col = acc_df.columns[0]
    acc_col = acc_df.columns[1]

    # Build lookup: feature_name -> accuracy
    acc_lookup = {}
    for _, row in acc_df.iterrows():
        acc_lookup[row[name_col].strip()] = float(row[acc_col])

    # Build full 24-feature table
    rows = []
    for i, en_name in enumerate(IMAGING_FEATURE_NAMES_EN):
        acc_val = acc_lookup.get(en_name, None)
        # Fuzzy match if exact match fails
        if acc_val is None:
            for key in acc_lookup:
                if key.lower() == en_name.lower():
                    acc_val = acc_lookup[key]
                    break
        if acc_val is None:
            for kw in ['portal', 'fat in mass', 'mosaic', 'blood products']:
                if kw in en_name.lower():
                    for key in acc_lookup:
                        if kw in key.lower():
                            acc_val = acc_lookup[key]
                            break
                    if acc_val is not None:
                        break
        retained = 'Yes' if (acc_val is not None and acc_val > threshold) else 'No'
        rows.append({
            'No.': i + 1,
            'Imaging Feature': en_name,
            'Accuracy (%)': f"{acc_val:.2f}" if acc_val is not None else 'N/A',
            'Retained for SHAP': retained,
        })

    df_table = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(output_dir, 'feature_accuracy_24features_table.csv')
    df_table.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  Saved accuracy table: {csv_path}")

    # Save Markdown
    md_path = os.path.join(output_dir, 'feature_accuracy_24features_table.md')
    n_selected = len(selected_indices)
    n_excluded = len(excluded_indices)
    md_lines = [
        "# 24个成像特征预测准确率（训练集5折交叉验证）\n",
        f"阈值: Accuracy > {threshold}% → 保留 {n_selected} 个特征, 过滤 {n_excluded} 个特征\n",
        "| No. | Imaging Feature | Accuracy (%) | Retained for SHAP |",
        "|---|---|---|---|",
    ]
    for _, row in df_table.iterrows():
        md_lines.append(
            f"| {row['No.']} | {row['Imaging Feature']} | "
            f"{row['Accuracy (%)']} | {row['Retained for SHAP']} |"
        )
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print(f"  Saved accuracy table (md): {md_path}")


# ===================== Main =====================
def main():
    parser = argparse.ArgumentParser(
        description='SHAP Analysis for the CT-EAIC classification model (hierarchical_simple + clinical, 24 img + 10 clinical)')
    parser.add_argument('--checkpoint',
                        default='checkpoints/classification_models/model_best.pth.tar')
    parser.add_argument('--model', default='uniformer_small_IL_features')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--num_feature_classes', type=int, default=24)
    parser.add_argument('--feature_fusion', default='hierarchical_simple')
    parser.add_argument('--clinical_dim', type=int, default=10)
    parser.add_argument('--include_clinical', action='store_true', default=True)
    parser.add_argument('--data_dir', default='classification_models/data/images/')
    parser.add_argument('--val_anno_file', default='classification_models/data/labels/val_fold1.txt')
    parser.add_argument('--skip_cases_file', default='classification_models/data/skip_cases.txt')
    parser.add_argument('--data1_file', default='classification_models/data/data1.xlsx')
    parser.add_argument('--test_data_file', default='classification_models/data/data2.xlsx',
                        help='Clinical spreadsheet for the external test set')
    parser.add_argument('--test_case_mapping_file', default='case_name_mapping_test.txt',
                        help='Case mapping for the external test set')
    parser.add_argument('--train_accuracy_csv',
                        default=None,
                        help='5-fold CV feature accuracy CSV on training set '
                             '(default: {output_dir}/train_5fold_feature_accuracy.csv). '
                             'If not found, falls back to per-dataset accuracy CSV.')
    parser.add_argument('--accuracy_csv',
                        default='outputs/classification_models/val/feature_accuracy.csv')
    parser.add_argument('--test_accuracy_csv',
                        default='outputs/classification_models/test/feature_accuracy.csv')
    parser.add_argument('--img_size', default=[20, 96, 96], type=int, nargs='+')
    parser.add_argument('--crop_size', default=[10, 80, 80], type=int, nargs='+')
    parser.add_argument('--val_transform_list', default=['center_crop'], nargs='+')
    parser.add_argument('--label_mode', default='original')
    parser.add_argument('--case_mapping_file', default='')
    parser.add_argument('--output_dir', default='outputs/shap/ct_eaic')
    parser.add_argument('--nsamples', type=int, default=500,
                        help='KernelExplainer nsamples')
    parser.add_argument('--max_eval_samples', type=int, default=200,
                        help='Max samples for SHAP evaluation (0=all)')
    parser.add_argument('--background_size', type=int, default=50,
                        help='Background distribution sample size')
    parser.add_argument('--threshold', type=float, default=80.0,
                        help='Imaging feature accuracy threshold (%%)')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    num_classes = args.num_classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # =============== 1. Load model ===============
    print(f"\n{'='*60}")
    print(f"Step 1: Loading the CT-EAIC classification model")
    print(f"{'='*60}")
    from timm.models import create_model
    from datasets.mp_liver_dataset import MultiPhaseLiverDataset

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=num_classes,
        num_feature_classes=args.num_feature_classes,
        feature_fusion=args.feature_fusion,
        clinical_dim=args.clinical_dim,
    )

    # PyTorch >= 2.6 defaults to weights_only=True, while this trusted project
    # checkpoint also stores its argparse.Namespace training configuration.
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)

    # Handle intermediate_fc dimension mismatch
    intermediate_fc_weight_shape = state_dict.get('intermediate_fc.0.weight', torch.empty(0)).shape
    if len(intermediate_fc_weight_shape) > 0 and intermediate_fc_weight_shape[1] != model.intermediate_fc[0].in_features:
        print(f"  [INFO] checkpoint intermediate_fc dim={intermediate_fc_weight_shape[1]}, "
              f"model dim={model.intermediate_fc[0].in_features}. Rebuilding...")
        fusion_dim = intermediate_fc_weight_shape[1]
        model.intermediate_fc = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.0),
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        filtered_missing = [k for k in missing if 'lr_head' not in k and 'auxiliary' not in k]
        if filtered_missing:
            print(f"  Missing keys: {filtered_missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")

    model = model.to(device).eval()
    print(f"  Model: {args.model}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Feature fusion: {args.feature_fusion}")
    print(f"  Num classes: {num_classes}")
    print(f"  Num imaging features: {args.num_feature_classes}")
    print(f"  Clinical dim: {args.clinical_dim}")
    cs = getattr(model, 'clinical_scale', None)
    if cs is not None:
        print(f"  Clinical scale (learned): {cs.item():.4f}")

    # =============== 2. Load datasets ===============
    print(f"\n{'='*60}")
    print(f"Step 2: Loading validation + test datasets")
    print(f"{'='*60}")
    import copy

    print(f"\n  [Val] anno_file={args.val_anno_file}")
    val_dataset = MultiPhaseLiverDataset(args, is_training=False)
    print(f"  [Val] Dataset size: {len(val_dataset)}")
    val_data = extract_features_from_model(model, val_dataset, device, batch_size=args.batch_size,
                                            include_clinical=args.include_clinical)
    val_N = len(val_data['labels'])
    print(f"  [Val] Extracted {val_N} samples, GAP: {val_data['gap'].shape}, "
          f"Feat: {val_data['feat'].shape}, Clinical: {val_data['clinical'].shape}")
    unique, counts = np.unique(val_data['labels'], return_counts=True)
    print(f"  [Val] Label distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    test_anno_file = os.path.join(os.path.dirname(args.val_anno_file), 'test.txt')
    print(f"\n  [Test] anno_file={test_anno_file}")
    test_args = copy.copy(args)
    test_args.val_anno_file = test_anno_file
    # The external test cases (case_3000+) belong to data2.xlsx and the test
    # mapping.  Set these before constructing the dataset so every loader uses
    # the correct case namespace.  The old post-hoc fallback only ran when the
    # whole dictionary was empty; a non-empty dictionary containing unrelated
    # training/validation cases therefore let every test case fall back to 0.
    test_args.data1_file = args.test_data_file
    test_args.case_mapping_file = args.test_case_mapping_file
    test_dataset = MultiPhaseLiverDataset(test_args, is_training=False)

    # Never silently produce a clinical SHAP plot when the test mapping failed.
    # A small number of genuinely incomplete rows may still use the model's
    # established zero-vector fallback; report them explicitly.
    test_case_names = test_dataset.get_case_names()
    missing_test_clinical = [
        name for name in test_case_names if name not in test_dataset.case_to_clinical
    ]
    matched_test_clinical = len(test_case_names) - len(missing_test_clinical)
    if test_case_names and matched_test_clinical == 0:
        preview = ', '.join(missing_test_clinical[:10])
        raise RuntimeError(
            f'Clinical mapping failed for all {len(test_case_names)} test cases '
            f'(first cases: {preview}). Check --test_data_file and '
            f'--test_case_mapping_file; refusing to substitute zero vectors.'
        )
    print(f"  [Test] Matched clinical data for {matched_test_clinical}/"
          f"{len(test_case_names)} cases")
    if missing_test_clinical:
        preview = ', '.join(missing_test_clinical[:10])
        print(f"  [WARNING] {len(missing_test_clinical)} cases have incomplete/missing "
              f"clinical rows and retain the model's zero fallback: {preview}")

    print(f"  [Test] Dataset size: {len(test_dataset)}")
    test_data = extract_features_from_model(model, test_dataset, device, batch_size=args.batch_size,
                                             include_clinical=args.include_clinical)
    test_N = len(test_data['labels'])
    print(f"  [Test] Extracted {test_N} samples, GAP: {test_data['gap'].shape}, "
          f"Feat: {test_data['feat'].shape}, Clinical: {test_data['clinical'].shape}")
    unique, counts = np.unique(test_data['labels'], return_counts=True)
    print(f"  [Test] Label distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    # =============== 3. Compute feature mean ===============
    feature_mean_all = val_data['feat'].mean(axis=0)

    # =============== 4. Cache setup ===============
    cache_dir = os.path.join(args.output_dir, '_cache')
    os.makedirs(cache_dir, exist_ok=True)

    def _load_or_compute_shap(ds_name, ds_data, step_num, bg_data,
                              sel_img_indices, non_sel_img_indices):
        cache_shap = os.path.join(cache_dir, f'{ds_name}_shap.npy')
        cache_X = os.path.join(cache_dir, f'{ds_name}_X.npy')
        if os.path.exists(cache_shap) and os.path.exists(cache_X):
            cached_shap = np.load(cache_shap)
            cached_X = np.load(cache_X)
            n_samples = len(ds_data['labels'])
            if args.max_eval_samples > 0 and args.max_eval_samples < n_samples:
                expected_idx = np.random.RandomState(42).choice(
                    n_samples, args.max_eval_samples, replace=False
                )
            else:
                expected_idx = np.arange(n_samples)
            expected_clinical = ds_data['clinical'][expected_idx]
            expected_features = len(sel_img_indices) + NUM_CLINICAL_FEATURES
            cache_valid = (
                cached_X.shape == (len(expected_idx), expected_features)
                and cached_shap.shape == (len(expected_idx), expected_features, num_classes)
                and np.allclose(
                    cached_X[:, -NUM_CLINICAL_FEATURES:], expected_clinical,
                    rtol=1e-6, atol=1e-7, equal_nan=True,
                )
            )
            if cache_valid:
                print(f"\nStep {step_num}: Loading cached {ds_name} SHAP values")
                return cached_shap, cached_X, expected_idx
            print(f"\nStep {step_num}: Ignoring stale {ds_name} SHAP cache "
                  "(feature layout or clinical values changed)")
        print(f"\n{'='*60}")
        print(f"Step {step_num}: Computing {ds_name} SHAP values")
        print(f"{'='*60}")
        shap_vals, X_test, eval_idx = compute_shap_for_data(
            ds_data, model, device, sel_img_indices, non_sel_img_indices,
            feature_mean_all, num_classes, nsamples=args.nsamples,
            max_eval_samples=args.max_eval_samples, background_data=bg_data,
        )
        np.save(cache_shap, shap_vals)
        np.save(cache_X, X_test)
        return shap_vals, X_test, eval_idx

    # =============== 5. Unified feature filtering from 5-fold CV ===============
    # Try loading 5-fold CV accuracy from training set
    train_acc_csv = args.train_accuracy_csv
    if train_acc_csv is None:
        train_acc_csv = os.path.join(args.output_dir, 'train_5fold_feature_accuracy.csv')

    if os.path.exists(train_acc_csv):
        print(f"\n{'='*60}")
        print(f"Step 5: Loading 5-fold CV feature accuracy from training set")
        print(f"{'='*60}")
        sel_img_idx, excl_img_idx, acc_df = load_and_filter_features(
            train_acc_csv, threshold=args.threshold
        )
        print(f"  Source: {train_acc_csv}")
    else:
        print(f"\n{'='*60}")
        print(f"Step 5: 5-fold CV accuracy not found, falling back to val accuracy")
        print(f"{'='*60}")
        print(f"  [INFO] Run compute_5fold_feature_accuracy.py first to generate:")
        print(f"         {train_acc_csv}")
        sel_img_idx, excl_img_idx, acc_df = load_and_filter_features(
            args.accuracy_csv, threshold=args.threshold
        )
        print(f"  Source: {args.accuracy_csv}")

    sel_img_names = [IMAGING_FEATURE_NAMES_EN[i] for i in sel_img_idx]
    non_sel_img_idx = [i for i in range(NUM_IMAGING_FEATURES) if i not in sel_img_idx]

    name_col = acc_df.columns[0]
    acc_col = acc_df.columns[1]
    feat_acc = {}
    for _, row in acc_df.iterrows():
        feat_acc[row[name_col].strip()] = float(row[acc_col])

    print(f"\n  Selected {len(sel_img_idx)} imaging features (accuracy > {args.threshold}%):")
    for i in sel_img_idx:
        acc_val = feat_acc.get(IMAGING_FEATURE_NAMES_EN[i], '?')
        print(f"    [{acc_val}%] {IMAGING_FEATURE_NAMES_EN[i]}")
    print(f"  Excluded {len(excl_img_idx)} imaging features:")
    for i, acc in excl_img_idx:
        print(f"    [{acc}%] {IMAGING_FEATURE_NAMES_EN[i]}")

    n_sel_img = len(sel_img_idx)
    combined_names = sel_img_names + CLINICAL_FEATURE_NAMES_EN
    print(f"  Total SHAP features: {n_sel_img + NUM_CLINICAL_FEATURES} "
          f"({n_sel_img} imaging + {NUM_CLINICAL_FEATURES} clinical)")

    # =============== 5b. Save supplementary accuracy table ===============
    _save_accuracy_table(acc_df, sel_img_idx, excl_img_idx, args.output_dir,
                         threshold=args.threshold)

    # =============== 6. Per-dataset: SHAP + plot (unified feature set) ===============
    # Background data: [selected_imaging, clinical] from val set
    bg_size = min(args.background_size, val_N)
    bg_indices = np.random.RandomState(42).choice(val_N, bg_size, replace=False)
    bg_img = np.array(
        [val_data['feat'][idx, sel_img_idx] for idx in bg_indices], dtype=np.float32
    )
    bg_clin = np.array(
        [val_data['clinical'][idx, :] for idx in bg_indices], dtype=np.float32
    )
    X_bg = np.concatenate([bg_img, bg_clin], axis=1)

    datasets_config = [
        ('val', 'Validation', val_data),
        ('test', 'Test', test_data),
    ]

    for ds_key, ds_name, ds_data in datasets_config:
        # Compute / load SHAP with the unified selected features
        ds_shap, ds_X, _ = _load_or_compute_shap(
            ds_key, ds_data, ds_key, X_bg, sel_img_idx, non_sel_img_idx
        )

        # Plot
        ds_dir = os.path.join(args.output_dir, ds_key)
        os.makedirs(ds_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Generating {ds_name} SHAP plots -> {ds_dir}/")
        print(f"  Using {len(sel_img_idx)} imaging features (from 5-fold CV training accuracy)")
        print(f"{'='*60}")

        # --- Combined imaging + clinical features (Image-only model violin style) ---
        print(f"\n  --- Combined Features ({len(combined_names)}) ---")
        for ci in range(num_classes):
            plot_shap_violin(ds_shap, ds_X, combined_names, ds_dir,
                             class_idx=ci)

        # --- Imaging features ---
        print(f"\n  --- Imaging Features ({n_sel_img}) ---")
        img_shap = ds_shap[:, :n_sel_img, :]
        img_X = ds_X[:, :n_sel_img]
        for ci in range(num_classes):
            # Malignant non-HCC imaging has a sparse positive tail in both val
            # and test. A symmetric central crop keeps its violin bodies visible
            # and the zero line near the center without showing the full tail.
            is_malignant_imaging = ci == 1
            plot_shap_violin(
                img_shap, img_X, sel_img_names, ds_dir,
                class_idx=ci, suffix='imaging',
                tail_percent=9.0 if is_malignant_imaging else 5.0,
                symmetric_xlim=is_malignant_imaging,
                symmetric_expansion=1.05 if is_malignant_imaging else 1.0,
            )

        # --- Clinical features ---
        print(f"\n  --- Clinical Features ({NUM_CLINICAL_FEATURES}) ---")
        clin_shap = ds_shap[:, n_sel_img:, :]
        clin_X = ds_X[:, n_sel_img:]
        for ci in range(num_classes):
            plot_shap_violin(clin_shap, clin_X, CLINICAL_FEATURE_NAMES_EN, ds_dir,
                             class_idx=ci, suffix='clinical')

        # --- Top-10 combined bar chart ---
        print(f"\n  --- Top-10 Combined Features Bar Chart ---")
        for ci in range(num_classes):
            plot_shap_bar_top15(ds_shap, combined_names, ds_dir, class_idx=ci, top_n=10)

        # --- Reports ---
        save_report(img_shap, img_X, sel_img_names, sel_img_idx,
                    feat_acc, ds_dir, num_classes=num_classes,
                    n_img_features=n_sel_img, report_suffix='imaging')
        save_report(clin_shap, clin_X, CLINICAL_FEATURE_NAMES_EN, [],
                    feat_acc, ds_dir, num_classes=num_classes,
                    n_img_features=0, report_suffix='clinical')

    print(f"\n{'='*60}")
    print(f"All outputs saved to: {args.output_dir}/")
    print(f"  ├── val/                                   (Validation SHAP)")
    print(f"  ├── test/                                  (Test SHAP)")
    print(f"  ├── feature_accuracy_24features_table.csv  (Supplementary table)")
    print(f"  └── feature_accuracy_24features_table.md   (Supplementary table)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
