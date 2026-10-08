#!/usr/bin/env python3
"""
SHAP Analysis for Image-only model (classification_models/ckpts/ImageOnly)
- Model: uniformer_small_IL_features with hierarchical_simple fusion
- 24 features, select those with 5-fold CV accuracy > 80% on training set
- Only violin plots, all English text
- Output: outputs/shap/image_only

Feature filtering:
  1. Run compute_5fold_feature_accuracy.py to get 5-fold CV accuracy on training set
  2. Features with accuracy > 80% are retained for SHAP analysis
  3. SHAP is computed on val and test using the unified feature set

Run: python interpretability/shap_image_only.py --checkpoint <IMAGE_ONLY_CHECKPOINT>
"""

import os
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

# ===================== 24 Feature Names (matching ALL_SIGN_NAMES_24, data1.xlsx col 18~41) =====================
FEATURE_NAMES_EN = [
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

NUM_FEATURES = len(FEATURE_NAMES_EN)  # 24

CLASS_NAMES = ['Benign', 'Malignant non-HCC', 'HCC']


# ===================== Feature Filtering =====================
def load_and_filter_features(accuracy_csv, threshold=80.0):
    """Filter features with accuracy > threshold%"""
    df = pd.read_csv(accuracy_csv, encoding='utf-8-sig')
    name_col = df.columns[0]
    acc_col = df.columns[1]

    selected_indices = []
    excluded_indices = []

    for i, en_name in enumerate(FEATURE_NAMES_EN):
        match = df[df[name_col].str.strip().str.lower() == en_name.strip().lower()]
        if len(match) == 0:
            if 'portal' in en_name.lower():
                match = df[df[name_col].str.contains('portal', case=False)]
            elif 'fat in mass' in en_name.lower():
                match = df[df[name_col].str.contains('Fat in Mass', case=False)]
            elif 'mosaic' in en_name.lower():
                match = df[df[name_col].str.contains('Mosaic', case=False)]
            elif 'blood products' in en_name.lower():
                match = df[df[name_col].str.contains('Blood products', case=False)]
            elif 'mild' in en_name.lower():
                match = df[df[name_col].str.contains('Mild', case=False)]

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
class HierarchicalSimpleSHAPPredictor(nn.Module):
    """
    SHAP wrapper for hierarchical_simple model.
    Fixed: GAP(512)
    Variable: selected feature probabilities (n_selected,)
    Non-selected: filled with population mean
    Output: num_classes softmax probabilities
    """

    def __init__(self, intermediate_fc, head, fixed_gap,
                 feature_mean_all, selected_indices, non_selected_indices):
        super().__init__()
        self.intermediate_fc = intermediate_fc
        self.head = head
        self.fixed_gap = fixed_gap
        self.feature_mean = feature_mean_all
        self.selected_idx = selected_indices
        self.non_selected_idx = non_selected_indices

    def forward(self, x_selected):
        if isinstance(x_selected, np.ndarray):
            x_selected = torch.from_numpy(x_selected).float()
        device = self.fixed_gap.device
        if not x_selected.is_cuda:
            x_selected = x_selected.to(device)

        B = x_selected.size(0)

        # Build full NUM_FEATURES-dim feature vector
        full_features = torch.zeros(B, NUM_FEATURES, device=device, dtype=x_selected.dtype)

        # Non-selected features -> population mean
        if len(self.non_selected_idx) > 0:
            ns_mean = torch.tensor(
                [self.feature_mean[j] for j in self.non_selected_idx],
                device=device, dtype=x_selected.dtype,
            )
            for k, j in enumerate(self.non_selected_idx):
                full_features[:, j] = ns_mean[k]

        # Selected features -> variable input
        for k, j in enumerate(self.selected_idx):
            full_features[:, j] = x_selected[:, k]

        # Concatenate: gap(512) + features(NUM_FEATURES)
        gap_exp = self.fixed_gap.expand(B, -1)
        combined = torch.cat([gap_exp, full_features], dim=-1)

        # intermediate_fc -> head -> softmax
        x = self.intermediate_fc(combined)
        logits = self.head(x)
        probs = torch.softmax(logits, dim=-1)
        return probs


# ===================== Feature Extraction =====================
def extract_features_from_model(model, dataset, device, batch_size=4):
    """Extract GAP(512) and 25 feature probabilities from each sample"""
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=False)

    all_gap, all_feat, all_labels = [], [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            # Dataset may return 2-5 elements: input, label, [feature_targets], [lr_label], [clinical]
            inputs = batch[0]
            labels = batch[1]

            inputs = inputs.to(device)
            x = model.forward_features(inputs)
            gap = x.flatten(2).mean(-1)
            feat_probs = torch.sigmoid(model.feature_fc_head(gap))

            all_gap.append(gap.cpu())
            all_feat.append(feat_probs.cpu())
            all_labels.append(labels)

    return {
        'gap': torch.cat(all_gap).numpy(),
        'feat': torch.cat(all_feat).numpy(),
        'labels': torch.cat(all_labels).numpy(),
    }


# ===================== SHAP Computation =====================
def compute_shap_for_data(data, model, device, selected_indices, non_selected_indices,
                          feature_mean_all, num_classes, nsamples=500,
                          max_eval_samples=200, background_size=50,
                          background_data=None):
    """Compute SHAP values for a dataset"""
    import shap

    N = len(data['labels'])

    if max_eval_samples > 0 and max_eval_samples < N:
        eval_indices = np.random.RandomState(42).choice(N, max_eval_samples, replace=False)
    else:
        eval_indices = np.arange(N)
    n_eval = len(eval_indices)

    if background_data is not None:
        X_bg = background_data
    else:
        bg_size = min(background_size, N)
        bg_indices = np.random.RandomState(42).choice(N, bg_size, replace=False)
        X_bg = np.array([data['feat'][idx, selected_indices] for idx in bg_indices], dtype=np.float32)

    X_test = np.array([data['feat'][idx, selected_indices] for idx in eval_indices], dtype=np.float32)
    background = shap.kmeans(X_bg, min(len(X_bg), 30))

    print(f"  Evaluating SHAP for {n_eval} samples (nsamples={nsamples}, background={len(X_bg)})")
    all_shap = np.zeros((n_eval, len(selected_indices), num_classes), dtype=np.float64)

    for i, sample_idx in enumerate(eval_indices):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"    Sample {i+1}/{n_eval} (idx={sample_idx}, true_label={data['labels'][sample_idx]})")

        gap_t = torch.from_numpy(data['gap'][sample_idx:sample_idx+1]).float().to(device)

        predictor = HierarchicalSimpleSHAPPredictor(
            model.intermediate_fc, model.head,
            gap_t,
            feature_mean_all, selected_indices, non_selected_indices,
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
def plot_shap_violin(shap_values, X_test, selected_names, output_dir, class_idx=2,
                     tail_percent=5.0, symmetric_xlim=False,
                     symmetric_expansion=1.0):
    """SHAP violin plot with outlier clipping that preserves every violin body.

    Compute the central range per feature before taking the shared envelope, so
    a dominant feature cannot have its main density discarded as pooled-matrix
    outliers. Extreme tails remain clipped to keep the plot readable.
    """
    import shap

    sv = shap_values[:, :, class_idx]

    # Preserve the requested central range of every feature, then add padding.
    feature_low, feature_high = np.percentile(
        sv, [tail_percent, 100.0 - tail_percent], axis=0
    )
    body_left = float(np.min(feature_low))
    body_right = float(np.max(feature_high))
    span = body_right - body_left
    margin = span * 0.10 if span > 0 else max(abs(body_left) * 0.10, 1e-6)
    xlim_left = body_left - margin
    xlim_right = body_right + margin
    if symmetric_xlim:
        symmetric_limit = max(abs(xlim_left), abs(xlim_right)) * symmetric_expansion
        xlim_left = -symmetric_limit
        xlim_right = symmetric_limit

    plt.figure(figsize=(12, max(8, len(selected_names) * 0.45)))
    shap.summary_plot(
        sv, X_test,
        feature_names=selected_names,
        show=False, max_display=len(selected_names),
        plot_type='violin',
        plot_size=None,
    )
    plt.title(f'SHAP Value Distribution for {CLASS_NAMES[class_idx]}', fontsize=13)
    plt.xlim(xlim_left, xlim_right)
    plt.tight_layout()
    path = os.path.join(output_dir, f'shap_violin_{CLASS_NAMES[class_idx]}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ===================== Report =====================
def save_report(shap_values, X_test, selected_names, selected_indices,
                feature_acc, output_dir, num_classes=3):
    """Save SHAP contribution report CSV"""
    rows = []
    for ci in range(num_classes):
        sv = shap_values[:, :, ci]
        mean_abs = np.mean(np.abs(sv), axis=0)
        mean_shap = np.mean(sv, axis=0)
        std_shap = np.std(sv, axis=0)

        for k, idx in enumerate(selected_indices):
            rows.append({
                'class_index': ci,
                'class_name': CLASS_NAMES[ci],
                'feature_name': selected_names[k],
                'feature_accuracy_%': feature_acc.get(selected_names[k], ''),
                'mean_abs_shap': float(mean_abs[k]),
                'mean_shap': float(mean_shap[k]),
                'std_shap': float(std_shap[k]),
                'mean_feature_value': float(np.mean(X_test[:, k])),
            })

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, 'shap_feature_contribution_report.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  Saved report: {path}")


# ===================== Accuracy Table =====================
def _save_accuracy_table(acc_df, selected_indices, excluded_indices, output_dir,
                         threshold=80.0):
    """Save supplementary accuracy table (24 features) as CSV and Markdown.

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
    for i, en_name in enumerate(FEATURE_NAMES_EN):
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
    parser = argparse.ArgumentParser(description='SHAP Analysis for Image-only model (hierarchical_simple, 24 features, >80%)')
    parser.add_argument('--checkpoint',
                        default='checkpoints/image_only/model_best.pth.tar')
    parser.add_argument('--model', default='uniformer_small_IL_features')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--num_feature_classes', type=int, default=24)
    parser.add_argument('--feature_fusion', default='hierarchical_simple')
    parser.add_argument('--data_dir', default='classification_models/data/images/')
    parser.add_argument('--val_anno_file', default='classification_models/data/labels/val_fold1.txt')
    parser.add_argument('--skip_cases_file', default='classification_models/data/skip_cases.txt')
    parser.add_argument('--data1_file', default='classification_models/data/data1.xlsx')
    parser.add_argument('--train_accuracy_csv',
                        default=None,
                        help='5-fold CV feature accuracy CSV on training set '
                             '(default: {output_dir}/train_5fold_feature_accuracy.csv). '
                             'If not found, falls back to per-dataset accuracy CSV.')
    parser.add_argument('--accuracy_csv',
                        default='outputs/image_only/val/feature_accuracy.csv')
    parser.add_argument('--test_accuracy_csv',
                        default='outputs/image_only/test/feature_accuracy.csv')
    parser.add_argument('--img_size', default=[20, 96, 96], type=int, nargs='+')
    parser.add_argument('--crop_size', default=[10, 80, 80], type=int, nargs='+')
    parser.add_argument('--val_transform_list', default=['center_crop'], nargs='+')
    parser.add_argument('--label_mode', default='original')
    parser.add_argument('--case_mapping_file', default='')
    parser.add_argument('--output_dir', default='outputs/shap/image_only')
    parser.add_argument('--nsamples', type=int, default=500,
                        help='KernelExplainer nsamples')
    parser.add_argument('--max_eval_samples', type=int, default=200,
                        help='Max samples for SHAP evaluation (0=all)')
    parser.add_argument('--background_size', type=int, default=50,
                        help='Background distribution sample size')
    parser.add_argument('--threshold', type=float, default=80.0,
                        help='Feature accuracy threshold (%%)')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    num_classes = args.num_classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # =============== 1. Load model ===============
    print(f"\n{'='*60}")
    print(f"Step 1: Loading image-only model")
    print(f"{'='*60}")
    from timm.models import create_model
    from datasets.mp_liver_dataset import MultiPhaseLiverDataset

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=num_classes,
        num_feature_classes=args.num_feature_classes,
        feature_fusion=args.feature_fusion,
    )

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)

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
    print(f"  Num features: {args.num_feature_classes}")

    # =============== 2. Load datasets ===============
    print(f"\n{'='*60}")
    print(f"Step 2: Loading validation + test datasets")
    print(f"{'='*60}")
    import copy

    print(f"\n  [Val] anno_file={args.val_anno_file}")
    val_dataset = MultiPhaseLiverDataset(args, is_training=False)
    print(f"  [Val] Dataset size: {len(val_dataset)}")
    val_data = extract_features_from_model(model, val_dataset, device, batch_size=args.batch_size)
    val_N = len(val_data['labels'])
    print(f"  [Val] Extracted {val_N} samples, GAP: {val_data['gap'].shape}, Feat: {val_data['feat'].shape}")
    unique, counts = np.unique(val_data['labels'], return_counts=True)
    print(f"  [Val] Label distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    test_anno_file = os.path.join(os.path.dirname(args.val_anno_file), 'test.txt')
    print(f"\n  [Test] anno_file={test_anno_file}")
    test_args = copy.copy(args)
    test_args.val_anno_file = test_anno_file
    test_dataset = MultiPhaseLiverDataset(test_args, is_training=False)
    print(f"  [Test] Dataset size: {len(test_dataset)}")
    test_data = extract_features_from_model(model, test_dataset, device, batch_size=args.batch_size)
    test_N = len(test_data['labels'])
    print(f"  [Test] Extracted {test_N} samples, GAP: {test_data['gap'].shape}, Feat: {test_data['feat'].shape}")
    unique, counts = np.unique(test_data['labels'], return_counts=True)
    print(f"  [Test] Label distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    # =============== 3. Compute feature mean ===============
    feature_mean_all = val_data['feat'].mean(axis=0)

    # =============== 4. Cache setup ===============
    cache_dir = os.path.join(args.output_dir, '_cache')
    os.makedirs(cache_dir, exist_ok=True)

    def _load_or_compute_shap(ds_name, ds_data, step_num, bg_data,
                              sel_indices, non_sel_indices):
        cache_shap = os.path.join(cache_dir, f'{ds_name}_shap.npy')
        cache_X = os.path.join(cache_dir, f'{ds_name}_X.npy')
        if os.path.exists(cache_shap) and os.path.exists(cache_X):
            cached_shap = np.load(cache_shap)
            cached_X = np.load(cache_X)
            # Validate cache shape matches current feature selection
            n_samples = len(ds_data['labels'])
            if args.max_eval_samples > 0 and args.max_eval_samples < n_samples:
                expected_n = args.max_eval_samples
            else:
                expected_n = n_samples
            expected_X_shape = (expected_n, len(sel_indices))
            expected_shap_shape = (expected_n, len(sel_indices), num_classes)
            if cached_X.shape == expected_X_shape and cached_shap.shape == expected_shap_shape:
                print(f"\nStep {step_num}: Loading cached {ds_name} SHAP values")
                return cached_shap, cached_X, None
            print(f"\nStep {step_num}: Ignoring stale {ds_name} SHAP cache "
                  f"(shape mismatch: cached_X={cached_X.shape} vs expected={expected_X_shape})")
        print(f"\n{'='*60}")
        print(f"Step {step_num}: Computing {ds_name} SHAP values")
        print(f"{'='*60}")
        shap_vals, X_test, eval_idx = compute_shap_for_data(
            ds_data, model, device, sel_indices, non_sel_indices,
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
        sel_idx, excl_idx, acc_df = load_and_filter_features(
            train_acc_csv, threshold=args.threshold
        )
        print(f"  Source: {train_acc_csv}")
    else:
        print(f"\n{'='*60}")
        print(f"Step 5: 5-fold CV accuracy not found, falling back to val accuracy")
        print(f"{'='*60}")
        print(f"  [INFO] Run compute_5fold_feature_accuracy.py first to generate:")
        print(f"         {train_acc_csv}")
        sel_idx, excl_idx, acc_df = load_and_filter_features(
            args.accuracy_csv, threshold=args.threshold
        )
        print(f"  Source: {args.accuracy_csv}")

    sel_names = [FEATURE_NAMES_EN[i] for i in sel_idx]
    non_sel_idx = [i for i in range(NUM_FEATURES) if i not in sel_idx]

    name_col = acc_df.columns[0]
    acc_col = acc_df.columns[1]
    feat_acc = {}
    for _, row in acc_df.iterrows():
        feat_acc[row[name_col].strip()] = float(row[acc_col])

    print(f"\n  Selected {len(sel_idx)} features (accuracy > {args.threshold}%):")
    for i in sel_idx:
        acc_val = feat_acc.get(FEATURE_NAMES_EN[i], '?')
        print(f"    [{acc_val}%] {FEATURE_NAMES_EN[i]}")
    print(f"  Excluded {len(excl_idx)} features:")
    for i, acc in excl_idx:
        print(f"    [{acc}%] {FEATURE_NAMES_EN[i]}")

    # =============== 5b. Save supplementary accuracy table ===============
    _save_accuracy_table(acc_df, sel_idx, excl_idx, args.output_dir,
                         threshold=args.threshold)

    # =============== 6. Per-dataset: SHAP + plot (unified feature set) ===============
    # Background data from val set using the unified selected indices
    bg_size = min(args.background_size, val_N)
    bg_indices = np.random.RandomState(42).choice(val_N, bg_size, replace=False)
    X_bg = np.array(
        [val_data['feat'][idx, sel_idx] for idx in bg_indices], dtype=np.float32
    )

    datasets_config = [
        ('val', 'Validation', val_data),
        ('test', 'Test', test_data),
    ]

    for ds_key, ds_name, ds_data in datasets_config:
        # Compute / load SHAP with the unified selected features
        ds_shap, ds_X, _ = _load_or_compute_shap(
            ds_key, ds_data, ds_key, X_bg, sel_idx, non_sel_idx
        )

        # Plot
        ds_dir = os.path.join(args.output_dir, ds_key)
        os.makedirs(ds_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Generating {ds_name} SHAP violin plots -> {ds_dir}/")
        print(f"  Using {len(sel_idx)} features (from 5-fold CV training accuracy)")
        print(f"{'='*60}")
        for ci in range(num_classes):
            is_val_malignant = ds_key == 'val' and ci == 1
            plot_shap_violin(
                ds_shap, ds_X, sel_names, ds_dir, class_idx=ci,
                tail_percent=7.5 if is_val_malignant else 5.0,
                symmetric_xlim=is_val_malignant,
                symmetric_expansion=1.25 if is_val_malignant else 1.0,
            )

        save_report(ds_shap, ds_X, sel_names, sel_idx,
                    feat_acc, ds_dir, num_classes=num_classes)

    print(f"\n{'='*60}")
    print(f"All outputs saved to: {args.output_dir}/")
    print(f"  ├── val/                              (Validation SHAP)")
    print(f"  ├── test/                             (Test SHAP)")
    print(f"  ├── feature_accuracy_24features_table.csv  (Supplementary table)")
    print(f"  └── feature_accuracy_24features_table.md   (Supplementary table)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
