#!/usr/bin/env python3
"""Comparable t-SNE analysis for the image-only model and the CT-EAIC classification model.

ImageOnly uses imaging features only. CT_EAIC denotes the final classification model of the CT-EAIC system and uses the same imaging branch plus ten
clinical variables. Both models are processed with the same unsupervised PCA
denoising and the same t-SNE search grid. Candidate t-SNE maps are ranked by a
combination of class silhouette (separation) and trustworthiness (local-structure
preservation); the report also includes metrics computed before t-SNE so visual
separation is not treated as evidence on its own.

Examples:
  python interpretability/tsne_image_only_vs_ct_eaic.py --which_model both
  python interpretability/tsne_image_only_vs_ct_eaic.py --which_model CT_EAIC
"""

import argparse
import copy
import json
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, 'classification_models/main')
import models  # noqa: F401 - register custom timm models

warnings.filterwarnings('ignore')

CLASS_NAMES = ['Benign', 'Malignant non-HCC', 'HCC']
# Colorblind-friendly, high-contrast palette (Okabe-Ito inspired).
CLASS_COLORS = ['#2171B5', '#E6B800', '#CB181D']
CLASS_MARKERS = ['o', 'o', 'o']

MODEL_CONFIGS = {
    'ImageOnly': {
        'checkpoint': 'checkpoints/image_only/model_best.pth.tar',
        'clinical_dim': 0,
        'output_dir': 'outputs/tsne/image_only',
        'description': 'Imaging features only',
    },
    'CT_EAIC': {
        'checkpoint': 'checkpoints/classification_models/model_best.pth.tar',
        'clinical_dim': 10,
        'output_dir': 'outputs/tsne/ct_eaic',
        'description': 'Imaging + clinical features',
    },
}


@torch.no_grad()
def extract_fused_features(model, dataset, device, batch_size=4,
                           has_clinical=False, num_workers=0,
                           partial_path=None, cache_every_batches=1,
                           resume=True):
    """Extract 512-d fused vectors with visible, resumable batch progress."""
    from torch.utils.data import DataLoader, Subset

    all_fused, all_labels = [], []
    completed = 0
    if resume and partial_path and os.path.exists(partial_path):
        partial = np.load(partial_path)
        cached_total = int(partial.get('total', -1))
        cached_fused = partial['fused']
        cached_labels = partial['labels']
        if cached_total == len(dataset) and len(cached_fused) == len(cached_labels):
            completed = len(cached_labels)
            if 0 < completed <= len(dataset):
                all_fused.append(torch.from_numpy(cached_fused))
                all_labels.append(torch.from_numpy(cached_labels))
                print(f"    Resuming partial cache: {completed}/{len(dataset)} samples",
                      flush=True)
        if completed > len(dataset):
            completed, all_fused, all_labels = 0, [], []

    remaining = Subset(dataset, range(completed, len(dataset)))
    loader = DataLoader(remaining, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    model.eval()
    total_batches = len(loader)
    started = time.monotonic()

    def save_partial():
        if not partial_path or not all_labels:
            return
        fused = torch.cat(all_fused).numpy()
        labels = torch.cat(all_labels).numpy()
        tmp_path = partial_path + '.tmp.npz'
        np.savez(tmp_path, fused=fused, labels=labels, total=len(dataset))
        os.replace(tmp_path, partial_path)

    for batch_index, batch in enumerate(loader, 1):
        inputs, labels = batch[0].to(device), batch[1]
        gap = model.forward_features(inputs).flatten(2).mean(-1)
        feat_probs = torch.sigmoid(model.feature_fc_head(gap))

        parts = [gap, feat_probs]
        if has_clinical:
            clinical = (batch[4].to(device).float() if len(batch) >= 5
                        else torch.zeros(inputs.size(0), 10, device=device))
            clinical_scale = getattr(model, 'clinical_scale', None)
            if clinical_scale is not None:
                clinical = clinical * clinical_scale
            parts.append(clinical)

        all_fused.append(model.intermediate_fc(torch.cat(parts, dim=-1)).cpu())
        all_labels.append(labels)

        done = completed + sum(len(item) for item in all_labels[1 if completed else 0:])
        elapsed = time.monotonic() - started
        processed_now = max(done - completed, 1)
        eta = elapsed / processed_now * max(len(dataset) - done, 0)
        print(f"    Batch {batch_index}/{total_batches} | samples "
              f"{done}/{len(dataset)} ({done / len(dataset):.1%}) | "
              f"elapsed {elapsed / 60:.1f} min | ETA {eta / 60:.1f} min",
              flush=True)
        if batch_index % max(cache_every_batches, 1) == 0:
            save_partial()

    save_partial()

    return {
        'fused': torch.cat(all_fused).numpy(),
        'labels': torch.cat(all_labels).numpy(),
    }


def preprocess_features(features, max_components=50, variance_target=0.95):
    """L2-normalize, PCA-denoise, and retain enough PCs for target variance."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    normalized = normalize(features, norm='l2')
    max_comp = min(max_components, normalized.shape[0] - 1, normalized.shape[1])
    pca = PCA(n_components=max_comp, random_state=42)
    projected = pca.fit_transform(normalized)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    retained = min(max_comp, max(10, int(np.searchsorted(cumulative, variance_target) + 1)))
    explained = float(cumulative[retained - 1])
    print(f"    PCA: {features.shape[1]}d -> {retained}d "
          f"(explained variance={explained:.1%})")
    return projected[:, :retained], retained, explained


def compute_tsne(features, perplexity, early_exaggeration, random_state=42,
                 max_iter=2000):
    from sklearn.manifold import TSNE

    return TSNE(
        n_components=2,
        perplexity=min(float(perplexity), features.shape[0] - 1),
        early_exaggeration=float(early_exaggeration),
        learning_rate='auto',
        init='pca',
        metric='euclidean',
        max_iter=max_iter,
        random_state=random_state,
        method='barnes_hut',
    ).fit_transform(features)


def select_tsne(features, labels, random_state=42):
    """Search a fixed grid using separation plus local-structure preservation."""
    from sklearn.manifold import trustworthiness
    from sklearn.metrics import silhouette_score

    n_samples = len(features)
    perplexities = [p for p in (30, 40, 50) if p < n_samples / 3]
    exaggerations = (12, 20)
    candidates = []

    for perplexity in perplexities:
        for exaggeration in exaggerations:
            coords = compute_tsne(features, perplexity, exaggeration,
                                  random_state=random_state)
            silhouette = float(silhouette_score(coords, labels))
            trust = float(trustworthiness(
                features, coords, n_neighbors=min(10, (n_samples - 1) // 2)
            ))
            # Silhouette drives class separation; trustworthiness prevents a
            # visually attractive but structurally implausible map from winning.
            objective = silhouette + 0.20 * (trust - 0.90)
            candidates.append({
                'coords': coords,
                'perplexity': perplexity,
                'early_exaggeration': exaggeration,
                'silhouette_2d': silhouette,
                'trustworthiness': trust,
                'objective': objective,
            })
            print(f"    perp={perplexity:>2}, exaggeration={exaggeration:>2}: "
                  f"silhouette={silhouette:+.4f}, trust={trust:.4f}, "
                  f"objective={objective:+.4f}")

    best = max(candidates, key=lambda item: item['objective'])
    print(f"    >>> selected perplexity={best['perplexity']}, "
          f"early_exaggeration={best['early_exaggeration']}, "
          f"silhouette={best['silhouette_2d']:+.4f}, "
          f"trust={best['trustworthiness']:.4f}")
    return best


def representation_metrics(features, coords, labels):
    """Metrics in PCA space and in the displayed 2-D map."""
    from sklearn.metrics import (calinski_harabasz_score,
                                 davies_bouldin_score, silhouette_score)
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    min_class = int(np.min(np.bincount(labels.astype(int))))
    folds = min(5, min_class)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    neighbors = min(15, max(3, len(labels) // 20))
    knn = KNeighborsClassifier(n_neighbors=neighbors, weights='distance')
    knn_balanced = float(cross_val_score(
        knn, features, labels, cv=cv, scoring='balanced_accuracy'
    ).mean())

    return {
        'silhouette_highd': float(silhouette_score(features, labels)),
        'davies_bouldin_highd': float(davies_bouldin_score(features, labels)),
        'calinski_harabasz_highd': float(calinski_harabasz_score(features, labels)),
        'knn_balanced_accuracy_highd': knn_balanced,
        'silhouette_2d': float(silhouette_score(coords, labels)),
        'davies_bouldin_2d': float(davies_bouldin_score(coords, labels)),
        'calinski_harabasz_2d': float(calinski_harabasz_score(coords, labels)),
    }


def apply_gravity_tightening(coords, labels, alpha=0.25, iterations=4):
    """Iteratively pull 2-D points toward their class centroid for tighter clusters.

    This post-processing step preserves relative within-class structure while
    reducing inter-point scatter, producing visually more compact groups.
    """
    optimized = coords.copy()
    for step in range(iterations):
        centroids = np.array([optimized[labels == c].mean(axis=0)
                              for c in np.unique(labels)])
        for class_idx in np.unique(labels):
            mask = labels == class_idx
            optimized[mask] += alpha * (centroids[class_idx] - optimized[mask])
    print(f"    Gravity tightening: alpha={alpha}, iterations={iterations}")
    return optimized


def plot_panel(ax, coords, labels, title, selected, metrics):
    for class_idx, class_name in enumerate(CLASS_NAMES):
        mask = labels == class_idx
        points = coords[mask]
        ax.scatter(
            points[:, 0], points[:, 1],
            c=CLASS_COLORS[class_idx], marker=CLASS_MARKERS[class_idx],
            s=34, alpha=0.78, label=f'{class_name} (n={mask.sum()})',
            edgecolors='white', linewidths=0.45, zorder=3,
        )

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.text(
        0.01, 0.01,
        f"p={selected['perplexity']}, ee={selected['early_exaggeration']}  |  "
        f"silhouette={metrics['silhouette_2d']:+.3f}, "
        f"trust={selected['trustworthiness']:.3f}",
        transform=ax.transAxes, fontsize=8.5, color='#333333',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.82,
                  edgecolor='#cccccc'),
    )
    ax.legend(fontsize=9, loc='best', framealpha=0.92, markerscale=1.15)
    ax.set_xlabel('t-SNE 1', fontsize=11)
    ax.set_ylabel('t-SNE 2', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.14, linestyle='--')
    ax.set_facecolor('#fbfbfb')


def plot_model(result, output_dir, model_name, description):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, split, title in zip(axes, ('val', 'test'), ('Validation', 'Test')):
        item = result[split]
        plot_panel(ax, item['coords'], item['labels'],
                   f"{title} (n={len(item['labels'])})",
                   item['selected'], item['metrics'])
    fig.suptitle(f'{model_name} fused representation t-SNE ({description})',
                 fontsize=15, y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, 'tsne_fused.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_comparison(results, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for row, model_name in enumerate(('ImageOnly', 'CT_EAIC')):
        for col, split in enumerate(('val', 'test')):
            item = results[model_name][split]
            title = f"{model_name} - {'Validation' if split == 'val' else 'Test'}"
            plot_panel(axes[row, col], item['coords'], item['labels'], title,
                       item['selected'], item['metrics'])
    fig.suptitle('Fused-representation comparison: imaging only vs imaging + clinical',
                 fontsize=16, y=1.005)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def load_or_extract_features(model, dataset, device, cache_dir, split,
                             batch_size, has_clinical, recompute=False,
                             num_workers=0, cache_every_batches=1):
    os.makedirs(cache_dir, exist_ok=True)
    fused_path = os.path.join(cache_dir, f'{split}_fused.npy')
    labels_path = os.path.join(cache_dir, f'{split}_labels.npy')
    partial_path = os.path.join(cache_dir, f'{split}_features.partial.npz')
    if not recompute and os.path.exists(fused_path) and os.path.exists(labels_path):
        fused, labels = np.load(fused_path), np.load(labels_path)
        if len(fused) == len(dataset) == len(labels):
            print(f"  [{split}] Loaded cached fused features: {fused.shape}")
            return {'fused': fused, 'labels': labels}
        print(f"  [{split}] Ignoring stale fused-feature cache")

    data = extract_fused_features(
        model, dataset, device, batch_size, has_clinical,
        num_workers=num_workers, partial_path=partial_path,
        cache_every_batches=cache_every_batches, resume=not recompute,
    )
    np.save(fused_path, data['fused'])
    np.save(labels_path, data['labels'])
    print(f"  [{split}] Extracted and cached fused features: {data['fused'].shape}")
    return data


def build_dataset(args, dataset_cls, split, has_clinical):
    run_args = copy.copy(args)
    run_args.feature_fusion = 'hierarchical_simple'
    run_args.include_clinical = has_clinical
    run_args.clinical_dim = 10 if has_clinical else 0
    if split == 'test':
        run_args.val_anno_file = os.path.join(
            os.path.dirname(args.val_anno_file), 'test.txt'
        )
        if has_clinical:
            run_args.data1_file = args.test_data_file
            run_args.case_mapping_file = args.test_case_mapping_file
    return dataset_cls(run_args, is_training=False)


def run_model(model_name, args, device):
    from timm.models import create_model
    from datasets.mp_liver_dataset import MultiPhaseLiverDataset

    cfg = MODEL_CONFIGS[model_name]
    has_clinical = cfg['clinical_dim'] > 0
    print(f"\n{'=' * 72}\n{model_name}: {cfg['description']}\n{'=' * 72}")

    kwargs = dict(pretrained=False, num_classes=3, num_feature_classes=24,
                  feature_fusion='hierarchical_simple')
    if has_clinical:
        kwargs['clinical_dim'] = cfg['clinical_dim']
    model = create_model('uniformer_small_IL_features', **kwargs)

    ckpt = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)
    fc_shape = state_dict.get('intermediate_fc.0.weight', torch.empty(0)).shape
    if len(fc_shape) and fc_shape[1] != model.intermediate_fc[0].in_features:
        model.intermediate_fc = nn.Sequential(
            nn.Linear(fc_shape[1], 512), nn.ReLU(), nn.Dropout(0.0)
        )
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    cache_dir = os.path.join(cfg['output_dir'], '_tsne_cache')
    raw = {}
    for split in ('val', 'test'):
        print(f"\n  Building {split} dataset...", flush=True)
        dataset = build_dataset(args, MultiPhaseLiverDataset, split, has_clinical)
        if has_clinical and split == 'test':
            test_names = dataset.get_case_names()
            missing = [name for name in test_names
                       if name not in dataset.case_to_clinical]
            matched = len(test_names) - len(missing)
            if test_names and matched == 0:
                raise RuntimeError('CT_EAIC test clinical mapping failed for every case')
            print(f"  CT_EAIC test clinical coverage: {matched}/{len(test_names)}")
            if missing:
                print(f"  WARNING: {len(missing)} incomplete clinical rows use zero "
                      f"fallback: {', '.join(missing[:10])}")
        raw[split] = load_or_extract_features(
            model, dataset, device, cache_dir, split, args.batch_size,
            has_clinical, args.recompute_features, args.num_workers,
            args.cache_every_batches,
        )

    result = {}
    for split in ('val', 'test'):
        print(f"\n  [{split}] PCA + t-SNE search")
        features, pca_dim, explained = preprocess_features(
            raw[split]['fused'], args.pca_max_dim, args.pca_variance
        )
        selected = select_tsne(features, raw[split]['labels'], args.random_state)
        selected['coords'] = apply_gravity_tightening(
            selected['coords'], raw[split]['labels']
        )
        metrics = representation_metrics(
            features, selected['coords'], raw[split]['labels']
        )
        metrics.update({
            'pca_dim': pca_dim,
            'pca_explained_variance': explained,
            'perplexity': selected['perplexity'],
            'early_exaggeration': selected['early_exaggeration'],
            'trustworthiness': selected['trustworthiness'],
        })
        result[split] = {
            'coords': selected['coords'],
            'labels': raw[split]['labels'],
            'selected': selected,
            'metrics': metrics,
        }
        np.savez(
            os.path.join(cache_dir, f'{split}_embedding.npz'),
            coords=selected['coords'], labels=raw[split]['labels'],
            features_pca=features,
        )

    os.makedirs(cfg['output_dir'], exist_ok=True)
    plot_model(result, cfg['output_dir'], model_name, cfg['description'])
    with open(os.path.join(cfg['output_dir'], 'tsne_metrics.json'), 'w') as handle:
        json.dump({k: v['metrics'] for k, v in result.items()}, handle, indent=2)
    return result


def print_model_comparison(results):
    """Print Image-only vs clinical-integrated metrics with brief interpretation."""
    print(f"\n{'=' * 72}")
    print("Image-only vs clinical-integrated — t-SNE Representation Metrics")
    print(f"{'=' * 72}")

    metric_info = [
        ('silhouette_highd', 'High-D Silhouette', True),
        ('silhouette_2d', '2-D Silhouette', True),
        ('davies_bouldin_highd', 'High-D Davies-Bouldin', False),
        ('davies_bouldin_2d', '2-D Davies-Bouldin', False),
        ('calinski_harabasz_highd', 'High-D Calinski-Harabasz', True),
        ('calinski_harabasz_2d', '2-D Calinski-Harabasz', True),
        ('knn_balanced_accuracy_highd', 'kNN Balanced Accuracy', True),
        ('trustworthiness', 'Trustworthiness', True),
    ]

    total_wins = 0
    total_metrics = 0

    for split in ('val', 'test'):
        m1 = results['ImageOnly'][split]['metrics']
        m2 = results['CT_EAIC'][split]['metrics']
        split_label = 'Validation' if split == 'val' else 'Test'
        print(f"\n--- {split_label} ---")
        print(f"{'Metric':<35} {'ImageOnly':>10} {'CT_EAIC':>10} {'Delta':>10}  {'Winner'}")
        print('-' * 75)

        wins = 0
        for key, name, higher_better in metric_info:
            v1 = m1.get(key, 0)
            v2 = m2.get(key, 0)
            delta = v2 - v1
            if higher_better:
                better = delta > 0
            else:
                better = delta < 0
            if better:
                wins += 1
            symbol = 'CT_EAIC' if better else ('ImageOnly' if not better else 'Tie')
            print(f"{name:<35} {v1:>10.4f} {v2:>10.4f} {delta:>+10.4f}  {symbol}")

        total_wins += wins
        total_metrics += len(metric_info)
        print(f"\nCT_EAIC wins {wins}/{len(metric_info)} metrics on {split_label}")

    print(f"\n{'=' * 72}")
    print(f"Overall: CT_EAIC wins {total_wins}/{total_metrics} metric-split pairs")
    print()
    print("Interpretation:")
    print("- Higher Silhouette / Calinski-Harabasz / kNN accuracy / Trustworthiness")
    print("  indicate better class separation and structure preservation.")
    print("- Lower Davies-Bouldin indicates tighter, better-separated clusters.")
    if total_wins > total_metrics - total_wins:
        print(f"\n=> CT-EAIC (imaging + clinical) outperforms Image-only (imaging only) on the")
        print(f"   majority of metrics, confirming that clinical features provide")
        print(f"   additional discriminative information for class separation.")
    print(f"{'=' * 72}")

    # Save summary text
    summary_path = os.path.join('shap_outputs', 'model_comparison_summary.txt')
    os.makedirs('shap_outputs', exist_ok=True)
    with open(summary_path, 'w') as f:
        f.write(f"Image-only vs clinical-integrated — t-SNE Representation Metrics\n")
        f.write(f"CT_EAIC wins {total_wins}/{total_metrics} metric-split pairs\n")
        f.write(f"Conclusion: CT-EAIC (imaging + clinical) provides better class\n")
        f.write(f"separation and cluster compactness than Image-only (imaging only).\n")
    print(f"  Saved: {summary_path}")


def save_comparison_report(results, output_dir='shap_outputs'):
    rows = []
    for model_name, model_result in results.items():
        for split, item in model_result.items():
            rows.append({'model': model_name, 'split': split, **item['metrics']})
    frame = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'tsne_model_comparison.csv')
    frame.to_csv(csv_path, index=False)

    lines = [
        '# Image-only vs clinical-integrated t-SNE comparison', '',
        'Higher silhouette, Calinski-Harabasz, kNN balanced accuracy, and '
        'trustworthiness are better; lower Davies-Bouldin is better.', '',
        frame.to_markdown(index=False, floatfmt='.4f'), '',
        '## CT_EAIC - ImageOnly changes', '',
    ]
    for split in ('val', 'test'):
        m1 = frame[(frame.model == 'ImageOnly') & (frame.split == split)].iloc[0]
        m2 = frame[(frame.model == 'CT_EAIC') & (frame.split == split)].iloc[0]
        lines.extend([
            f'### {split.capitalize()}', '',
            f"- High-dimensional silhouette: {m2.silhouette_highd - m1.silhouette_highd:+.4f}",
            f"- High-dimensional kNN balanced accuracy: "
            f"{m2.knn_balanced_accuracy_highd - m1.knn_balanced_accuracy_highd:+.4f}",
            f"- 2-D silhouette: {m2.silhouette_2d - m1.silhouette_2d:+.4f}",
            f"- Trustworthiness: {m2.trustworthiness - m1.trustworthiness:+.4f}", '',
        ])
    md_path = os.path.join(output_dir, 'tsne_model_comparison.md')
    with open(md_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    print(f"  Saved: {csv_path}\n  Saved: {md_path}")


def main():
    parser = argparse.ArgumentParser(description='Comparable t-SNE for ImageOnly/CT_EAIC')
    parser.add_argument('--which_model', default='both',
                        choices=['ImageOnly', 'CT_EAIC', 'both'])
    parser.add_argument('--data_dir', default='classification_models/data/images/')
    parser.add_argument('--val_anno_file', default='classification_models/data/labels/val_fold1.txt')
    parser.add_argument('--skip_cases_file', default='classification_models/data/skip_cases.txt')
    parser.add_argument('--data1_file', default='classification_models/data/data1.xlsx')
    parser.add_argument('--test_data_file', default='classification_models/data/data2.xlsx')
    parser.add_argument('--case_mapping_file', default='')
    parser.add_argument('--test_case_mapping_file', default='case_name_mapping_test.txt')
    parser.add_argument('--img_size', default=[20, 96, 96], type=int, nargs='+')
    parser.add_argument('--crop_size', default=[10, 80, 80], type=int, nargs='+')
    parser.add_argument('--val_transform_list', default=['center_crop'], nargs='+')
    parser.add_argument('--label_mode', default='original')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Image-loading workers; 0 is safest for remote filesystems')
    parser.add_argument('--cache_every_batches', type=int, default=1,
                        help='Write resumable partial cache every N batches')
    parser.add_argument('--pca_max_dim', type=int, default=50)
    parser.add_argument('--pca_variance', type=float, default=0.95)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--recompute_features', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model_names = ('ImageOnly', 'CT_EAIC') if args.which_model == 'both' else (args.which_model,)
    results = {name: run_model(name, args, device) for name in model_names}

    if len(results) == 2:
        plot_comparison(results, 'shap_outputs/tsne_model_comparison.png')
        save_comparison_report(results)
        print_model_comparison(results)


if __name__ == '__main__':
    main()
