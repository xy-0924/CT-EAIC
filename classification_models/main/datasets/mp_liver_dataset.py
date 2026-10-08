import os
import re
import json
import torch
import numpy as np
import pandas as pd
from functools import partial
from timm.data.loader import _worker_init
from timm.data.distributed_sampler import OrderedDistributedSampler
try:
    from datasets.transforms import *
except:
    from transforms import *


# 临床变量列名（10维：data1.xlsx第5~14列，实验室指标）
CLINICAL_COL_NAMES = ['性别', '年龄', 'AFP', 'PLT', 'ALB', 'ALT', 'AST', 'ALP', 'TBIL', 'PT']
# 连续变量在10维clinical向量中的索引（性别=0, 年龄=1, AFP=2, 其余3~9为二分类）
CLINICAL_CONTINUOUS_INDICES = [1, 2]  # Age and AFP are continuous variables


def _resolve_data1_and_mapping_paths(args):
    """解析 data1.xlsx 和 case_name_mapping.txt 的路径
    
    支持多种运行目录（项目根目录 or classification_models/ 子目录）
    """
    data1_path = getattr(args, 'data1_file', 'classification_models/data/data1.xlsx')
    mapping_path = getattr(args, 'case_mapping_file', '')
    
    # 解析 data1_path：尝试多种候选路径
    if not os.path.exists(data1_path):
        candidates = [
            os.path.join('..', data1_path),  # 从 classification_models/ 运行时，classification_models/data/... -> ../classification_models/data/...
            data1_path.replace('classification_models/', '', 1),  # classification_models/data/data1.xlsx -> data/data1.xlsx
        ]
        for c in candidates:
            if os.path.exists(c):
                data1_path = c
                break
    
    # 解析 mapping_path
    if not mapping_path:
        data1_dir = os.path.dirname(data1_path)
        project_root = os.path.dirname(data1_dir) if data1_dir else '.'
        candidates = [
            'case_name_mapping.txt',
            os.path.join(project_root, 'case_name_mapping.txt'),
            '../case_name_mapping.txt',
            '../../case_name_mapping.txt',
            os.path.join('..', 'case_name_mapping.txt'),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                mapping_path = candidate
                break
    
    return data1_path, mapping_path


# Final study configuration uses all 24 predefined binary imaging features.
# The optional --selected-features argument is retained only for ablation/research use;
# it is not used for the final study models reported in the manuscript.

class MultiPhaseLiverDataset(torch.utils.data.Dataset):
    def __init__(self, args, is_training=True):
        self.args = args
        self.size = args.img_size
        self.is_training = is_training
        self.selected_features = getattr(args, 'selected_features', None)
        img_list = []
        lab_list = []
        phase_list = ['0000', '0001', '0002', '0003']  # 4模态

        # 加载标签
        if is_training:
            anno_path = args.train_anno_file
        else:
            anno_path = args.val_anno_file

        label_mode = getattr(args, 'label_mode', 'original')
        case_to_label, case_to_features, case_to_lr = self._load_labels(anno_path, args, label_mode=label_mode)

        # 从data1.xlsx补充征象标签（仅当anno文件中未找到feature列时）
        if not case_to_features:
            feature_labels_from_xlsx = self._load_feature_labels(args)
            case_to_features.update(feature_labels_from_xlsx)

        self.case_to_features = case_to_features
        self.case_to_lr = case_to_lr
        
        # 加载临床变量（10维实验室指标，不含肿瘤大小）
        self.case_to_clinical = self._load_clinical_features(args)
        
        # Z-score standardization is applied only to continuous variables (Age and AFP); sex and PLT/ALB/ALT/AST/ALP/TBIL/PT are binary 0/1 variables.
        if getattr(args, 'normalize_clinical', False) and self.case_to_clinical:
            self._normalize_clinical_features(args)
        
        # 如果指定了 selected_features，对征象标签进行筛选（可选消融；最终模型使用全部24个征象）
        if self.selected_features is not None and case_to_features:
            sel = self.selected_features
            for k in case_to_features:
                case_to_features[k] = case_to_features[k][sel]
            print(f"Filtered features to {len(sel)} indices: {sel}")
        
        # === 加载需要跳过的病例（多病灶/征象缺失） ===
        self.skip_cases = self._load_skip_cases(args)
        if self.skip_cases:
            print(f"Skip {len(self.skip_cases)} cases with missing/mismatched feature labels")

        # === 多病灶自动映射：为有子目录 _1/_2 的 base case 扩展标签字典 ===
        data_dir = args.data_dir
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        multi_lesion_count = 0
        for base_name in list(case_to_label.keys()):
            # 仅对没有直接目录的 base name 检查子目录
            base_path = os.path.join(data_dir, base_name)
            if os.path.isdir(base_path):
                continue  # 有直接目录，不需要映射
            # 检查是否存在 _1 子目录（使用主病灶）
            sub_name = f"{base_name}_1"
            sub_path = os.path.join(data_dir, sub_name)
            if os.path.isdir(sub_path) and sub_name not in case_to_label:
                case_to_label[sub_name] = case_to_label[base_name]
                if base_name in case_to_features:
                    case_to_features[sub_name] = case_to_features[base_name]
                if base_name in case_to_lr:
                    case_to_lr[sub_name] = case_to_lr[base_name]
                multi_lesion_count += 1
        if multi_lesion_count > 0:
            print(f"Multi-lesion mapping: added {multi_lesion_count} primary ROI entries")

        skipped_count = 0
        for case_name in sorted(os.listdir(data_dir)):
            case_path = os.path.join(data_dir, case_name)
            if not os.path.isdir(case_path):
                continue
            if case_name not in case_to_label:
                continue
            # 跳过无征象标签的病例
            if case_name in self.skip_cases:
                skipped_count += 1
                continue

            mp_img_list = []
            all_exist = True
            for phase in phase_list:
                img_path = os.path.join(case_path, f'{phase}.nii.gz')
                if not os.path.exists(img_path):
                    all_exist = False
                    break
                mp_img_list.append(img_path)

            if all_exist:
                img_list.append(mp_img_list)
                lab_list.append(case_to_label[case_name])
            else:
                print(f"  WARNING: case '{case_name}' is in labels but missing phase images, skipped")

        # 训练时应用下采样策略
        if is_training:
            img_list, lab_list = self._undersample_majority_class(img_list, lab_list, args)
            # 方案二：少数类重复采样（LR-3/LR-4 等）
            minority_repeat = getattr(args, 'minority_repeat', 1)
            if minority_repeat > 1:
                minority_classes = getattr(args, 'minority_classes', None)
                img_list, lab_list = self._oversample_minority(
                    img_list, lab_list,
                    target_classes=minority_classes,
                    repeat=minority_repeat,
                    label_mode=getattr(args, 'label_mode', 'original'))

        self.img_list = img_list
        self.lab_list = lab_list
        print(f"{'Train' if is_training else 'Val'} dataset: loaded {len(img_list)} samples (skipped {skipped_count})")

    def _load_labels(self, anno_path, args, label_mode='original'):
        """加载标签，支持原始patient_id格式（通过映射文件转换）或case_xxx直接格式

        Args:
            label_mode: 'original' 使用第2列 (0/1/2); 'lr' 使用第4列 (LR grade)

        Returns:
            (case_to_label, case_to_features, case_to_lr): 主标签字典、24维征象标签字典、LR等级字典
        """
        case_to_label = {}
        case_to_features = {}
        case_to_lr = {}

        if not os.path.exists(anno_path):
            raise FileNotFoundError(f"Annotation file not found: {anno_path}")

        # LR 标签字符串到整数的映射（按语义排序）
        lr_mapping = {
            '1/2': 0,
            '3': 1,
            '4': 2,
            '5': 3,
            'M': 4,
        }

        # 病理类别（良恶性）字符串到整数的映射
        pathology_mapping = {
            'liangxing': 0,
            'noHCC': 1,
            'HCC': 2,
        }

        # 使用 pandas 读取，兼容带header和不带header的格式
        try:
            # 先读取第一行判断是否为header
            with open(anno_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
            skip_header = 0
            if first_line.startswith('casename'):
                skip_header = 1
            df = pd.read_csv(anno_path, sep='\t', header=None, skiprows=skip_header)

            # 提前加载 skip_cases，过滤无效行（列数不足的病例）
            _early_skip = self._load_skip_cases(args)
            if _early_skip:
                df = df[~df.iloc[:, 0].astype(str).str.strip().isin(_early_skip)]

            # 丢弃列数不足（缺少LR标签）的行，避免后续解析报错
            min_cols = 3 if label_mode == 'lr' else 2
            if df.shape[1] < min_cols:
                raise ValueError(f"Annotation file has only {df.shape[1]} columns, need at least {min_cols}")
            incomplete_mask = df.iloc[:, min_cols - 1].isna() | (df.iloc[:, min_cols - 1].astype(str).str.strip() == '')
            if incomplete_mask.any():
                dropped = df.loc[incomplete_mask, df.columns[0]].tolist()
                print(f"WARNING: Dropping {len(dropped)} cases with missing labels: {dropped}")
                df = df[~incomplete_mask]

            if label_mode == 'lr':
                # 第3列是 "LR X" 格式（pandas 按 tab 分隔，空格保留在同一列中）
                raw_labels = dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 2].astype(str)))
                # 提取空格后的 LR 值并映射为整数
                raw_labels = {
                    k: lr_mapping.get(v.strip().split()[-1], -1)
                    for k, v in raw_labels.items()
                }
                # 检查是否有未映射的标签
                invalid = [k for k, v in raw_labels.items() if v == -1]
                if invalid:
                    raise ValueError(f"Unknown LR labels for cases: {invalid}")
            else:
                # original模式：第2列是病理类别字符串，需要映射为整数
                raw_labels_str = dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))
                raw_labels = {}
                for k, v in raw_labels_str.items():
                    v_clean = v.strip()
                    if v_clean in pathology_mapping:
                        raw_labels[k] = pathology_mapping[v_clean]
                    else:
                        raise ValueError(f"Unknown pathology label '{v_clean}' for case {k}. Expected: {list(pathology_mapping.keys())}")

            # 同时提取LR等级（第3列，无论label_mode是什么）
            raw_lr_labels = dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 2].astype(str)))
            for k, v in raw_lr_labels.items():
                lr_val = lr_mapping.get(v.strip().split()[-1], -1)
                if lr_val != -1:
                    case_to_lr[k] = lr_val

            # 提取征象标签（取最后24列）
            num_features = 24
            if df.shape[1] >= 28:
                for idx, row in df.iterrows():
                    case_name = str(row.iloc[0]).strip()
                    feat_vals = []
                    for col_idx in range(df.shape[1] - num_features, df.shape[1]):
                        val = row.iloc[col_idx]
                        feat_vals.append(float(val) if pd.notna(val) else 0.0)
                    case_to_features[case_name] = torch.tensor(feat_vals, dtype=torch.float32)
        except Exception:
            anno = np.loadtxt(anno_path, dtype=np.str_)
            if label_mode == 'lr':
                # numpy 默认按任意空白分隔，所以第4列是 LR 值
                raw_labels = dict(zip(anno[:, 0], anno[:, 3]))
                raw_labels = {
                    k: lr_mapping.get(v.strip(), -1)
                    for k, v in raw_labels.items()
                }
                invalid = [k for k, v in raw_labels.items() if v == -1]
                if invalid:
                    raise ValueError(f"Unknown LR labels for cases: {invalid}")
            else:
                raw_labels = dict(zip(anno[:, 0], anno[:, 1].astype(int)))

            # numpy读取时，同时提取LR等级
            if anno.shape[1] >= 4:
                for i in range(anno.shape[0]):
                    case_name = str(anno[i, 0]).strip()
                    lr_val = lr_mapping.get(anno[i, 2].strip().split()[-1], -1)
                    if lr_val != -1:
                        case_to_lr[case_name] = lr_val

            # numpy读取时，提取征象标签（取最后24列）
            num_features = 24
            if anno.shape[1] >= 28:
                for i in range(anno.shape[0]):
                    case_name = str(anno[i, 0]).strip()
                    feat_vals = []
                    for col_idx in range(anno.shape[1] - num_features, anno.shape[1]):
                        feat_vals.append(float(anno[i, col_idx]))
                    case_to_features[case_name] = torch.tensor(feat_vals, dtype=torch.float32)

        # 如果有映射文件，将 original_id -> label 转换为 case_xxx -> label
        mapping_file = getattr(args, 'case_mapping_file', None)
        if mapping_file and os.path.exists(mapping_file):
            try:
                map_df = pd.read_csv(mapping_file, sep='\t')
                mapped_labels = {}
                for _, row in map_df.iterrows():
                    original_id = str(row.iloc[0]).strip()
                    case_name = str(row.iloc[1]).strip()
                    if original_id in raw_labels:
                        mapped_labels[case_name] = raw_labels[original_id]

                # Some annotation files (notably labels/test.txt) already use
                # converted case_xxx names.  In that situation none of the raw
                # CT IDs in the mapping file can match, and replacing the label
                # dictionary would silently produce an empty dataset.  Only
                # apply the mapping when it actually mapped at least one row.
                case_to_label = mapped_labels if mapped_labels else raw_labels
            except Exception as e:
                print(f"Warning: failed to load mapping file: {e}")
                case_to_label = raw_labels
        else:
            # 假设 anno 中的 id 就是 case_xxx 文件夹名
            case_to_label = raw_labels

        return case_to_label, case_to_features, case_to_lr

    def _load_skip_cases(self, args):
        """加载需要跳过的病例列表（多病灶/征象缺失）"""
        skip_file = getattr(args, 'skip_cases_file', '')
        if skip_file and os.path.exists(skip_file):
            with open(skip_file, 'r') as f:
                # 只取每行的第一列（支持 Tab/空格分隔的多列格式）
                return set(line.strip().split()[0] for line in f if line.strip())
        return set()

    def _undersample_majority_class(self, img_list, lab_list, args):
        """对多数类进行随机下采样，平衡数据分布
        
        Args:
            img_list: 图像路径列表
            lab_list: 标签列表
            args: 参数配置，包含 max_samples_per_class（每类最大样本数）
        
        Returns:
            下采样后的 img_list, lab_list
        """
        max_samples = getattr(args, 'max_samples_per_class', 200)
        if max_samples is None or max_samples <= 0:
            return img_list, lab_list
        
        # 按类别分组
        class_indices = {}
        for idx, label in enumerate(lab_list):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        # 对超过阈值的类别进行随机采样
        new_indices = []
        for label, indices in class_indices.items():
            if len(indices) > max_samples:
                # 随机选择max_samples个样本
                import random
                random.seed(42)  # 固定随机种子保证可复现性
                selected = random.sample(indices, max_samples)
                new_indices.extend(selected)
                print(f"  Class {label}: {len(indices)} -> {len(selected)} samples (undersampled)")
            else:
                new_indices.extend(indices)
                print(f"  Class {label}: {len(indices)} samples (kept)")
        
        # 重新排序
        new_indices.sort()
        
        new_img_list = [img_list[i] for i in new_indices]
        new_lab_list = [lab_list[i] for i in new_indices]
        
        print(f"After undersampling: {len(new_img_list)} total samples")
        return new_img_list, new_lab_list

    def _oversample_minority(self, img_list, lab_list, target_classes=None, repeat=2, label_mode='original'):
        """方案二：对少数类进行重复采样（同一 case 以不同增强形式多次出现）

        与 WeightedRandomSampler 不同：此方法在 dataset 层面直接复制样本，
        保证每个 epoch 中少数类样本出现指定次数，配合不同的随机增强使用。

        Args:
            img_list: 图像路径列表
            lab_list: 标签列表
            target_classes: 需要过采样的类别索引列表。
                            None 时自动检测：lr 模式下为 [1, 2] (LR-3, LR-4)
            repeat: 重复次数 (2 = 原始1次 + 额外1次)
            label_mode: 标签模式 ('lr' 或 'original')

        Returns:
            过采样后的 img_list, lab_list
        """
        if repeat <= 1:
            return img_list, lab_list

        # 自动检测少数类
        if target_classes is None:
            if label_mode == 'lr':
                target_classes = [1, 2]  # LR-3, LR-4
            else:
                # original 模式：统计后取最少的两类
                from collections import Counter
                counts = Counter(lab_list)
                sorted_classes = sorted(counts.items(), key=lambda x: x[1])
                target_classes = [c for c, _ in sorted_classes[:2]]

        target_set = set(target_classes)

        extra_imgs, extra_labs = [], []
        for img, lab in zip(img_list, lab_list):
            if lab in target_set:
                for _ in range(repeat - 1):
                    extra_imgs.append(img)
                    extra_labs.append(lab)

        if extra_imgs:
            from collections import Counter
            orig_counts = Counter(lab_list)
            new_img_list = img_list + extra_imgs
            new_lab_list = lab_list + extra_labs
            new_counts = Counter(new_lab_list)
            print(f"  [方案二] Minority oversampling (repeat={repeat}):")
            for cls in sorted(target_set):
                print(f"    Class {cls}: {orig_counts[cls]} -> {new_counts[cls]} samples")
            print(f"    Total: {len(img_list)} -> {len(new_img_list)} samples")
            return new_img_list, new_lab_list

        return img_list, lab_list

    def _load_feature_labels(self, args):
        """从 data1.xlsx 加载 24 维征象标签
            
        映射路径：
        case_xxx → case_name_mapping[original_case_name] → data1.xlsx[ID] → 征象列17~41
        """
        feature_labels = {}
            
        # 1. 解析路径（兼容多种运行目录）
        data1_path, mapping_path = _resolve_data1_and_mapping_paths(args)
            
        if not os.path.exists(data1_path) or not mapping_path or not os.path.exists(mapping_path):
            print(f"Warning: data1.xlsx or mapping file not found, feature labels will be zeros (data1_path={data1_path}, mapping_path={mapping_path})")
            return feature_labels
        
        try:
            # data1.xlsx: 第0,1行为header，第2行开始为数据
            df1 = pd.read_excel(data1_path, header=None)
            # 建立CT编号 → data1行的映射（第0列是ID/CT编号）
            ct_to_row = {}
            for idx, val in df1.iloc[2:, 0].items():
                if pd.notna(val):
                    ct_to_row[str(val).strip()] = idx
            
            # 读取case_name_mapping
            map_df = pd.read_csv(mapping_path, sep='\t')
            
            for _, row in map_df.iterrows():
                case_name = str(row['converted_case_name']).strip()
                ct = str(row['original_case_name']).strip()
                
                if ct not in ct_to_row:
                    continue
                
                data1_row = ct_to_row[ct]
                # 提取24个征象列（第18~41列；不包含总APHE）
                feature_vals = []
                valid = True
                for col_idx in range(18, 42):
                    val = df1.iloc[data1_row, col_idx]
                    if pd.isna(val):
                        valid = False
                        break
                    feature_vals.append(float(val))
                
                if valid and len(feature_vals) == 24:
                    feature_labels[case_name] = torch.tensor(feature_vals, dtype=torch.float32)
        except Exception as e:
            print(f"Warning: failed to load feature labels: {e}")
        
        return feature_labels

    def _load_tumor_size(self, args):
        """从 data1.xlsx 加载肿瘤大小（连续变量，单位mm）
        
        肿瘤大小位于 data1.xlsx 第16列（0-indexed），子header行(row 2)标记为"大小（mm）"
        
        Returns:
            case_to_size: dict[case_name -> float (mm)]，缺失值用中位数填充
        """
        case_to_size = {}
        data1_path, mapping_path = _resolve_data1_and_mapping_paths(args)

        if not os.path.exists(data1_path) or not mapping_path or not os.path.exists(mapping_path):
            print(f"Warning: data1.xlsx or mapping file not found, tumor size will be default")
            return case_to_size

        try:
            df1 = pd.read_excel(data1_path, header=None)
            ct_to_row = {}
            for idx, val in df1.iloc[2:, 0].items():
                if pd.notna(val):
                    ct_to_row[str(val).strip()] = idx

            map_df = pd.read_csv(mapping_path, sep='\t')

            for _, row in map_df.iterrows():
                case_name = str(row['converted_case_name']).strip()
                ct = str(row['original_case_name']).strip()

                if ct not in ct_to_row:
                    continue

                data1_row = ct_to_row[ct]
                val = df1.iloc[data1_row, 16]  # 第16列: tumor size
                if pd.isna(val):
                    continue
                # 处理带单位的数值（如 '25mm'）
                if isinstance(val, str):
                    match = re.search(r'[-+]?[0-9]*\.?[0-9]+', val)
                    if match:
                        case_to_size[case_name] = float(match.group())
                else:
                    case_to_size[case_name] = float(val)

            print(f"Loaded tumor size for {len(case_to_size)} cases")
            if case_to_size:
                sizes = list(case_to_size.values())
                print(f"  Tumor size range: {min(sizes):.1f} ~ {max(sizes):.1f} mm, "
                      f"median: {np.median(sizes):.1f} mm")
        except Exception as e:
            print(f"Warning: failed to load tumor size: {e}")

        return case_to_size

    def _load_and_append_tumor_size(self, args):
        """加载肿瘤大小并追加到征象向量末尾（归一化到0~1范围）
        
        肿瘤大小除以100进行简单归一化（典型范围10~100mm → 0.1~1.0）
        缺失值用中位数填充
        追加后征象向量从9维变为10维（9个二分类征象 + 1个连续变量）
        """
        case_to_size = self._load_tumor_size(args)
        if not case_to_size:
            print("Warning: no tumor size data available, using 0 as default")
            median_size = 0.0
        else:
            median_size = np.median(list(case_to_size.values()))

        if not self.case_to_features:
            return

        for case_name in list(self.case_to_features.keys()):
            v = self.case_to_features[case_name]
            size_mm = case_to_size.get(case_name, median_size)
            # 归一化: 除以100使值在0~1范围
            size_normalized = size_mm / 100.0
            self.case_to_features[case_name] = torch.cat(
                [v, torch.tensor([size_normalized], dtype=torch.float32)]
            )

        feat_dim = len(next(iter(self.case_to_features.values())))
        print(f"Appended tumor size to feature vector (dim: {feat_dim - 1} -> {feat_dim})")

    def _load_clinical_features(self, args):
        """从data1.xlsx加载临床变量（实验室指标，10维）

        临床变量列（data1.xlsx）:
            第5列: 性别, 第6列: 年龄, 第7列: AFP, 第8列: PLT, 第9列: ALB,
            第10列: ALT, 第11列: AST, 第12列: ALP, 第13列: TBIL, 第14列: PT
        
        注意：肿瘤大小(tumor size)由 _load_and_append_tumor_size 加载并追加到征象向量末尾
        
        Returns:
            case_to_clinical: dict[case_name -> torch.tensor(10)]
        """
        clinical_features = {}

        # 解析路径（兼容多种运行目录）
        data1_path, mapping_path = _resolve_data1_and_mapping_paths(args)

        if not os.path.exists(data1_path) or not mapping_path or not os.path.exists(mapping_path):
            print(f"Warning: data1.xlsx or mapping file not found, clinical features will be zeros (data1_path={data1_path}, mapping_path={mapping_path})")
            return clinical_features

        try:
            df1 = pd.read_excel(data1_path, header=None)
            # 建立CT编号 → data1行的映射（第0列是ID/CT编号）
            ct_to_row = {}
            for idx, val in df1.iloc[2:, 0].items():
                if pd.notna(val):
                    ct_to_row[str(val).strip()] = idx

            map_df = pd.read_csv(mapping_path, sep='\t')

            for _, row in map_df.iterrows():
                case_name = str(row['converted_case_name']).strip()
                ct = str(row['original_case_name']).strip()

                if ct not in ct_to_row:
                    continue

                data1_row = ct_to_row[ct]
                # Extract 10 clinical variables (columns 5-14). Age and AFP are continuous; sex and PLT/ALB/ALT/AST/ALP/TBIL/PT are pre-encoded binary 0/1 variables.
                # 第5列是性别：'男'→0, '女'→1
                clinical_vals = []
                valid = True
                for col_idx in range(5, 15):
                    val = df1.iloc[data1_row, col_idx]
                    if pd.isna(val):
                        valid = False
                        break
                    # 性别列特殊处理
                    if col_idx == 5:
                        if isinstance(val, str):
                            val = val.strip()
                            if val == '男':
                                clinical_vals.append(0.0)
                            elif val == '女':
                                clinical_vals.append(1.0)
                            else:
                                valid = False
                                break
                        else:
                            clinical_vals.append(float(val))
                    else:
                        # 处理带单位的数值（如 '45岁'）
                        if isinstance(val, str):
                            match = re.search(r'[-+]?[0-9]*\.?[0-9]+', val)
                            if match:
                                clinical_vals.append(float(match.group()))
                            else:
                                valid = False
                                break
                        else:
                            clinical_vals.append(float(val))

                if valid and len(clinical_vals) == 10:
                    clinical_features[case_name] = torch.tensor(clinical_vals, dtype=torch.float32)
        except Exception as e:
            print(f"Warning: failed to load clinical features: {e}")

        return clinical_features

    def _normalize_clinical_features(self, args):
        """对临床变量中的连续变量做 z-score 标准化，二分类变量保持不变
        
        Continuous variables (Age idx=1, AFP idx=2): (x - mean) / std
        Binary variables (sex, PLT, ALB, ALT, AST, ALP, TBIL, PT): kept as 0/1
        
        训练集：计算 mean/std 并保存到 JSON
        验证/测试集：加载训练集的 mean/std 并应用
        """
        stats_file = getattr(args, 'clinical_stats_file', '') or os.path.join(
            getattr(args, 'output', 'classification_models/ckpts'), 'clinical_stats.json')
        
        if not self.case_to_clinical:
            return
        
        idx = CLINICAL_CONTINUOUS_INDICES
        
        if self.is_training:
            # 从训练集计算连续变量的统计量
            all_vals = torch.stack(list(self.case_to_clinical.values()))
            cont_vals = all_vals[:, idx]
            mean = cont_vals.mean(dim=0)
            std = cont_vals.std(dim=0)
            std = torch.clamp(std, min=1e-6)  # 防止除零
            
            # 保存到文件（供 val/test 加载）
            os.makedirs(os.path.dirname(os.path.abspath(stats_file)), exist_ok=True)
            stats_data = {
                'continuous_indices': idx,
                'mean': mean.tolist(),
                'std': std.tolist(),
                'feature_names': [CLINICAL_COL_NAMES[i] for i in idx],
            }
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, indent=2, ensure_ascii=False)
            
            print(f"Clinical z-score normalization (training set, n={len(all_vals)}):")
            for k, i in enumerate(idx):
                print(f"  {CLINICAL_COL_NAMES[i]}: mean={mean[k]:.2f}, std={std[k]:.2f}")
            print(f"  Binary features (unchanged): {[CLINICAL_COL_NAMES[i] for i in range(10) if i not in idx]}")
            print(f"  Stats saved to: {stats_file}")
        else:
            # 验证/测试集：加载训练集统计量
            if not os.path.exists(stats_file):
                print(f"Warning: clinical stats file not found: {stats_file}, skipping normalization")
                return
            with open(stats_file, 'r') as f:
                stats_data = json.load(f)
            mean = torch.tensor(stats_data['mean'], dtype=torch.float32)
            std = torch.tensor(stats_data['std'], dtype=torch.float32)
            std = torch.clamp(std, min=1e-6)
            print(f"Clinical z-score normalization loaded from: {stats_file}")
        
        # 对所有病例应用标准化（仅修改连续变量维度）
        for case_name in self.case_to_clinical:
            v = self.case_to_clinical[case_name].clone()
            v[idx] = (v[idx] - mean) / std
            self.case_to_clinical[case_name] = v

    def get_case_names(self):
        """返回与数据集索引顺序一致的 case name 列表"""
        return [os.path.basename(os.path.dirname(imgs[0])) for imgs in self.img_list]

    def __getitem__(self, index):
        args = self.args
        image = self.load_mp_images(self.img_list[index])
        if self.is_training:
            image = self.transforms(image, args.train_transform_list)
        else:
            image = self.transforms(image, args.val_transform_list)
        image = image.copy()
        label = int(self.lab_list[index])

        # 征象标签：若未加载feature数据，直接返回二元组，避免collate报错
        if self.case_to_features:
            case_name = os.path.basename(os.path.dirname(self.img_list[index][0]))
            features = self.case_to_features.get(case_name)
            if features is None:
                feat_dim = len(self.selected_features) if self.selected_features else 24
                features = torch.zeros(feat_dim, dtype=torch.float32)

            # LR多任务模式：返回 (image, label, features, lr_label[, clinical])
            use_lr_multitask = getattr(args, 'feature_fusion', None) in ('hierarchical', 'hierarchical_simple')
            include_clinical = getattr(args, 'include_clinical', False)
            if use_lr_multitask:
                lr = self.case_to_lr.get(case_name, -1)
                if lr < 0:
                    lr = 0  # fallback，避免负数索引报错
                # 同时需要临床变量：返回 5元组 (image, label, features, lr_label, clinical)
                if include_clinical:
                    clinical = self.case_to_clinical.get(case_name)
                    if clinical is None:
                        clinical_dim = getattr(args, 'clinical_dim', 10)
                        clinical = torch.zeros(clinical_dim, dtype=torch.float32)
                    return (image, label, features, lr, clinical)
                return (image, label, features, lr)

            # 额外特征：LR等级 + 临床变量（旧逻辑兼容）
            if include_clinical:
                clinical_dim = getattr(args, 'clinical_dim', 0)
                # clinical_dim 应为10维实验室指标（不含肿瘤大小，肿瘤大小已加入征象向量）
                extra_list = []
                # LR等级 (one-hot 编码，5类)
                lr = self.case_to_lr.get(case_name, -1)
                if lr >= 0:
                    lr_onehot = torch.zeros(5, dtype=torch.float32)
                    lr_onehot[lr] = 1.0
                    extra_list.append(lr_onehot)
                else:
                    extra_list.append(torch.zeros(5, dtype=torch.float32))
                # 临床变量（如果clinical_dim要求包含）
                if clinical_dim == 0 or clinical_dim > 5:
                    clinical = self.case_to_clinical.get(case_name)
                    if clinical is not None:
                        extra_list.append(clinical)
                    else:
                        extra_list.append(torch.zeros(clinical_dim if clinical_dim > 0 else 10, dtype=torch.float32))
                extra_features = torch.cat(extra_list, dim=0)
                return (image, label, features, extra_features)

            return (image, label, features)

        return (image, label)

    def load_mp_images(self, mp_img_list):
        mp_image = []
        for img in mp_img_list:
            image = load_nii_file(img)
            image = resize3D(image, self.size)
            image = image_normalization(image)
            mp_image.append(image[None, ...])
        mp_image = np.concatenate(mp_image, axis=0)
        return mp_image

    def transforms(self, mp_image, transform_list):
        args = self.args
        if 'channel_cutout' in transform_list:
            mp_image = random_channel_cutout(mp_image, cutout_num=args.cutcnum, p=args.cutcprob, mode=args.cutcmode)
        if 'center_crop' in transform_list:
            mp_image = center_crop(mp_image, args.crop_size)
        if 'random_crop' in transform_list:
            mp_image = random_crop(mp_image, args.crop_size)
        if 'z_flip' in transform_list:
            mp_image = random_flip(mp_image, mode='z', p=args.flip_prob)
        if 'x_flip' in transform_list:
            mp_image = random_flip(mp_image, mode='x', p=args.flip_prob)
        if 'y_flip' in transform_list:
            mp_image = random_flip(mp_image, mode='y', p=args.flip_prob)
        if 'rotation' in transform_list:
            mp_image = rotate(mp_image, args.angle)
        if 'random_intensity' in transform_list:
            mp_image = random_intensity(mp_image, 0.1, p=0.25)
        return mp_image

    def __len__(self):
        return len(self.img_list)

    def get_sample_weights(self, balance=True, target_samples=200, scale='sqrt', oversample_factor=1.0):
        """计算每个样本的采样权重（用于过采样/下采样平衡）
        
        Args:
            balance: 是否启用类别平衡采样
            target_samples: 每个类别的目标样本数（用于计算权重）
            scale: 权重缩放策略
                - 'linear': 原始线性反比（权重差异大，接近完全平衡）
                - 'sqrt': 平方根缩放（温和过采样，推荐）
                - 'log': 对数缩放（最温和）
            oversample_factor: 额外过采样乘数（>1时增强少数类权重）
        
        Returns:
            weights: 每个样本的采样权重 (List[float])
        """
        import math
        
        if not balance:
            return None
        
        # 统计每个类别的样本数
        from collections import Counter
        class_counts = Counter(self.lab_list)
        print(f"Class distribution for sampling: {dict(sorted(class_counts.items()))}")
        
        # 计算每个类别的基础权重: target_samples / class_count
        class_weights = {}
        for cls, count in class_counts.items():
            class_weights[cls] = target_samples / count
        
        # 应用缩放策略
        if scale == 'sqrt':
            # 平方根缩放：温和过采样（推荐）
            # 例如：原始权重4.0 → sqrt(4.0) = 2.0
            class_weights = {cls: math.sqrt(w) for cls, w in class_weights.items()}
        elif scale == 'log':
            # 对数缩放：最温和
            class_weights = {cls: math.log(w + 1) for cls, w in class_weights.items()}
        # 'linear' 不做额外处理
        
        # 应用过采样乘数
        if oversample_factor != 1.0:
            # 找到最大类别（通常是LR5或LR1/2）
            max_count = max(class_counts.values())
            for cls, count in class_counts.items():
                if count < max_count:
                    class_weights[cls] *= oversample_factor
        
        # 为每个样本分配权重
        sample_weights = [class_weights[label] for label in self.lab_list]
        
        # 归一化使权重和为1
        total_weight = sum(sample_weights)
        sample_weights = [w / total_weight for w in sample_weights]
        
        # 打印实际采样比例
        print(f"Sampling weights per class (scale={scale}):")
        for cls in sorted(class_weights):
            print(f"  Class {cls}: weight={class_weights[cls]:.3f}")
        
        return sample_weights

def create_loader(
        dataset=None,
        batch_size=1,
        is_training=False,
        num_aug_repeats=0,
        num_workers=1,
        distributed=False,
        collate_fn=None,
        pin_memory=False,
        persistent_workers=True,
        worker_seeding='all',
        use_balanced_sampler=False,  # 是否使用类别平衡采样器
        balanced_sampler_scale='sqrt',  # 采样缩放策略: 'linear', 'sqrt'(默认), 'log'
):

    sampler = None
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)
    else:
        assert num_aug_repeats == 0, "RepeatAugment not currently supported in non-distributed or IterableDataset use"

    # 类别平衡采样：对少数类过采样，对多数类下采样
    if use_balanced_sampler and is_training and hasattr(dataset, 'get_sample_weights'):
        sample_weights = dataset.get_sample_weights(balance=True, scale=balanced_sampler_scale)
        if sample_weights is not None:
            # WeightedRandomSampler会自动实现过采样/下采样平衡
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True  # 允许重复采样，实现过采样
            )
            print(f"Using WeightedRandomSampler for class balancing (scale={balanced_sampler_scale})")

    loader_args = dict(
        batch_size=batch_size,
        shuffle=not isinstance(dataset, torch.utils.data.IterableDataset) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
        worker_init_fn=partial(_worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
    try:
        loader = torch.utils.data.DataLoader(dataset, **loader_args)
    except TypeError as e:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = torch.utils.data.DataLoader(dataset, **loader_args)
    return loader
