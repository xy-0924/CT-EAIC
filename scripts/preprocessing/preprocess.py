"""
医学影像数据预处理脚本（增强版）：支持 ANTs 配准与断点续传。

功能：
    在 preprocess.py 基础上增加 ANTsPy 配准功能，可将不同模态图像配准到同一空间。
    支持断点续传：已处理的病例会自动跳过，未完成的可继续执行。
    支持保存配准后的中间结果，标签合并策略为并集。

使用示例：
    # 修改脚本底部的路径和参数后运行
    python preprocess_copy.py

    # 启用配准（默认关闭，配准步骤较耗时）
    # 将 enable_registration 设为 True

    # 在其他脚本中导入调用
    from preprocess_copy import preprocess
    preprocess(
        path_HCC='',
        path_noHCC='',
        path_liangxing='data/img/待传',
        path_nnUNet_raw='data/nnUNet_raw/Dataset003_HCC',
        min_merged_label_voxels=500,
        enable_registration=False,  # 默认关闭配准
    )
"""

from __future__ import annotations

from pathlib import Path
import re

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import ants


def build_case_file_index(case_dir: Path) -> dict[str, Path]:
    """提取病例文件夹中的所有 nii.gz 文件名并去后缀。"""
    file_index: dict[str, Path] = {}
    for p in case_dir.iterdir():
        if p.is_file() and p.name.lower().endswith(".nii.gz"):
            file_index[p.name[:-7].lower()] = p
    return file_index


def find_label_by_alias(case_files: dict[str, Path], alias: str) -> Path | None:
    """
    按 alias 查找标签文件（大小写不敏感，支持额外后缀）:
    - 精确: a-label
    - 前缀: a-label-*, a-label_*, a-label.*
    """
    base = f"{alias}-label"
    exact = case_files.get(base)
    if exact is not None:
        return exact

    valid_prefixes = (f"{base}-", f"{base}_", f"{base}.")
    for key in sorted(case_files.keys()):
        if key.startswith(valid_prefixes):
            return case_files[key]
    return None


def save_relabelled_mask(mask_img: nib.Nifti1Image, dst_label: Path, roi_value: int) -> None:
    """
    将原始标签重映射为:
    - 背景: 0
    - ROI: roi_value
    """
    mask = mask_img.get_fdata()
    relabelled = np.zeros(mask.shape, dtype=np.uint8)
    relabelled[mask > 0] = np.uint8(roi_value)
    out = nib.Nifti1Image(relabelled, affine=mask_img.affine, header=mask_img.header)
    nib.save(out, str(dst_label))


def resample_image_to_reference(
    src_img: nib.Nifti1Image,
    ref_img: nib.Nifti1Image,
    is_label: bool = False,
) -> nib.Nifti1Image:
    """
    将 src_img 重采样到 ref_img 的空间网格

    保证：
    - shape一致
    - spacing一致
    - orientation一致
    - affine一致

    Parameters
    ----------
    src_img : Nifti1Image
        原始图像
    ref_img : Nifti1Image
        参考图像
    is_label : bool
        是否为标签（决定插值方式）

    Returns
    -------
    Nifti1Image
        resampled image
    """

    # 1 将图像转换为标准RAS方向（右侧、前侧、上侧），确保坐标系一致
    try:
        src_img = nib.as_closest_canonical(src_img)
        ref_img = nib.as_closest_canonical(ref_img)
    except Exception as e:
        fname = getattr(src_img, 'get_filename', lambda: None)() or "unknown"
        raise RuntimeError(f"读取文件失败 (corrupted file): {fname}, error: {e}") from e

    # 2 如果 shape + affine 已经一致就不resample
    if (
        src_img.shape == ref_img.shape
        and np.allclose(src_img.affine, ref_img.affine)
    ):
        return src_img

    # 3 选择插值方式
    order = 0 if is_label else 1

    # 4 执行 resample
    resampled = resample_from_to(
        src_img,
        (ref_img.shape, ref_img.affine),
        order=order,
    )

    data = resampled.get_fdata()

    # 5 处理 dtype
    if is_label:
        data = np.rint(data)  # 防止0.999这种情况
        data = data.astype(np.uint8)
    else:
        data = data.astype(np.float32)

    # 6 创建新 NIfTI（避免 header 冲突）
    new_img = nib.Nifti1Image(data, ref_img.affine)

    # 7 保留 qform / sform
    new_img.set_qform(ref_img.affine)
    new_img.set_sform(ref_img.affine)

    return new_img


def register_modalities_with_ants(
    modality_imgs: dict[int, nib.Nifti1Image],
    label_imgs: dict[int, nib.Nifti1Image],
    ref_mod_idx: int = 0,
    tmp_dir: Path = None,
) -> tuple[dict[int, nib.Nifti1Image], dict[int, nib.Nifti1Image]]:
    """
    使用 ANTsPy 将其他模态配准到参考模态(默认为 A 通道)
    
    Parameters
    ----------
    modality_imgs : dict[int, nib.Nifti1Image]
        原始模态图像字典
    label_imgs : dict[int, nib.Nifti1Image]
        原始标签图像字典
    ref_mod_idx : int
        参考模态索引(默认为 0,即 A 通道)
    tmp_dir : Path
        临时文件目录
        
    Returns
    -------
    tuple[dict[int, nib.Nifti1Image], dict[int, nib.Nifti1Image]]
        配准后的模态图像和标签图像字典
    """
    import tempfile
    import shutil
    
    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    ref_img = modality_imgs[ref_mod_idx]
    
    # 保存参考图像为临时文件
    ref_tmp = tmp_dir / "ref.nii.gz"
    nib.save(ref_img, str(ref_tmp))
    ref_ants = ants.image_read(str(ref_tmp))
    
    registered_modalities = {ref_mod_idx: ref_img}
    registered_labels = {ref_mod_idx: label_imgs[ref_mod_idx]}
    
    for mod_idx in sorted(modality_imgs.keys()):
        if mod_idx == ref_mod_idx:
            continue
            
        src_img = modality_imgs[mod_idx]
        src_lbl = label_imgs[mod_idx]
        
        # 保存移动图像为临时文件
        src_tmp = tmp_dir / f"src_{mod_idx}.nii.gz"
        src_lbl_tmp = tmp_dir / f"src_lbl_{mod_idx}.nii.gz"
        nib.save(src_img, str(src_tmp))
        nib.save(src_lbl, str(src_lbl_tmp))
        
        src_ants = ants.image_read(str(src_tmp))
        src_lbl_ants = ants.image_read(str(src_lbl_tmp))
        
        # 使用刚性配准(可根据需要调整为仿射或非线性配准)
        print(f"[ANTs] Registering modality {mod_idx} to reference modality {ref_mod_idx}...")
        registration = ants.registration(
            fixed=ref_ants,
            moving=src_ants,
            type_of_transform='Rigid',  # 可选: 'Rigid', 'Affine', 'SyN'
            verbose=False
        )
        
        # 直接使用配准结果中的warped图像
        warped_mod_ants = registration['warpedmovout']
        warped_mod_tmp = tmp_dir / f"warped_mod_{mod_idx}.nii.gz"
        ants.image_write(warped_mod_ants, str(warped_mod_tmp))
        
        # 应用相同变换到标签图像
        warped_lbl_tmp = tmp_dir / f"warped_lbl_{mod_idx}.nii.gz"
        warped_lbl_ants = ants.apply_transforms(
            fixed=ref_ants,
            moving=src_lbl_ants,
            transformlist=registration['fwdtransforms'],
            interpolator='nearestNeighbor'
        )
        ants.image_write(warped_lbl_ants, str(warped_lbl_tmp))
        
        # 读取回 nibabel 格式并强制加载数据到内存
        warped_mod_nib = nib.load(str(warped_mod_tmp))
        warped_mod_data = warped_mod_nib.get_fdata().astype(np.float32)
        warped_mod_nib = nib.Nifti1Image(warped_mod_data, warped_mod_nib.affine)
        
        warped_lbl_nib = nib.load(str(warped_lbl_tmp))
        lbl_data = warped_lbl_nib.get_fdata()
        lbl_data = np.rint(lbl_data).astype(np.uint8)
        warped_lbl_nib = nib.Nifti1Image(lbl_data, warped_lbl_nib.affine)
        
        registered_modalities[mod_idx] = warped_mod_nib
        registered_labels[mod_idx] = warped_lbl_nib
        print(f"[ANTs] Successfully registered modality {mod_idx}")
    
    # 清理临时文件
    shutil.rmtree(tmp_dir, ignore_errors=True)
    
    return registered_modalities, registered_labels


def merge_labels_union(label_imgs: list[nib.Nifti1Image]) -> nib.Nifti1Image:
    """将同一病例的多个模态标签合并为二值前景(并集)。"""
    ref = label_imgs[0]
    merged = np.zeros(ref.shape, dtype=bool)
    for img in label_imgs:
        merged |= img.get_fdata() > 0
    merged = merged.astype(np.uint8)
    return nib.Nifti1Image(merged, affine=ref.affine, header=ref.header.copy())





def preprocess(
    path_HCC: str,
    path_noHCC: str,
    path_liangxing: str,
    path_nnUNet_raw: str,
    min_merged_label_voxels: int = 1,
    save_registered: bool = True,
    enable_registration: bool = False,
) -> None:
    """
    按顺序读取 path_HCC、path_noHCC、path_liangxing 下的病例。
    每个病例要求匹配 4 个模态及其对应 alias-label（a/d/p/s|ps）。
    在转存前，先根据空间信息将 4 个模态和 4 个标签统一到同一网格。

    标签映射:
    - path_HCC 的 ROI -> 1
    - path_noHCC 的 ROI -> 1
    - path_liangxing 的 ROI -> 1
    - 背景 -> 0
    """
    dst_images_ts = Path(path_nnUNet_raw) / "imagesTr"
    dst_labels_ts = Path(path_nnUNet_raw) / "labelsTr"
    dst_registered = Path(path_nnUNet_raw) / "registered_intermediate"
    mapping_txt = Path("case_name_mapping.txt")
    skip_txt = Path("附件/case_skip_data.txt")
    dst_images_ts.mkdir(parents=True, exist_ok=True)
    dst_labels_ts.mkdir(parents=True, exist_ok=True)
    if save_registered:
        dst_registered.mkdir(parents=True, exist_ok=True)

    # 从现有映射文件读取最大ID和已处理的病例
    case_idx = 1
    processed_cases = set()
    if mapping_txt.exists():
        existing_lines = mapping_txt.read_text(encoding="utf-8").strip().split("\n")
        if len(existing_lines) > 1:
            max_id = 0
            for line in existing_lines[1:]:  # 跳过表头
                parts = line.split("\t")
                if len(parts) >= 2:
                    match = re.search(r'case_(\d+)', parts[1])
                    if match:
                        max_id = max(max_id, int(match.group(1)))
                        processed_cases.add(parts[0])  # 记录原始病例名
            case_idx = max_id + 1
    
    # 初始化跳过文件
    skip_header = "group\tcase_folder\treason"
    skip_lines: list[str] = []

    required_modalities = [
        (["a"], 0),
        (["d"], 1),
        (["p"], 2),
        (["s", "ps"], 3),
    ]

    source_groups = [
        ("HCC", Path(path_HCC) if path_HCC else None, 1),
        ("noHCC", Path(path_noHCC) if path_noHCC else None, 1),
        ("liangxing", Path(path_liangxing) if path_liangxing else None, 1),
    ]

    header = "original_case_name\tconverted_case_name\tgroup"
    mapping_lines: list[str] = []
    
    # 如果映射文件已存在，先追加模式打开
    mapping_file_mode = "a" if mapping_txt.exists() else "w"
    if mapping_file_mode == "w":
        mapping_txt.write_text(header + "\n", encoding="utf-8")

    for group_name, src_root, roi_value in source_groups:
        if src_root is None or not src_root.exists():
            print(f"[SKIP] Source path does not exist: {src_root}")
            skip_lines.append(f"{group_name}\t{src_root}\tsource_path_not_exist")
            continue

        for case_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
            # 跳过已处理的病例
            if case_dir.name in processed_cases:
                print(f"[SKIP] Already processed: {case_dir.name} ({group_name})")
                continue
            
            case_files = build_case_file_index(case_dir) # 文件名转小写 去后缀

            matched_items: list[tuple[int, str, Path, Path]] = []
            missing_modalities: list[str] = []
            for aliases, mod_idx in required_modalities:
                selected: tuple[int, str, Path, Path] | None = None
                for alias in aliases:
                    img_file = case_files.get(alias)
                    label_file = find_label_by_alias(case_files, alias) #支持额外后缀的标签文件
                    if img_file is not None and label_file is not None:
                        selected = (mod_idx, alias, img_file, label_file)
                        break

                if selected is None:
                    missing_modalities.append("/".join(a.upper() for a in aliases))
                else:
                    matched_items.append(selected)

            if len(matched_items) != 4:
                missing_modalities_str = ",".join(missing_modalities)
                reason = f"missing_modality_or_label:{missing_modalities_str}"
                print(f"[SKIP] ({group_name}) {case_dir.name}: {reason}")
                skip_lines.append(f"{group_name}\t{case_dir.name}\t{reason}")
                continue

            matched_items.sort(key=lambda x: x[0])

            try:
                modality_imgs = {
                    mod_idx: nib.load(str(img_path))
                    for mod_idx, _, img_path, _ in matched_items
                }
                label_imgs = {
                    mod_idx: nib.load(str(label_path))
                    for mod_idx, _, _, label_path in matched_items
                }
            except Exception as e:
                reason = f"nifti_read_error:{e}"
                print(f"[SKIP] ({group_name}) {case_dir.name}: {reason}")
                skip_lines.append(f"{group_name}\t{case_dir.name}\t{reason}")
                continue

            # 以 A 通道(0000)作为统一空间参考
            ref_img = modality_imgs[0]
            
            if enable_registration:
                # 步骤1：检查模态配准文件是否存在
                registered_modalities = {}
                registered_labels = {}
                missing_mod_indices = []
                
                for mod_idx, alias, _, _ in matched_items:
                    reg_file = dst_registered / f"{case_dir.name}_mod{mod_idx:04d}_registered.nii.gz"
                    if reg_file.exists():
                        registered_modalities[mod_idx] = nib.load(str(reg_file))
                    else:
                        missing_mod_indices.append(mod_idx)
                
                # 步骤2：对缺失的模态进行配准，标签使用相同的配准变换
                if len(missing_mod_indices) == 0:
                    # 所有模态配准文件都存在，需要对标签应用相同的配准
                    print(f"[LOAD] Using existing registered modalities for case {case_dir.name}")
                    print(f"[ANTs] Registering labels using existing modality transforms for case {case_dir.name}...")
                    # 使用已配准的模态和原始标签，函数会计算配准变换并应用到标签
                    _, registered_labels = register_modalities_with_ants(
                        registered_modalities, label_imgs, ref_mod_idx=0
                    )
                    print(f"[ANTs] Completed label registration for case {case_dir.name}")
                elif len(missing_mod_indices) == 4:
                    # 全部需要配准
                    print(f"[ANTs] Starting registration for case {case_dir.name}...")
                    registered_modalities, registered_labels = register_modalities_with_ants(
                        modality_imgs, label_imgs, ref_mod_idx=0
                    )
                    print(f"[ANTs] Completed registration for case {case_dir.name}")
                else:
                    # 部分已有，部分需要配准
                    print(f"[LOAD] Using {4-len(missing_mod_indices)} existing registered modalities for case {case_dir.name}")
                    ref_mod_idx = [m for m in [0,1,2,3] if m not in missing_mod_indices][0]
                    
                    # 只对缺失的模态和标签进行配准
                    partial_modality_imgs = {m: modality_imgs[m] for m in missing_mod_indices}
                    partial_label_imgs = {m: label_imgs[m] for m in missing_mod_indices}
                    
                    print(f"[ANTs] Registering {len(missing_mod_indices)} modalities for case {case_dir.name}...")
                    new_registered_modalities, new_registered_labels = register_modalities_with_ants(
                        partial_modality_imgs, partial_label_imgs, ref_mod_idx=ref_mod_idx
                    )
                    print(f"[ANTs] Completed partial registration for case {case_dir.name}")
                    
                    # 合并已有和新的配准结果
                    registered_modalities.update(new_registered_modalities)
                    registered_labels.update(new_registered_labels)
                
                # 保存配准后的中间结果（只保存模态，标签不单独保存）
                if save_registered and len(missing_mod_indices) > 0:
                    for mod_idx in (missing_mod_indices if len(missing_mod_indices) < 4 else range(4)):
                        reg_file = dst_registered / f"{case_dir.name}_mod{mod_idx:04d}_registered.nii.gz"
                        nib.save(registered_modalities[mod_idx], str(reg_file))
            else:
                # 跳过配准，直接使用原始模态和标签
                print(f"[SKIP REGISTRATION] Using original modalities for case {case_dir.name}")
                registered_modalities = modality_imgs
                registered_labels = label_imgs
            
            # 步骤3：重采样 - 模态和标签使用相同的重采样操作
            resampled_modalities: dict[int, nib.Nifti1Image] = {}
            resampled_labels: list[nib.Nifti1Image] = []
            
            try:
                for mod_idx, alias, _, _ in matched_items:
                    src_mod = registered_modalities[mod_idx]
                    src_lbl = registered_labels[mod_idx]

                    # 模态和标签都重采样到参考空间
                    resampled_mod = resample_image_to_reference(src_mod, ref_img, is_label=False)
                    resampled_lbl = resample_image_to_reference(src_lbl, ref_img, is_label=True)

                    resampled_modalities[mod_idx] = resampled_mod
                    resampled_labels.append(resampled_lbl)
                    print(f"[RESAMPLE] ({group_name}) {case_dir.name} alias={alias.upper()} -> ref=A")
            except Exception as e:
                reason = f"resample_failed:{e}"
                print(f"[SKIP] ({group_name}) {case_dir.name} {reason}")
                skip_lines.append(f"{group_name}\t{case_dir.name}\t{reason}")
                case_idx += 1
                continue

            # 步骤4：合并4个模态的标签为并集，确保只有0和1
            merged_label = merge_labels_union(resampled_labels)
            merged_voxel_count = int(np.count_nonzero(merged_label.get_fdata()))
            converted_case_name = f"case_{case_idx:04d}"
            if merged_voxel_count < min_merged_label_voxels:
                reason = f"merged_label_too_small:{merged_voxel_count}<{min_merged_label_voxels}"
                print(
                    f"[SKIP SAVE] ({group_name}) {case_dir.name} {reason}"
                )
                skip_lines.append(f"{group_name}\t{case_dir.name}\t{reason}")
                case_idx += 1
                continue

            for mod_idx in range(4):
                dst_file = dst_images_ts / f"case_{case_idx:04d}_{mod_idx:04d}.nii.gz"
                nib.save(resampled_modalities[mod_idx], str(dst_file))
                print(f"[COPY] ({group_name}) {case_dir.name} modality_{mod_idx:04d} -> {dst_file}")
                
                # 仅在启用配准时保存配准后的中间结果
                if save_registered and enable_registration:
                    reg_file = dst_registered / f"{case_dir.name}_mod{mod_idx:04d}_registered.nii.gz"
                    nib.save(resampled_modalities[mod_idx], str(reg_file))

            dst_label = dst_labels_ts / f"case_{case_idx:04d}.nii.gz"
            save_relabelled_mask(merged_label, dst_label, roi_value=roi_value)
            print(
                f"[COPY] ({group_name}) {case_dir.name} merged_4labels -> {dst_label} "
                f"(background=0, roi={roi_value}, nonzero_voxels={merged_voxel_count})"
            )

            mapping_line = f"{case_dir.name}\t{converted_case_name}\t{group_name}"
            with mapping_txt.open("a", encoding="utf-8") as f:
                f.write(mapping_line + "\n")
            print(f"[SAVE] Mapped: {mapping_line}")
            case_idx += 1

    # 保存跳过记录
    skip_txt.parent.mkdir(parents=True, exist_ok=True)
    skip_txt.write_text(skip_header + "\n", encoding="utf-8")
    if skip_lines:
        with skip_txt.open("a", encoding="utf-8") as f:
            for line in skip_lines:
                f.write(line + "\n")
    print(f"[SAVE] Skip file: {skip_txt} (saved {len(skip_lines)} rows)")
    
    print(f"[SAVE] Mapping file: {mapping_txt} (started from case_{case_idx - len(mapping_lines):04d} to case_{case_idx - 1:04d})")

    group_counts = {"HCC": 0, "noHCC": 0, "liangxing": 0}
    for line in mapping_lines:
        parts = line.split("\t")
        if len(parts) >= 3 and parts[2] in group_counts:
            group_counts[parts[2]] += 1

    print("[COUNT] Processed cases in this run:")
    print(f"  HCC: {group_counts['HCC']}")
    print(f"  noHCC: {group_counts['noHCC']}")
    print(f"  liangxing: {group_counts['liangxing']}")


if __name__ == "__main__":
    path_HCC = ''
    path_noHCC = ''
    path_liangxing = 'data/img/待传'
    path_nnUNet_raw = 'data/nnUNet_raw/Dataset003_HCC'

    min_merged_label_voxels = 500
    enable_registration = False  # 设置为 True 可启用配准步骤
    preprocess(
        path_HCC,
        path_noHCC,
        path_liangxing,
        path_nnUNet_raw,
        min_merged_label_voxels=min_merged_label_voxels,
        enable_registration=enable_registration,
    )
