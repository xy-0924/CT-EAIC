"""
classification数据预处理脚本
python classification_models/preprocess_classification_data.py

功能：
从nnUNet原始数据集中读取图像和标签，基于标签ROI中心裁切图像

输出：
- classification_models/data/images/: 每个患者一个文件夹，包含不同模态的裁切后图像
- classification_models/data/masks/: 每个患者一个文件夹，包含裁切后的mask
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage


def get_roi_bboxes_from_label(label_path, min_voxel_count=50):
    """
    从标签文件中提取所有ROI的外接框（bounding box）

    通过连通区域分析检测多个ROI，并过滤掉过小的区域。
    注意：nibabel读取的nifti数据维度为 (x, y, z)，其中 z 为层数方向。

    Args:
        label_path: 标签文件路径
        min_voxel_count: 最小体素数量阈值，小于此值的连通区域将被忽略

    Returns:
        bboxes: ROI外接框列表，每个元素为 (x_min, x_max, y_min, y_max, z_min, z_max)
                为闭区间（包含边界点）。按体积从大到小排序。
        affine: 图像的affine矩阵
    """
    label_img = nib.load(label_path)
    label_data = label_img.get_fdata()

    # 二值化标签数据
    binary_data = (label_data > 0).astype(int)

    # 连通区域分析
    labeled_array, num_features = ndimage.label(binary_data)

    if num_features == 0:
        raise ValueError(f"标签文件中没有ROI区域: {label_path}")

    bboxes_with_size = []
    for i in range(1, num_features + 1):
        # 找到第i个连通区域的坐标
        coords = np.where(labeled_array == i)
        voxel_count = len(coords[0])

        # nibabel读取的nifti数据维度为 (x, y, z)
        x_min = int(np.min(coords[0]))
        x_max = int(np.max(coords[0]))
        y_min = int(np.min(coords[1]))
        y_max = int(np.max(coords[1]))
        z_min = int(np.min(coords[2]))
        z_max = int(np.max(coords[2]))

        bboxes_with_size.append((voxel_count, (x_min, x_max, y_min, y_max, z_min, z_max)))

    # 按体积从大到小排序
    bboxes_with_size.sort(key=lambda x: x[0], reverse=True)
    
    # 过滤逻辑：
    # 1. 如果有多个连通区域，过滤掉小于阈值的
    # 2. 如果只有一个连通区域，无论大小都保留
    if len(bboxes_with_size) > 1:
        # 多个区域时，过滤掉过小的
        filtered = [item for item in bboxes_with_size if item[0] >= min_voxel_count]
        if len(filtered) == 0:
            # 如果全部都被过滤掉了，保留最大的那个
            filtered = [bboxes_with_size[0]]
        bboxes_with_size = filtered
    
    # 只返回bbox部分
    return [item[1] for item in bboxes_with_size], label_img.affine


def crop_image_by_bbox(image_path, bbox, margin=(5, 5, 5)):
    """
    基于ROI外接框+Margin裁切图像

    注意：nibabel读取的nifti数据维度为 (x, y, z)，其中 z 为层数方向。
    输出的cropped_data保持nibabel的 (x, y, z) 维度顺序。

    Args:
        image_path: 图像文件路径
        bbox: ROI外接框 (x_min, x_max, y_min, y_max, z_min, z_max)
        margin: 向外扩张的像素数 (dz, dy, dx)，对应 (z, y, x) 方向。注意：z轴不扩张，保持原始层数

    Returns:
        cropped_data: 裁切后的图像数据，形状为 (x_crop, y_crop, z_crop)
        affine: 图像的affine矩阵
        actual_bbox: 实际裁切使用的边界 (x_start, x_end, y_start, y_end, z_start, z_end)
    """
    img = nib.load(image_path)
    img_data = img.get_fdata()
    affine = img.affine

    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    dz_margin, dy_margin, dx_margin = margin

    # nibabel数据维度为 (x, y, z)
    # 向外扩张margin
    x_start = max(0, x_min - dx_margin)
    x_end = min(img_data.shape[0] - 1, x_max + dx_margin)
    y_start = max(0, y_min - dy_margin)
    y_end = min(img_data.shape[1] - 1, y_max + dy_margin)
    z_start = max(0, z_min - dz_margin)
    z_end = min(img_data.shape[2] - 1, z_max + dz_margin)

    # 裁切图像（使用 end+1 因为Python切片是右开区间）
    # 输出保持 (x, y, z) 维度顺序
    cropped_data = img_data[x_start:x_end+1, y_start:y_end+1, z_start:z_end+1]

    return cropped_data, affine, (x_start, x_end, y_start, y_end, z_start, z_end)


def extract_patient_id_from_filename(filename):
    """
    从文件名中提取患者ID
    
    Args:
        filename: 文件名，如 case_001_0000.nii.gz 或 case_001.nii.gz
        
    Returns:
        patient_id: 患者ID，如 case_001
    """
    # 移除 .nii.gz 后缀
    base_name = filename.replace('.nii.gz', '')
    
    # nnUNet图像文件格式: case_xxx_0000.nii.gz (有两个下划线)
    # nnUNet标签文件格式: case_xxx.nii.gz (只有一个下划线)
    # 通过统计下划线数量来判断
    if base_name.count('_') == 2:
        # 图像文件，去掉最后的模态编号
        parts = base_name.rsplit('_', 1)
        return parts[0]
    else:
        # 标签文件，直接返回
        return base_name


def preprocess_images(dataset_path, output_images_dir, output_masks_dir, margin=(0, 15, 15), case_ids=None):
    """
    预处理图像：基于标签ROI外接框+Margin裁切图像
    
    Args:
        dataset_path: nnUNet数据集路径，如 data/nnUNet_raw/Dataset001_HCC
        output_images_dir: 输出图像目录
        output_masks_dir: 输出mask目录
        margin: 向外扩张的像素数 (dz, dy, dx)
        case_ids: 指定要处理的case ID列表，如 ["case_001", "case_005"]。为None时处理全部。
    """
    images_dir = os.path.join(dataset_path, 'imagesTr')
    labels_dir = os.path.join(dataset_path, 'labelsTr')
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        raise FileNotFoundError(f"数据集路径不存在: {images_dir} 或 {labels_dir}")
    
    # 获取所有标签文件
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.nii.gz')]
    
    # 如果指定了case_ids，只保留对应的标签文件
    if case_ids is not None:
        label_files = [f for f in label_files if extract_patient_id_from_filename(f) in case_ids]
        print(f"指定处理 {len(case_ids)} 个case")
    
    print(f"找到 {len(label_files)} 个标签文件")
    
    # 创建输出目录
    os.makedirs(output_images_dir, exist_ok=True)
    
    # 按患者分组处理
    patient_dict = {}
    for label_file in label_files:
        patient_id = extract_patient_id_from_filename(label_file)
        if patient_id not in patient_dict:
            patient_dict[patient_id] = []
        patient_dict[patient_id].append(label_file)
    
    print(f"找到 {len(patient_dict)} 个患者")
    
    # 快速跳过已处理的case：比较输入和输出目录
    existing_img_dirs = set(os.listdir(output_images_dir)) if os.path.exists(output_images_dir) else set()
    existing_mask_dirs = set(os.listdir(output_masks_dir)) if os.path.exists(output_masks_dir) else set()
    
    def _has_patient_output(patient_id, dir_list):
        # 指定case_ids时不做跳过，强制重新处理
        if case_ids is not None:
            return False
        for d in dir_list:
            if d == patient_id or (d.startswith(patient_id + '_') and d[len(patient_id)+1:].isdigit()):
                return True
        return False
    
    patients_to_process = {}
    skipped_count = 0
    for patient_id in patient_dict:
        if _has_patient_output(patient_id, existing_img_dirs) and _has_patient_output(patient_id, existing_mask_dirs):
            skipped_count += 1
        else:
            patients_to_process[patient_id] = patient_dict[patient_id]
    
    if skipped_count > 0:
        print(f"快速跳过已处理的患者: {skipped_count} 个")
    print(f"需要处理的患者: {len(patients_to_process)} / {len(patient_dict)}")
    
    # 记录各患者裁切尺寸
    size_stats = []
    
    # 处理每个患者
    for patient_id in tqdm(patients_to_process.keys(), desc="处理患者"):
        # 获取该患者的标签文件（通常只有一个）
        label_file = patient_dict[patient_id][0]
        label_path = os.path.join(labels_dir, label_file)

        try:
            # 获取所有ROI外接框（过滤掉小于20个体素的区域）
            bboxes, label_affine = get_roi_bboxes_from_label(label_path, min_voxel_count=20)

            # 查找该患者的所有图像文件（不同模态）
            image_files = [f for f in os.listdir(images_dir)
                          if f.startswith(patient_id + '_') and f.endswith('.nii.gz')]

            # 对每个ROI分别处理
            for roi_idx, bbox in enumerate(bboxes):
                # 单个ROI时文件夹名保持原样，多个ROI时添加后缀 _1, _2, ...
                if len(bboxes) == 1:
                    roi_patient_id = patient_id
                else:
                    roi_patient_id = f"{patient_id}_{roi_idx + 1}"

                patient_output_dir = os.path.join(output_images_dir, roi_patient_id)
                mask_output_dir = os.path.join(output_masks_dir, roi_patient_id)

                # 检查是否已处理：图像和mask均存在则跳过
                already_processed = True
                if not os.path.isdir(patient_output_dir) or not os.path.isdir(mask_output_dir):
                    already_processed = False
                else:
                    # 检查所有模态图像是否已存在
                    for image_file in image_files:
                        modal_id = image_file.replace(patient_id + '_', '').replace('.nii.gz', '')
                        if not os.path.exists(os.path.join(patient_output_dir, f"{modal_id}.nii.gz")):
                            already_processed = False
                            break
                    # 检查mask是否已存在
                    if already_processed and not os.path.exists(os.path.join(mask_output_dir, f"{roi_patient_id}.nii.gz")):
                        already_processed = False
                if already_processed:
                    print(f"  跳过已处理: {roi_patient_id}")
                    continue

                os.makedirs(patient_output_dir, exist_ok=True)
                os.makedirs(mask_output_dir, exist_ok=True)

                for image_file in image_files:
                    image_path = os.path.join(images_dir, image_file)

                    # 提取模态编号
                    modal_id = image_file.replace(patient_id + '_', '').replace('.nii.gz', '')

                    # 裁切图像
                    cropped_data, image_affine, actual_bbox = crop_image_by_bbox(
                        image_path, bbox, margin
                    )

                    # 保存裁切后的图像
                    output_filename = f"{modal_id}.nii.gz"
                    output_path = os.path.join(patient_output_dir, output_filename)

                    cropped_img = nib.Nifti1Image(cropped_data, image_affine)
                    nib.save(cropped_img, output_path)

                    # 同步裁切并保存mask
                    label_path = os.path.join(labels_dir, label_file)
                    label_img = nib.load(label_path)
                    label_data = label_img.get_fdata()
                    
                    # 重新计算该ROI在原始标签中的坐标范围（用于裁切mask）
                    x_start, x_end, y_start, y_end, z_start, z_end = actual_bbox
                    cropped_mask = label_data[x_start:x_end+1, y_start:y_end+1, z_start:z_end+1]
                    
                    # Mask直接以患者ID命名，如 case_001.nii.gz
                    mask_output_filename = f"{roi_patient_id}.nii.gz"
                    mask_output_path = os.path.join(mask_output_dir, mask_output_filename)
                    cropped_mask_img = nib.Nifti1Image(cropped_mask, label_affine)
                    nib.save(cropped_mask_img, mask_output_path)

                # 记录裁切尺寸
                dx = actual_bbox[1] - actual_bbox[0] + 1
                dy = actual_bbox[3] - actual_bbox[2] + 1
                dz = actual_bbox[5] - actual_bbox[4] + 1
                size_stats.append((roi_patient_id, dz, dy, dx))

                if len(bboxes) > 1:
                    print(f"患者 {patient_id} 检测到 {len(bboxes)} 个ROI，"
                          f"第 {roi_idx + 1} 个ROI已保存到 {roi_patient_id}/")

        except Exception as e:
            print(f"处理患者 {patient_id} 时出错: {e}")
            continue
    
    # 输出裁切尺寸统计信息
    if size_stats:
        sizes = np.array(size_stats)[:, 1:].astype(int)
        print(f"\n裁切尺寸统计:")
        print(f"  Depth (Z轴): 最小={sizes[:, 0].min()}, 最大={sizes[:, 0].max()}, 均值={sizes[:, 0].mean():.1f}")
        print(f"  Height (Y轴): 最小={sizes[:, 1].min()}, 最大={sizes[:, 1].max()}, 均值={sizes[:, 1].mean():.1f}")
        print(f"  Width (X轴): 最小={sizes[:, 2].min()}, 最大={sizes[:, 2].max()}, 均值={sizes[:, 2].mean():.1f}")
    
    print(f"图像预处理完成！结果保存在: {output_images_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="classification数据预处理")
    parser.add_argument("--case", type=str, nargs="+", default=None,
                        help="指定要处理的case ID，如 --case case_001 case_005。不指定则处理全部")
    args = parser.parse_args()

    print("=" * 80)
    print("classification数据预处理")
    print("=" * 80)

    # 配置路径
    # __file__ = classification_models/preprocess/preprocess_classification_data.py
    # parent = classification_models/preprocess/, parent.parent = classification_models/, parent.parent.parent = 项目根目录
    project_root = Path(__file__).parent.parent.parent
    dataset_path = project_root / "data" / "nnUNet_raw" / "Dataset001_HCC"
    output_images_dir = project_root / "classification_models" / "data" / "images"
    output_masks_dir = project_root / "classification_models" / "data" / "masks"

    print("\n步骤1: 预处理图像（基于ROI外接框+Margin裁切）")
    print("-" * 80)
    preprocess_images(
        dataset_path=str(dataset_path),
        output_images_dir=str(output_images_dir),
        output_masks_dir=str(output_masks_dir),
        margin=(0, 15, 15),  # 只扩张XY平面，Z轴不扩张
        case_ids=args.case
    )

    print("\n" + "=" * 80)
    print("数据预处理完成！")
    print(f"图像保存在: {output_images_dir}")
    print(f"Mask保存在: {output_masks_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
