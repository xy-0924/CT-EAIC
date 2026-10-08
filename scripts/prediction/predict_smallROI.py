#!/usr/bin/env python3
"""
针对小目标 ROI 优化的 nnUNet 预测脚本。

功能：
    基于训练好的 nnUNet 模型对新数据进行分割预测，并针对小目标 ROI 进行优化：
    1. 支持 Test Time Augmentation (TTA) 提高小目标检测率
    2. 支持多折模型集成预测
    3. 后处理：移除小连通域噪声，保留真实小目标
    4. 支持填充小孔洞

使用示例：
    # 标准预测（使用所有 fold）
    python predict_smallROI.py -i /path/to/input -o /path/to/output -d Dataset001_HCC

    # 启用 TTA
    python predict_smallROI.py -i /path/to/input -o /path/to/output -d Dataset001_HCC --tta

    # 指定 fold 并调整最小连通域大小
    python predict_smallROI.py -i /path/to/input -o /path/to/output -d Dataset001_HCC -f "0,1" --min_component_size 50
"""

import os
import argparse
import numpy as np
import torch
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isdir, subfiles
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import SimpleITK as sitk
from scipy import ndimage


def remove_small_connected_components(segmentation: np.ndarray, min_size: int = 50) -> np.ndarray:
    """
    移除小的连通域，但保留可能是真实小目标的区域
    
    Args:
        segmentation: 分割结果
        min_size: 最小连通域大小（体素数）
    
    Returns:
        处理后的分割结果
    """
    result = np.zeros_like(segmentation)
    
    # 对每个类别分别处理（跳过背景 0）
    for class_label in np.unique(segmentation)[1:]:
        class_mask = (segmentation == class_label).astype(np.int32)
        
        # 标记连通域
        labeled_array, num_features = ndimage.label(class_mask)
        
        # 计算每个连通域的大小
        component_sizes = ndimage.sum(class_mask, labeled_array, range(1, num_features + 1))
        
        # 保留大于 min_size 的连通域
        for i, size in enumerate(component_sizes, 1):
            if size >= min_size:
                result[labeled_array == i] = class_label
    
    return result


def fill_small_holes(segmentation: np.ndarray, max_hole_size: int = 100) -> np.ndarray:
    """
    填充小孔洞，有助于保持小目标的完整性
    """
    result = segmentation.copy()
    
    for class_label in np.unique(segmentation)[1:]:
        class_mask = (segmentation == class_label).astype(np.int32)
        
        # 找到孔洞（背景中的连通域）
        inverted = 1 - class_mask
        labeled_array, num_features = ndimage.label(inverted)
        
        # 计算每个孔洞的大小
        hole_sizes = ndimage.sum(inverted, labeled_array, range(1, num_features + 1))
        
        # 填充小孔洞
        for i, size in enumerate(hole_sizes, 1):
            if size <= max_hole_size:
                result[labeled_array == i] = class_label
    
    return result


def predict_with_tta(predictor: nnUNetPredictor, input_image: np.ndarray, 
                     do_mirroring: bool = True) -> np.ndarray:
    """
    使用 Test Time Augmentation (TTA) 进行预测
    通过对输入图像进行翻转，然后平均结果，提高小目标检测率
    """
    # 基础预测
    logits = predictor.predict_single_npy_array(
        input_image,
        predictor.plans_manager,
        predictor.configuration_manager,
        predictor.label_manager,
        predictor.dataset_json,
        do_mirroring=do_mirroring
    )
    
    return logits


def main():
    parser = argparse.ArgumentParser(description='小目标 ROI 优化预测脚本')
    parser.add_argument('-i', '--input_folder', type=str, required=True,
                        help='输入图像文件夹')
    parser.add_argument('-o', '--output_folder', type=str, required=True,
                        help='输出分割结果文件夹')
    parser.add_argument('-d', '--dataset', type=str, default='Dataset001_HCC',
                        help='数据集名称')
    parser.add_argument('-f', '--folds', type=str, default='all',
                        help='使用的 fold，如 "all" 或 "0,1,2,3,4"')
    parser.add_argument('--tta', action='store_true',
                        help='启用 Test Time Augmentation')
    parser.add_argument('--min_component_size', type=int, default=30,
                        help='最小连通域大小（体素数），小于此值的将被移除')
    parser.add_argument('--fill_holes', action='store_true',
                        help='填充小孔洞')
    parser.add_argument('--step_size', type=float, default=0.5,
                        help='滑动窗口步长（越小越精确但越慢，默认 0.5）')
    
    args = parser.parse_args()
    
    # 设置路径
    nnUNet_results = os.environ.get('nnUNet_results', './data/nnUNet_results')
    model_folder = join(nnUNet_results, args.dataset, 'nnUNetTrainer_smallROI__nnUNetPlans__2d')
    
    if not isdir(model_folder):
        print(f"错误: 模型文件夹不存在: {model_folder}")
        print("请确保已经使用 nnUNetTrainer_smallROI 完成训练")
        return
    
    # 创建输出文件夹
    os.makedirs(args.output_folder, exist_ok=True)
    
    # 初始化预测器
    print("初始化预测器...")
    predictor = nnUNetPredictor(
        tile_step_size=args.step_size,
        use_gaussian=True,
        use_mirroring=args.tta,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    
    # 初始化模型
    if args.folds.lower() == 'all':
        folds = (0, 1, 2, 3, 4)
    else:
        folds = [int(f) for f in args.folds.split(',')]
    
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=folds,
        checkpoint_name='checkpoint_final.pth'
    )
    
    print(f"使用 folds: {folds}")
    print(f"TTA: {'启用' if args.tta else '禁用'}")
    print(f"最小连通域大小: {args.min_component_size}")
    print("")
    
    # 获取输入文件列表
    input_files = subfiles(args.input_folder, suffix='.nii.gz', join=False)
    
    print(f"开始预测 {len(input_files)} 个文件...")
    
    for i, filename in enumerate(input_files):
        print(f"[{i+1}/{len(input_files)}] 处理: {filename}")
        
        input_path = join(args.input_folder, filename)
        output_path = join(args.output_folder, filename)
        
        # 读取输入图像
        image, props = SimpleITKIO().read_images([input_path])
        
        # 预测
        if args.tta:
            # 使用 TTA
            segmentation = predictor.predict_single_npy_array(
                image,
                predictor.plans_manager,
                predictor.configuration_manager,
                predictor.label_manager,
                predictor.dataset_json,
                do_mirroring=True
            )
        else:
            # 标准预测
            segmentation = predictor.predict_single_npy_array(
                image,
                predictor.plans_manager,
                predictor.configuration_manager,
                predictor.label_manager,
                predictor.dataset_json,
                do_mirroring=False
            )
        
        # 后处理
        if args.min_component_size > 0:
            segmentation = remove_small_connected_components(
                segmentation, 
                min_size=args.min_component_size
            )
        
        if args.fill_holes:
            segmentation = fill_small_holes(segmentation)
        
        # 保存结果
        SimpleITKIO().write_seg(segmentation, output_path, props)
        print(f"  已保存: {output_path}")
    
    print("")
    print(f"预测完成! 结果保存在: {args.output_folder}")


if __name__ == '__main__':
    main()