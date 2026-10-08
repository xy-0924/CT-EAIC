import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size


class nnUNetTrainer_smallROI(nnUNetTrainer):
    """
    针对小目标 ROI 优化的 Trainer
    主要改进：
    1. 增加前景过采样比例
    2. 使用 Focal Loss 替代标准 Dice+CE，解决类别不平衡
    3. 增加小目标权重
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        # 1. 增加前景过采样比例（从 0.0 改为 0.5）
        self.oversample_foreground_percent = 0.5
        
    def configure_loss_function(self):
        """
        使用 Focal Loss + Dice Loss 组合，更关注难样本和小目标
        """
        from nnunetv2.training.loss.focal_loss import FocalDiceLoss
        
        # 使用 Focal + Dice 组合损失
        # Focal Loss 解决类别不平衡，Dice Loss 直接优化分割重叠度
        self.loss = FocalDiceLoss(
            focal_weight=0.5,
            dice_weight=0.5,
            gamma=2.0,  # 聚焦参数，越大越关注难样本
            alpha=0.25,  # 平衡参数
            batch_dice=False,  # 对每个样本单独计算，对小目标更敏感
            smooth=1e-5
        )
        
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """
        调整数据增强策略，减少对小目标的破坏
        """
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        
        # 减小旋转角度范围，避免小目标被旋转出 patch
        if dim == 2:
            do_dummy_2d_data_aug = False
            rotation_for_DA = (-15. / 360 * 2. * 3.141592653589793, 15. / 360 * 2. * 3.141592653589793)
            mirror_axes = (0, 1)
        elif dim == 3:
            do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > 1.5
            rotation_for_DA = (-15. / 360 * 2. * 3.141592653589793, 15. / 360 * 2. * 3.141592653589793)
            mirror_axes = (0, 1, 2)
        else:
            raise RuntimeError()
        
        # 使用更保守的缩放范围计算初始 patch size
        initial_patch_size = get_patch_size(patch_size[-dim:],
                                            rotation_for_DA,
                                            rotation_for_DA,
                                            rotation_for_DA,
                                            (0.85, 1.25))
        
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]
        
        self.print_to_log_file(f'do_dummy_2d_data_aug: {do_dummy_2d_data_aug}')
        self.inference_allowed_mirroring_axes = mirror_axes
        
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes


class nnUNetTrainer_smallROI_noMirroring(nnUNetTrainer_smallROI):
    """
    更保守的版本：禁用镜像翻转，对小目标更友好
    """
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, initial_patch_size = super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        self.inference_allowed_mirroring_axes = None  # 禁用镜像
        return rotation_for_DA, initial_patch_size
