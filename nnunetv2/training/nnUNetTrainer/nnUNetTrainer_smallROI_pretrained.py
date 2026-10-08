import torch
import os
from batchgenerators.utilities.file_and_folder_operations import join, isfile
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_smallROI import nnUNetTrainer_smallROI


class nnUNetTrainer_smallROI_pretrained(nnUNetTrainer_smallROI):
    """
    支持从预训练模型加载权重的 Trainer
    用于增量学习：在 Dataset001 上预训练，然后在 Dataset004 上继续训练
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, 
                 unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        # 预训练模型路径（从环境变量或手动设置）
        self.pretrained_model_path = os.environ.get('nnUNet_pretrained_model_path', None)
        
    def initialize(self):
        """
        初始化网络，并加载预训练权重
        """
        super().initialize()
        
        if self.pretrained_model_path is not None and isfile(self.pretrained_model_path):
            self.load_pretrained_weights(self.pretrained_model_path)
    
    def load_pretrained_weights(self, pretrained_path: str):
        """
        加载预训练权重
        
        Args:
            pretrained_path: 预训练模型路径 (.pth 文件)
        """
        self.print_to_log_file("=" * 60)
        self.print_to_log_file(f"加载预训练权重: {pretrained_path}")
        
        try:
            # 加载预训练模型
            pretrained_dict = torch.load(pretrained_path, map_location=self.device)
            
            # 如果是完整的 checkpoint，提取模型状态
            if 'network_weights' in pretrained_dict:
                pretrained_dict = pretrained_dict['network_weights']
            elif 'state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['state_dict']
            
            # 获取当前模型状态
            model_dict = self.network.state_dict()
            
            # 过滤掉不匹配的层（如分类头）
            # 通常预训练模型和新模型的编码器部分可以共享
            pretrained_dict_filtered = {}
            for k, v in pretrained_dict.items():
                if k in model_dict:
                    if model_dict[k].shape == v.shape:
                        pretrained_dict_filtered[k] = v
                    else:
                        self.print_to_log_file(f"  跳过形状不匹配的层: {k}")
                else:
                    self.print_to_log_file(f"  跳过不存在的层: {k}")
            
            # 更新模型权重
            model_dict.update(pretrained_dict_filtered)
            self.network.load_state_dict(model_dict)
            
            self.print_to_log_file(f"成功加载 {len(pretrained_dict_filtered)}/{len(pretrained_dict)} 层预训练权重")
            self.print_to_log_file("=" * 60)
            
        except Exception as e:
            self.print_to_log_file(f"加载预训练权重失败: {e}")
            self.print_to_log_file("将从头开始训练")
            self.print_to_log_file("=" * 60)


class nnUNetTrainer_smallROI_finetune(nnUNetTrainer_smallROI_pretrained):
    """
    微调版本：加载预训练权重后，冻结部分层进行微调
    """
    
    def load_pretrained_weights(self, pretrained_path: str):
        """
        加载预训练权重并冻结编码器层
        """
        super().load_pretrained_weights(pretrained_path)
        
        self.print_to_log_file("=" * 60)
        self.print_to_log_file("启用微调模式：冻结编码器层")
        
        # 冻结编码器层（通常包含 'encoder', 'stages' 等关键字）
        frozen_layers = 0
        for name, param in self.network.named_parameters():
            # 冻结编码器相关层
            if any(keyword in name.lower() for keyword in ['encoder', 'stages', 'down']):
                param.requires_grad = False
                frozen_layers += 1
                self.print_to_log_file(f"  冻结: {name}")
        
        self.print_to_log_file(f"共冻结 {frozen_layers} 层参数")
        self.print_to_log_file("解码器层保持可训练")
        self.print_to_log_file("=" * 60)
