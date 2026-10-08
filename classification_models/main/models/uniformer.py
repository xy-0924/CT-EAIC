# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from collections import OrderedDict
from distutils.fancy_getopt import FancyGetopt
from re import M
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import math
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath, to_2tuple


layer_scale = False
init_value = 1e-6


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm3d(dim)
        self.conv1 = nn.Conv3d(dim, dim, 1)
        self.conv2 = nn.Conv3d(dim, dim, 1)
        self.attn = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        global layer_scale
        self.ls = layer_scale
        if self.ls:
            global init_value
            print(f"Use layer_scale: {layer_scale}, init_values: {init_value}")
            self.gamma_1 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_value * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        if self.ls:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, C, D, H, W )
        return x        
   

class head_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(head_embedding, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 2, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels // 2),
            nn.GELU(),
            nn.Conv3d(out_channels // 2, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class middle_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(middle_embedding, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=None, padding=0):
        super().__init__()
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        # num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        # self.img_size = img_size
        # self.patch_size = patch_size
        # self.num_patches = num_patches
        if stride is None:
            stride = patch_size
        else:
            stride = stride
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x

# TODO: add aux-head
class UniFormer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, depth=[3, 4, 8, 3], img_size=224, in_chans=3, num_classes=1000, embed_dim=[64, 128, 320, 512],
                 head_dim=64, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, conv_stem=False, aux_head=False,
                 num_feature_classes=0, video_mode=False, feature_fusion=None, clinical_dim=0, clinical_scale_init=1.0,
                 head_drop_rate=0., fusion_hidden_dim=512):
        """
        Args:
            depth (list): depth of each stage
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (list): embedding dimension of each stage
            head_dim (int): head dimension
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer (nn.Module): normalization layer
            conv_stem (bool): whether use overlapped patch stem
            video_mode (bool): whether to use video-compatible patch embed config (k400 pretrained)
            fusion_hidden_dim (int): hidden dimension for intermediate_fc and classification head in fusion modes
                (default: 512, matching original design; set to 128-256 to reduce overfitting on small datasets)
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) 
        self.aux_head = aux_head
        if conv_stem:
            self.patch_embed1 = head_embedding(in_channels=in_chans, out_channels=embed_dim[0])
            # self.patch_embed2 = middle_embedding(in_channels=embed_dim[0], out_channels=embed_dim[1])
            # self.patch_embed3 = middle_embedding(in_channels=embed_dim[1], out_channels=embed_dim[2])
            # self.patch_embed4 = middle_embedding(in_channels=embed_dim[2], out_channels=embed_dim[3])
    
            self.patch_embed2 = middle_embedding(in_channels=embed_dim[0], out_channels=embed_dim[1])
            self.patch_embed3 = middle_embedding(in_channels=embed_dim[1], out_channels=embed_dim[2], stride=(1, 2, 2))
            self.patch_embed4 = middle_embedding(in_channels=embed_dim[2], out_channels=embed_dim[3], stride=(1, 2, 2))

        elif video_mode:
            # Video-compatible patch embed config to match kinetics-400 pretrained weights
            self.patch_embed1 = PatchEmbed(
                img_size=img_size, patch_size=(3, 4, 4), in_chans=in_chans, embed_dim=embed_dim[0],
                stride=(2, 4, 4), padding=(1, 0, 0))
            self.patch_embed2 = PatchEmbed(
                img_size=img_size // 4, patch_size=(1, 2, 2), in_chans=embed_dim[0], embed_dim=embed_dim[1],
                stride=(1, 2, 2))
            self.patch_embed3 = PatchEmbed(
                img_size=img_size // 8, patch_size=(1, 2, 2), in_chans=embed_dim[1], embed_dim=embed_dim[2],
                stride=(1, 2, 2))
            self.patch_embed4 = PatchEmbed(
                img_size=img_size // 16, patch_size=(1, 2, 2), in_chans=embed_dim[2], embed_dim=embed_dim[3],
                stride=(1, 2, 2))

        else:
            self.patch_embed1 = PatchEmbed(
                img_size=img_size, patch_size=2, in_chans=in_chans, embed_dim=embed_dim[0])
            self.patch_embed2 = PatchEmbed(
                img_size=img_size // 4, patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1])
            self.patch_embed3 = PatchEmbed(
                img_size=img_size // 8, patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2], stride=(1, 2, 2))
            self.patch_embed4 = PatchEmbed(
                img_size=img_size // 16, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3], stride=(1, 2, 2))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dim]
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]], norm_layer=norm_layer)
            for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([
            SABlock(
                dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]], norm_layer=norm_layer)
            for i in range(depth[2])])
        self.blocks4 = nn.ModuleList([
            SABlock(
                dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i+depth[0]+depth[1]+depth[2]], norm_layer=norm_layer)
        for i in range(depth[3])])
        self.norm = nn.BatchNorm3d(embed_dim[-1])
        
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Feature fusion mode: None (default), 'hierarchical', or 'hierarchical_simple'
        self.feature_fusion = feature_fusion
        
        # Hierarchical mode: 征象概率 → 中间FC → 病灶分类（两级全连接结构）
        self.num_feature_classes = num_feature_classes
        self.clinical_dim = clinical_dim  # 临床变量维度（如10维实验室指标）
        # 临床特征缩放系数：可学习参数，用于增强临床特征在融合中的权重
        # 初始值由 clinical_scale_init 控制，默认为 1.0（不改变现有行为）
        if clinical_dim > 0 and feature_fusion in ('hierarchical', 'hierarchical_simple'):
            self.clinical_scale = nn.Parameter(torch.tensor(float(clinical_scale_init)))
        else:
            self.clinical_scale = None
        # === Late Fusion 模式：临床注意力调节 + 残差连接 ===
        # 临床特征不独立预测，而是生成注意力权重来调节主分支融合特征
        # 残差连接确保主分支信号不受破坏，临床特征只能"锦上添花"
        self.late_fusion = (feature_fusion == 'late_fusion')
        if self.late_fusion and clinical_dim > 0:
            # 临床注意力模块：输入 = [主分支融合特征(512) + 临床特征(clinical_dim)]
            # 输出 = 注意力权重(512)，用于逐通道调节主分支特征
            self.clinical_attention = nn.Sequential(
                nn.Linear(fusion_hidden_dim + clinical_dim, 128),
                nn.ReLU(),
                nn.Linear(128, fusion_hidden_dim),
                nn.Sigmoid(),  # 输出 [0,1] 注意力权重
            )
            # 可学习门控：控制临床注意力的整体强度，初始值0（零初始化，训练初期不干扰主分支）
            self.late_fusion_gate = nn.Parameter(torch.tensor(0.0))
        if self.feature_fusion == 'hierarchical' and num_feature_classes > 0:
            # 第一级全连接层：征象分类器（sigmoid输出征象概率），基于GAP特征
            self.feature_head = None
            self.feature_fc_head = nn.Sequential(
                nn.Linear(embed_dim[-1], 512),
                nn.ReLU(),
                nn.Dropout(head_drop_rate),
                nn.Linear(512, num_feature_classes)
            )
            # LR等级预测头：从GAP特征预测5类LR等级（辅助任务）
            self.lr_head = nn.Linear(embed_dim[-1], 5)
            # 中间全连接层：融合骨干特征 + 征象概率 + LR等级one-hot + 临床变量
            fusion_dim = embed_dim[-1] + num_feature_classes + 5 + clinical_dim
            self.intermediate_fc = nn.Sequential(
                nn.Linear(fusion_dim, fusion_hidden_dim),
                nn.ReLU(),
                nn.Dropout(head_drop_rate),
            )
            # 第二级全连接层：病灶分类器
            self.head = nn.Linear(fusion_hidden_dim, num_classes) if num_classes > 0 else nn.Identity()
            if self.aux_head:
                self.auxiliary_head = nn.Linear(fusion_hidden_dim, 2) if num_classes > 0 else nn.Identity()
        elif self.feature_fusion in ('hierarchical_simple', 'late_fusion') and num_feature_classes > 0:
            # 简化层级模式（兼容旧checkpoint）：仅融合骨干特征 + 征象概率，无LR头
            # late_fusion模式：主分支仅用图像+征象，临床特征通过独立分支晚期融合
            self.feature_head = None
            self.feature_fc_head = nn.Sequential(
                nn.Linear(embed_dim[-1], 512),
                nn.ReLU(),
                nn.Dropout(head_drop_rate),
                nn.Linear(512, num_feature_classes)
            )
            if self.late_fusion:
                # late_fusion：主分支不拼接临床特征
                fusion_dim = embed_dim[-1] + num_feature_classes
            else:
                # hierarchical_simple：主分支拼接临床特征
                fusion_dim = embed_dim[-1] + num_feature_classes + clinical_dim
            self.intermediate_fc = nn.Sequential(
                nn.Linear(fusion_dim, fusion_hidden_dim),
                nn.ReLU(),
                nn.Dropout(head_drop_rate),
            )
            # 第二级全连接层：病灶分类器
            self.head = nn.Linear(fusion_hidden_dim, num_classes) if num_classes > 0 else nn.Identity()
            if self.aux_head:
                self.auxiliary_head = nn.Linear(fusion_hidden_dim, 2) if num_classes > 0 else nn.Identity()
        else:
            # Classifier head (feature_fusion=None)
            self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
            if self.aux_head:
                self.auxiliary_head = nn.Linear(embed_dim[-1], 2) if num_classes > 0 else nn.Identity()
            # Feature head: 征象预测头，放在GAP之前，基于3D特征图
            if num_feature_classes > 0:
                self.feature_head = nn.Conv3d(embed_dim[-1], num_feature_classes, kernel_size=1)
            else:
                self.feature_head = None
        
        self.head_drop = nn.Dropout(head_drop_rate)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # if isinstance(m, nn.Linear):
        #     trunc_normal_(m.weight, std=.02)
        #     if isinstance(m, nn.Linear) and m.bias is not None:
        #         nn.init.constant_(m.bias, 0)
        # if isinstance(m, nn.LayerNorm):
        #     nn.init.constant_(m.bias, 0)
        #     nn.init.constant_(m.weight, 1.0)
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def freeze_backbone(self):
        """冻结 backbone 所有参数，只保留 head/feature_head/fusion 层可训练"""
        frozen_modules = [
            self.patch_embed1, self.patch_embed2, self.patch_embed3, self.patch_embed4,
            self.blocks1, self.blocks2, self.blocks3, self.blocks4,
            self.norm, self.pre_logits,
        ]
        for module in frozen_modules:
            for param in module.parameters():
                param.requires_grad = False
        trainable = [n for n, p in self.named_parameters() if p.requires_grad]
        return trainable

    def forward_features(self, x):
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
        x = self.patch_embed4(x)
        for blk in self.blocks4:
            x = blk(x)
        x = self.norm(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x, return_feature_map=False, teacher_features=None, teacher_lr=None, extra_features=None):
        """
        Args:
            x: input tensor
            return_feature_map: whether to return raw feature map
            teacher_features: (B, num_features) ground truth feature labels.
                              When provided, uses GT features for fusion (teacher forcing)
                              instead of predicted features.
            teacher_lr: (B,) integer LR grade labels (0-4).
                        When provided, uses GT LR for fusion (teacher forcing)
                        instead of predicted LR.
            extra_features: (B, clinical_dim) clinical variables (e.g. lab indicators).
                            Always real input, used in both training and inference.
        """
        x = self.forward_features(x)           # (B, 512, D', H', W')
        
        # Feature head: 在GAP之前，基于3D特征图预测征象
        feature_out = None
        feature_map = None
        if self.feature_head is not None:
            feature_map = self.feature_head(x)                  # (B, num_features, D', H', W')
            feature_out = torch.sigmoid(feature_map)            # sigmoid激活
            feature_out = feature_out.mean(dim=(2, 3, 4))       # 空间平均 → (B, num_features)
        
        x = x.flatten(2).mean(-1)                              # GAP → (B, 512)
        
        # LR等级预测头：从GAP特征预测5类LR等级
        lr_pred = None
        if self.feature_fusion == 'hierarchical' and hasattr(self, 'lr_head'):
            lr_pred = self.lr_head(x)  # (B, 5) logits
        
        # 层级结构（hierarchical）：征象FC + LR预测 + 中间FC → 分类
        # 支持 Teacher Forcing：训练时用真实征象+真实LR，推理时用预测值
        if self.feature_fusion == 'hierarchical' and hasattr(self, 'feature_fc_head'):
            feature_out = torch.sigmoid(self.feature_fc_head(x))  # (B, num_features)
            # 征象部分：真实 or 预测
            fusion_features = teacher_features if teacher_features is not None else feature_out
            # LR等级部分：真实 one-hot or 预测 one-hot
            if teacher_lr is not None:
                lr_onehot = torch.zeros(x.size(0), 5, device=x.device, dtype=x.dtype)
                lr_onehot.scatter_(1, teacher_lr.unsqueeze(1).long(), 1.0)
            else:
                lr_onehot = torch.softmax(lr_pred, dim=-1)
            # 临床变量部分：始终使用真实输入（训练和推理均如此），并应用可学习的缩放系数
            if self.clinical_dim > 0 and extra_features is not None:
                scaled_clinical = extra_features * self.clinical_scale if self.clinical_scale is not None else extra_features
                combined = torch.cat([x, fusion_features, lr_onehot, scaled_clinical], dim=-1)  # (B, 512+num_features+5+clinical_dim)
            else:
                combined = torch.cat([x, fusion_features, lr_onehot], dim=-1)  # (B, 512+num_features+5)
            x = self.intermediate_fc(combined)  # (B, 512)
        
        # 简化层级结构（hierarchical_simple）：征象FC + 中间FC → 分类（无LR头，兼容旧checkpoint）
        # late_fusion模式：主分支与hierarchical_simple相同，但不拼接临床特征
        if self.feature_fusion in ('hierarchical_simple', 'late_fusion') and hasattr(self, 'feature_fc_head'):
            feature_out = torch.sigmoid(self.feature_fc_head(x))  # (B, num_features)
            # 征象部分：真实 or 预测
            fusion_features = teacher_features if teacher_features is not None else feature_out
            if self.late_fusion:
                # late_fusion：主分支仅用图像特征+征象概率，不拼接临床变量
                combined = torch.cat([x, fusion_features], dim=-1)  # (B, 512+num_features)
            else:
                # hierarchical_simple：拼接临床变量（应用可学习的缩放系数）
                if self.clinical_dim > 0 and extra_features is not None:
                    scaled_clinical = extra_features * self.clinical_scale if self.clinical_scale is not None else extra_features
                    combined = torch.cat([x, fusion_features, scaled_clinical], dim=-1)  # (B, 512+num_features+clinical_dim)
                else:
                    combined = torch.cat([x, fusion_features], dim=-1)  # (B, 512+num_features)
            x = self.intermediate_fc(combined)  # (B, 512)
            # === Late Fusion：临床注意力调节 + 残差连接 ===
            # 临床特征生成注意力权重，逐通道调节主分支融合特征
            # 残差连接保证即使 gate=0 也不影响主分支训练
            if self.late_fusion and hasattr(self, 'clinical_attention') and extra_features is not None:
                attn_input = torch.cat([x, extra_features], dim=-1)  # (B, 512+clinical_dim)
                attn_weights = self.clinical_attention(attn_input)    # (B, 512) sigmoid注意力
                x = x + self.late_fusion_gate * attn_weights * x     # 残差连接
        
        has_feature_output = self.feature_head is not None or (self.feature_fusion in ('hierarchical', 'hierarchical_simple', 'late_fusion') and hasattr(self, 'feature_fc_head'))
        has_lr_output = self.feature_fusion == 'hierarchical' and hasattr(self, 'lr_head')
        
        x = self.head_drop(x)
        if self.aux_head:
            out1 = self.head(x)
            out2 = self.auxiliary_head(x)
            if has_feature_output and has_lr_output:
                return out1, feature_out, lr_pred
            if has_feature_output:
                return out1, out2, feature_out
            return out1, out2
        else:
            out1 = self.head(x)  # (B, num_classes)
            if has_feature_output and has_lr_output:
                if return_feature_map:
                    return out1, feature_out, lr_pred, feature_map
                return out1, feature_out, lr_pred
            if has_feature_output:
                if return_feature_map:
                    return out1, feature_out, feature_map
                return out1, feature_out
            return out1

def uniformer_small(pretrained=True, video_mode=False, **kwargs):
    model = UniFormer(
        depth=[3, 4, 8, 3],
        embed_dim=[64, 128, 320, 512], head_dim=64, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), video_mode=video_mode, **kwargs)
    model.default_cfg = _cfg()
    return model

def uniformer_small_mtask(pretrained=True, video_mode=False, **kwargs):
    model = UniFormer(
        depth=[3, 4, 8, 3],
        embed_dim=[64, 128, 320, 512], head_dim=64, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), aux_head=True, video_mode=video_mode, **kwargs)
    model.default_cfg = _cfg()
    return model

# def uniformer_small_plus(pretrained=True, **kwargs):
#     model = UniFormer(
#         depth=[3, 5, 9, 3], conv_stem=True,
#         embed_dim=[64, 128, 320, 512], head_dim=32, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     return model

# def uniformer_small_plus_dim64(pretrained=True, **kwargs):
#     model = UniFormer(
#         depth=[3, 5, 9, 3], conv_stem=True,
#         embed_dim=[64, 128, 320, 512], head_dim=64, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     return model

def uniformer_base(pretrained=True, **kwargs):
    model = UniFormer(
        depth=[5, 8, 20, 7],
        embed_dim=[64, 128, 320, 512], head_dim=64, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

# def uniformer_base_ls(pretrained=True, **kwargs):
#     global layer_scale
#     layer_scale = True
#     model = UniFormer(
#         depth=[5, 8, 20, 7],
#         embed_dim=[64, 128, 320, 512], head_dim=64, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     return model
    
@register_model
def uniformer_small_IL(num_classes=2, 
                       num_phase=4,
                       pretrained=None, 
                       pretrained_cfg=None,
                       **kwards):
    '''
    Concat multi-phase images with image-level
    '''
    model = uniformer_small(in_chans=num_phase, num_classes=num_classes, video_mode=True, **kwards)
    return model

@register_model
def uniformer_small_IL_features(num_classes=3, 
                                 num_phase=4,
                                 num_feature_classes=24,
                                 feature_fusion=None,
                                 pretrained=None, 
                                 pretrained_cfg=None,
                                 **kwards):
    '''
    Concat multi-phase images with image-level, with feature prediction head
    Supports feature_fusion: None (default), 'hierarchical', or 'hierarchical_simple'
    Final-study default num_feature_classes=24 (24 predefined binary imaging features)
    '''
    # timm >= 1.0 injects this factory-only metadata argument into registered
    # model entrypoints.  It is not a UniFormer architecture parameter.
    kwards.pop('pretrained_cfg_overlay', None)
    kwards.pop('cache_dir', None)
    model = uniformer_small(
        in_chans=num_phase, 
        num_classes=num_classes, 
        num_feature_classes=num_feature_classes,
        feature_fusion=feature_fusion,
        video_mode=True,
        **kwards)
    return model

@register_model
def uniformer_small_IL_mtask(num_classes=2, 
                       num_phase=4,
                       pretrained=None, 
                       pretrained_cfg=None,
                       **kwards):
    '''
    Concat multi-phase images with image-level
    '''
    model = uniformer_small_mtask(in_chans=num_phase, num_classes=num_classes, video_mode=True, **kwards)
    return model

@register_model
def uniformer_base_IL(num_classes=2, 
                      num_phase=4,
                      pretrained=None, 
                      pretrained_cfg=None,
                      **kwards):
    '''
    Concat multi-phase images with image-level
    '''
    model = uniformer_base(in_chans=num_phase, num_classes=num_classes, video_mode=True, **kwards)
    return model
