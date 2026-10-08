#!/usr/bin/env python3
import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.models import (create_model, safe_model_name, 
                         resume_checkpoint, load_checkpoint, 
                         model_parameters)

from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, BinaryCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError: 
    has_wandb = False

from metrics import *
from datasets.mp_liver_dataset import MultiPhaseLiverDataset, create_loader
import models
from models.create_model import create_model_custom
from torch.autograd import Variable
from loss_func.ordinal_loss import OrdinalCrossEntropyLoss, LabelDifferenceLoss, FocalOrdinalLoss
from loss_func.focal_loss import FocalLoss

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')
parser = argparse.ArgumentParser(description='LLD-MMRI 2023 Training')
# Dataset parameters
parser.add_argument('--data_dir', default='classification_models/data/images', type=str)
parser.add_argument('--train_anno_file', default='', type=str)
parser.add_argument('--val_anno_file', default='', type=str)
parser.add_argument('--fold', default=1, type=int, help='Cross-validation fold number (1-5), used to auto-generate train/val annotation files if not explicitly specified.')
parser.add_argument('--train_transform_list', default=['random_crop',
                                                       'z_flip', 
                                                       'x_flip', 
                                                       'y_flip', 
                                                       'rotation',
                                                       'random_intensity'], 
                                                       nargs='+', type=str)
parser.add_argument('--val_transform_list', default=['center_crop'], nargs='+', type=str)
parser.add_argument('--img_size', default=(20, 96, 96), type=int, nargs='+', help='input image size.')
parser.add_argument('--crop_size', default=(10, 80, 80), type=int, nargs='+', help='cropped image size.')
parser.add_argument('--flip_prob', default=0.5, type=float, help='Random flip prob (default: 0.5)')
parser.add_argument('--reprob', type=float, default=0.25, help='Random erase prob (default: 0.25)')
parser.add_argument('--rcprob', type=float, default=0.25, help='Random contrast prob (default: 0.25)')
parser.add_argument('--angle', default=45, type=int)

# Model parameters
parser.add_argument('--model', default='uniformer_small_IL', type=str, metavar='MODEL',
                    help='Name of model to train (default: "uniformer_small_IL")')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 2)')
parser.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                    help='validation batch size override (default: None)')

# Optimizer parameters
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.05,
                    help='weight decay (default: 0.05)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                    help='amount to decay each learning rate cycle (default: 0.5)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
parser.add_argument('--lr-k-decay', type=float, default=1.0,
                    help='learning rate k-decay for cosine/poly (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 1e-6)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Regularization parameters
parser.add_argument('--bce-loss', action='store_true', default=False,
                    help='Enable BCE loss w/ Mixup/CutMix use.')
parser.add_argument('--bce-target-thresh', type=float, default=None,
                    help='Threshold for binarizing softened BCE targets (default: None, disabled)')
parser.add_argument('--smoothing', type=float, default=0,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
parser.add_argument('--head-drop-rate', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate for fusion layers and classification head (default: 0.0)')
parser.add_argument('--fusion-hidden-dim', type=int, default=512,
                    help='Hidden dimension for intermediate_fc and classification head in fusion modes '
                         '(default: 512, matching original design; set to 128-256 to reduce overfitting)')
parser.add_argument('--tf-decay-start', type=int, default=-1,
                    help='Epoch to start Teacher Forcing decay (default: -1, disabled). '
                         'E.g. 50 means start reducing TF probability from epoch 50.')
parser.add_argument('--tf-decay-end', type=int, default=-1,
                    help='Epoch when Teacher Forcing fully decays to 0 (default: -1, disabled). '
                         'E.g. 90 means TF probability reaches 0 at epoch 90. '
                         'Between start and end, TF probability linearly decays from 1.0 to 0.0.')
parser.add_argument('--tf-noise', type=float, default=0.0,
                    help='TF noise blend ratio (default: 0.0, disabled). '
                         'When >0, blends TF features with model predictions: '
                         'features = (1-r)*GT + r*pred. '
                         'This trains the fusion layer to handle imperfect features, '
                         'directly addressing train-eval distribution shift. '
                         'Recommended: 0.2-0.3 for hierarchical/hierarchical_simple modes.')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='reduce',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--worker-seeding', type=str, default='all',
                    help='worker seed mode (default: all)')
parser.add_argument('--log-interval', type=int, default=25, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist', type=int, default=1, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('-j', '--workers', type=int, default=8, metavar='N',
                    help='how many training processes to use (default: 8)')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--no-ddp-bb', action='store_true', default=False,
                    help='Force broadcast buffers for native DDP to off.')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval-metric', default='f1', type=str, metavar='EVAL_METRIC',
                    help='Main metric (default: "f1"')
parser.add_argument('--report-metrics', default=['acc', 'f1', 'recall', 'precision', 'kappa'], 
                    nargs='+', choices=['acc', 'f1', 'recall', 'precision', 'kappa'], 
                    type=str, help='All evaluation metrics')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')

# Data mixup configs
parser.add_argument('--is-mixup', action='store_true', default=False,
                    help='unable mixup augmentation')
parser.add_argument('--alpha', type=float, default=0.0,
                    help='param of beta distribution for data mixup')  

# Channel cutout configs
parser.add_argument("--cutcnum", default=1, type=int, help='numbers of random cut channels.')
parser.add_argument('--cutcprob', type=float, default=0.25, help='Random channel-cut prob (default: 0.25)')
parser.add_argument('--cutcmode', default='zeros', type=str)
parser.add_argument('--case-mapping-file', default='', type=str,
                    help='Path to case name mapping file (original_id -> case_xxx)')
parser.add_argument('--label-mode', default='lr', type=str, choices=['original', 'lr'],
                    help='Which label column to use: "original" (2nd col, 0/1/2) or "lr" (4th col, LR grade). Default: "lr"')
parser.add_argument('--class-weights', default=None, type=float, nargs='+',
                    help='Class weights for imbalanced dataset (e.g., --class-weights 1.0 5.0 10.0)')
parser.add_argument('--focal-loss', action='store_true', default=False,
                    help='Use Focal Loss to focus on hard/misclassified examples (recommended for imbalanced data)')
parser.add_argument('--focal-gamma', type=float, default=2.0,
                    help='Focal Loss focusing parameter (default: 2.0, gamma=0 is standard CE, higher=more focus on hard examples)')
parser.add_argument('--use-balanced-sampler', action='store_true', default=False,
                    help='Use WeightedRandomSampler to balance classes by oversampling minority and undersampling majority')
parser.add_argument('--balanced-sampler-scale', default='sqrt', type=str, choices=['linear', 'sqrt', 'log'],
                    help='Scale strategy for balanced sampler: "linear" (full balance), "sqrt" (mild, default), "log" (mildest)')
parser.add_argument('--max-samples-per-class', default=550, type=int,
                    help='Maximum samples per class for oversampling/undersampling (default: 550, set 0 to disable)')
parser.add_argument('--skip-cases-file', default='classification_models/data/skip_cases.txt', type=str,
                    help='Path to skip cases file (cases with missing/mismatched feature labels)')
parser.add_argument('--data1-file', default='classification_models/data/data1.xlsx', type=str,
                    help='Path to data1.xlsx containing 24 feature labels')
parser.add_argument('--feature-loss-weight', default=0.05, type=float,
                    help='Weight for feature prediction loss (default: 0.05, 10-feature BCE scale ~2-5, actual contribution ~0.1-0.25)')
parser.add_argument('--num-feature-classes', default=24, type=int,
                    help='Number of feature classes for feature head (default: 10, 9 binary + 1 tumor size)')
parser.add_argument('--feature-fusion', default=None, type=str, choices=[None, 'hierarchical', 'hierarchical_simple', 'late_fusion'],
                    help='Feature fusion mode for models with feature head: None (default), hierarchical, hierarchical_simple, or late_fusion')
parser.add_argument('--selected-features', default=None, type=int, nargs='+',
                    help='Optional ablation only: subset of feature indices (0-23). The final study models use all 24 features.')
parser.add_argument('--freeze-backbone', action='store_true', default=False,
                    help='Freeze backbone parameters, only train head/fusion layers (Stage 2)')
parser.add_argument('--lr-loss-weight', default=0.3, type=float,
                    help='Weight for LR grade prediction loss (default: 0.3, actual contribution ~0.2-0.36)')
parser.add_argument('--include-clinical', action='store_true', default=False,
                    help='Include clinical variables (lab indicators) as extra fusion input')
parser.add_argument('--clinical-dim', default=10, type=int,
                    help='Dimension of clinical variables (default: 10, lab indicators only, tumor size is in features)')
parser.add_argument('--clinical-scale', default=1.0, type=float,
                    help='Initial scale factor for clinical features in fusion (default: 1.0, no scaling). '
                         'Set >1.0 to boost clinical feature weight. The scale is a learnable parameter '
                         'that will be fine-tuned during training.')
parser.add_argument('--normalize-clinical', action='store_true', default=False,
                    help='Apply z-score normalization to continuous clinical variables (age, AFP). '
                         'Binary variables (sex, PLT, ALB, etc.) are kept unchanged. '
                         'Training set computes and saves stats; val/test loads them automatically.')
parser.add_argument('--clinical-stats-file', default='', type=str,
                    help='Path to save/load clinical normalization stats JSON '
                         '(default: auto, saved in output dir during training)')

# === Ordinal Loss (方案一：排序感知损失) ===
parser.add_argument('--ordinal-alpha', default=0.0, type=float,
                    help='Ordinal soft-label smoothing strength (方案一). '
                         '0.0 = disabled (use standard CE), recommended: 0.1~0.3. '
                         'Distributes probability to adjacent LR classes.')
parser.add_argument('--ordinal-type', default='soft', type=str,
                    choices=['soft', 'diff', 'focal'],
                    help='Ordinal loss variant (方案一): '
                         '"soft" = OrdinalCE with soft-label (default), '
                         '"diff" = CE + ordinal distance penalty, '
                         '"focal" = Focal + ordinal soft-label.')
parser.add_argument('--ordinal-gamma', default=1.5, type=float,
                    help='Focal exponent for ordinal-type=focal (default: 1.5)')

# === Minority Oversampling (方案二：少数类重复采样) ===
parser.add_argument('--minority-repeat', default=1, type=int,
                    help='Repeat factor for minority classes (方案二). '
                         '1 = no oversampling (default). '
                         '2 = double sample LR-3/LR-4 (or specified --minority-classes). '
                         'Works independently of --use-balanced-sampler.')
parser.add_argument('--minority-classes', default=None, type=int, nargs='+',
                    help='Class indices to oversample (方案二, default: auto-detect LR-3/LR-4 = [1, 2] in lr mode)')

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Auto-generate annotation file paths based on fold if not explicitly specified
    if not args.train_anno_file:
        args.train_anno_file = f'classification_models/data/labels/train_fold{args.fold}.txt'
    if not args.val_anno_file:
        args.val_anno_file = f'classification_models/data/labels/val_fold{args.fold}.txt'

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    setup_default_logging()
    args, args_text = _parse_args()
    
    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else: 
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")
             
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    random_seed(args.seed, args.rank)

    # Auto infer num_classes based on label_mode if not explicitly set
    if args.num_classes is None:
        if getattr(args, 'label_mode', 'original') == 'lr':
            args.num_classes = 5
        else:
            args.num_classes = 3
        if args.local_rank == 0:
            _logger.info(f'Auto inferred num_classes={args.num_classes} based on label_mode={args.label_mode}')

    # 构建模型额外参数
    model_kwargs = dict(
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint)
    
    # 如果模型名包含 _features，添加 feature head 相关参数
    if 'features' in args.model:
        selected = getattr(args, 'selected_features', None)
        if selected is not None:
            num_feat = len(selected) + 1  # +1 for tumor size appended to feature vector
            model_kwargs['num_feature_classes'] = num_feat
            if args.local_rank == 0:
                _logger.info(f'Using {num_feat} selected features: {selected}')
        else:
            model_kwargs['num_feature_classes'] = getattr(args, 'num_feature_classes', 24)
        model_kwargs['feature_fusion'] = getattr(args, 'feature_fusion', None)
        # 临床变量维度：hierarchical融合时可引入额外临床特征
        if getattr(args, 'include_clinical', False):
            model_kwargs['clinical_dim'] = getattr(args, 'clinical_dim', 10)
            # 临床特征缩放系数：控制临床特征在融合中的权重
            clinical_scale = getattr(args, 'clinical_scale', 1.0)
            model_kwargs['clinical_scale_init'] = clinical_scale
            if args.local_rank == 0:
                _logger.info(f'Clinical variables enabled: dim={model_kwargs["clinical_dim"]}, '
                             f'scale_init={clinical_scale} (learnable)')
        # 融合层/分类头专用正则化
        head_drop = getattr(args, 'head_drop_rate', 0.0)
        if head_drop > 0:
            model_kwargs['head_drop_rate'] = head_drop
            if args.local_rank == 0:
                _logger.info(f'Head/fusion dropout rate: {head_drop}')
        # 融合层隐藏维度：控制fusion head参数量，防止过拟合
        fusion_hidden = getattr(args, 'fusion_hidden_dim', 512)
        if fusion_hidden != 512:
            model_kwargs['fusion_hidden_dim'] = fusion_hidden
            if args.local_rank == 0:
                _logger.info(f'Fusion hidden dim: {fusion_hidden} (reduced from 512)')
    
    # model = create_model(
    model = create_model_custom(
        args.model,
        **model_kwargs)
    # print(model)
    if args.local_rank == 0:
        _logger.info(f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    # move model to GPU, enable channels last layout if set
    model.cuda()

    # Stage 2: 冻结 backbone，只训练 head/fusion 层
    if getattr(args, 'freeze_backbone', False):
        trainable_names = model.freeze_backbone()
        total_params = sum(m.numel() for m in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if args.local_rank == 0:
            _logger.info(f'Backbone frozen. Trainable params: {trainable_params}/{total_params} ({100*trainable_params/total_params:.1f}%)')
            _logger.info(f'Trainable layers: {trainable_names}')

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        if has_apex and use_amp == 'apex':
            # Apex SyncBN preferred unless native amp is activated
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    # 构建优化器：如果冻结了 backbone，只传入可训练参数
    if getattr(args, 'freeze_backbone', False):
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = create_optimizer_v2(trainable_params, **optimizer_kwargs(cfg=args))
    else:
        optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank], broadcast_buffers=not args.no_ddp_bb)
        # NOTE: EMA model does not need to be wrapped by DDP

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    # create the train and eval datasets/dataloader
    dataset_train = MultiPhaseLiverDataset(args, is_training=True)

    dataset_eval = MultiPhaseLiverDataset(args, is_training=False)

    loader_train = create_loader(dataset_train, 
                                 batch_size=args.batch_size,
                                 is_training=True,
                                 num_workers=args.workers,
                                 distributed=args.distributed,
                                 pin_memory=args.pin_mem,
                                 use_balanced_sampler=getattr(args, 'use_balanced_sampler', False),
                                 balanced_sampler_scale=getattr(args, 'balanced_sampler_scale', 'sqrt'))
    
    loader_eval = create_loader(dataset_eval, 
                                batch_size=args.batch_size,
                                is_training=False,
                                num_workers=args.workers,
                                distributed=args.distributed,
                                pin_memory=args.pin_mem)

    # Compute class weights for imbalanced dataset
    if args.class_weights:
        class_weights = torch.tensor(args.class_weights, dtype=torch.float32)
        _logger.info(f'Using class weights: {class_weights.tolist()}')
    else:
        class_weights = None

    # === 方案一: Ordinal Loss (排序感知损失) ===
    ordinal_alpha = getattr(args, 'ordinal_alpha', 0.0)
    ordinal_type = getattr(args, 'ordinal_type', 'soft')
    cw_list = class_weights.tolist() if class_weights is not None else None

    if ordinal_alpha > 0:
        if ordinal_type == 'soft':
            train_loss_fn = OrdinalCrossEntropyLoss(
                num_classes=args.num_classes, alpha=ordinal_alpha, class_weights=cw_list)
            _logger.info(f'[方案一] Using OrdinalCE Loss (alpha={ordinal_alpha}, class_weights={cw_list})')
        elif ordinal_type == 'diff':
            train_loss_fn = LabelDifferenceLoss(
                num_classes=args.num_classes, lam=ordinal_alpha, class_weights=cw_list)
            _logger.info(f'[方案一] Using LabelDifference Loss (lam={ordinal_alpha}, class_weights={cw_list})')
        elif ordinal_type == 'focal':
            train_loss_fn = FocalOrdinalLoss(
                num_classes=args.num_classes, gamma=getattr(args, 'ordinal_gamma', 1.5),
                alpha_smooth=ordinal_alpha, class_weights=cw_list)
            _logger.info(f'[方案一] Using FocalOrdinal Loss (gamma={getattr(args, "ordinal_gamma", 1.5)}, '
                         f'alpha={ordinal_alpha}, class_weights={cw_list})')
        else:
            raise ValueError(f'Unknown ordinal-type: {ordinal_type}')
    elif getattr(args, 'focal_loss', False):
        focal_gamma = getattr(args, 'focal_gamma', 2.0)
        train_loss_fn = FocalLoss(gamma=focal_gamma, class_weights=cw_list)
        _logger.info(f'Using Focal Loss (gamma={focal_gamma}, class_weights={cw_list})')
    elif args.smoothing or getattr(args, 'feature_fusion', None) in ('hierarchical', 'hierarchical_simple', 'late_fusion'):
        smoothing_val = args.smoothing if args.smoothing > 0 else 0.1
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=smoothing_val, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=smoothing_val)
        if args.local_rank == 0 and not args.smoothing:
            _logger.info(f'Auto-enabled label smoothing={smoothing_val} for {args.feature_fusion} fusion mode')
    else:
        train_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # Feature prediction loss (9征象多标签二分类)
    feature_loss_fn = nn.BCELoss().cuda()  # 注意：模型forward已输出sigmoid，故用BCELoss而非BCEWithLogitsLoss
    feature_loss_weight = getattr(args, 'feature_loss_weight', 1.0)

    # LR grade prediction loss (5类分类)
    lr_loss_fn = nn.CrossEntropyLoss().cuda()
    lr_loss_weight = getattr(args, 'lr_loss_weight', 0.1)

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    metric_savers = None
    output_dir = None

    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = safe_model_name(args.model)
        output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
        
        # 清空输出目录，避免新旧文件混在一起
        if os.path.isdir(output_dir):
            import shutil
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
        
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(
                model=model, optimizer=optimizer, 
                args=args, model_ema=model_ema, 
                amp_scaler=loss_scaler,
                checkpoint_dir=output_dir, recovery_dir=output_dir, 
                decreasing=decreasing, max_history=args.checkpoint_hist)
        
        best_metrics = {}
        metric_savers = {}
        for metric in args.report_metrics:
            best_metrics[metric] = {'value': None, 'epoch': None}
            metric_savers[metric] = CheckpointSaver(
                                    model=model, optimizer=optimizer, 
                                    args=args, model_ema=model_ema, 
                                    amp_scaler=loss_scaler,
                                    checkpoint_prefix=f'best_{metric}_checkpoint',
                                    checkpoint_dir=output_dir,
                                    recovery_dir=output_dir, 
                                    decreasing=decreasing, 
                                    max_history=args.checkpoint_hist)
        
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            # 日志：Teacher Forcing 概率状态
            tf_start = getattr(args, 'tf_decay_start', -1)
            tf_end = getattr(args, 'tf_decay_end', -1)
            if tf_start >= 0 and tf_end > tf_start and args.local_rank == 0:
                if epoch < tf_start:
                    tf_prob = 1.0
                elif epoch >= tf_end:
                    tf_prob = 0.0
                else:
                    tf_prob = 1.0 - (epoch - tf_start) / (tf_end - tf_start)
                _logger.info(f'Epoch {epoch}: Teacher Forcing probability = {tf_prob:.3f} '
                             f'(decay: {tf_start}→{tf_end})')

            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema,
                feature_loss_fn=feature_loss_fn, feature_loss_weight=feature_loss_weight,
                lr_loss_fn=lr_loss_fn, lr_loss_weight=lr_loss_weight)

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                ema_eval_metrics = validate(
                    model_ema.module, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if output_dir is not None:
                update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None, log_wandb=args.log_wandb and has_wandb)

            if metric_savers is not None:
                # Save the best checkpoint for this metric
                for metric in args.report_metrics:
                    if best_metrics[metric]['value'] is None or (eval_metrics[metric] > best_metrics[metric]['value']):
                        best_metrics[metric]['value'] = eval_metrics[metric]
                        best_metrics[metric]['epoch'] = epoch
                        ckpt_saver = metric_savers[metric]
                        ckpt_saver.save_checkpoint(epoch, metric=best_metrics[metric]['value'])

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

# mixup data in one batch
def mixup_data(x, y, alpha=0.5, use_cuda=True):
    if alpha > 0:
        lamb = np.random.beta(alpha, alpha)
    else:
        lamb = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lamb * x + (1 - lamb) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lamb


def mixup_criterion(criterion, pred, y_a, y_b, lamb):
    return lamb * criterion(pred, y_a) + (1 - lamb) * criterion(pred, y_b)


def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None,
        feature_loss_fn=None, feature_loss_weight=1.0,
        lr_loss_fn=None, lr_loss_weight=0.1):

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, batch in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        
        # 解包batch：可能包含征象标签、LR标签和临床特征
        lr_label = None
        clinical_features = None
        if len(batch) == 5:
            input, target, feature_targets, lr_label, clinical_features = batch
            if feature_targets is not None:
                feature_targets = feature_targets.cuda()
            if lr_label is not None:
                if isinstance(lr_label, torch.Tensor):
                    lr_label = lr_label.cuda()
                else:
                    lr_label = torch.tensor(lr_label, dtype=torch.long).cuda()
            if clinical_features is not None:
                clinical_features = clinical_features.cuda()
        elif len(batch) == 4:
            input, target, feature_targets, lr_label = batch
            if feature_targets is not None:
                feature_targets = feature_targets.cuda()
            if lr_label is not None:
                if isinstance(lr_label, torch.Tensor):
                    lr_label = lr_label.cuda()
                else:
                    # lr_label可能是整数列表，转换为tensor
                    lr_label = torch.tensor(lr_label, dtype=torch.long).cuda()
        elif len(batch) == 3:
            input, target, feature_targets = batch
            # 特征标签缺失时设为None，后续跳过feature loss
            if feature_targets is not None:
                feature_targets = feature_targets.cuda()
        else:
            input, target = batch
            feature_targets = None
        
        input, target = input.cuda(), target.cuda()

        # Teacher Forcing: 训练时用真实征象标签和真实LR等级替代模型预测值送入融合层
        # 推理时不传 teacher_features/teacher_lr，自动使用模型预测值
        # 支持渐进衰减：从 tf_decay_start 到 tf_decay_end，TF 概率从 1.0 线性下降到 0.0
        _teacher_features = None
        _teacher_lr = None
        if (getattr(args, 'feature_fusion', None) in ('hierarchical', 'hierarchical_simple', 'late_fusion')
            and feature_targets is not None):
            # 计算当前 epoch 的 TF 概率
            tf_start = getattr(args, 'tf_decay_start', -1)
            tf_end = getattr(args, 'tf_decay_end', -1)
            if tf_start >= 0 and tf_end > tf_start:
                if epoch < tf_start:
                    tf_prob = 1.0
                elif epoch >= tf_end:
                    tf_prob = 0.0
                else:
                    tf_prob = 1.0 - (epoch - tf_start) / (tf_end - tf_start)
                # 按概率决定是否使用 Teacher Forcing
                if torch.rand(1).item() < tf_prob:
                    _teacher_features = feature_targets
                    if lr_label is not None:
                        _teacher_lr = lr_label
                # else: 不使用 TF，让模型用自己的预测值
            else:
                # 未启用衰减，始终使用 Teacher Forcing
                _teacher_features = feature_targets
                if lr_label is not None:
                    _teacher_lr = lr_label

        if args.is_mixup:
            # Apply mixup
            input, targets_a, targets_b, lamb = mixup_data(input, target, args.alpha)
            input, targets_a, targets_b = map(Variable, (input, targets_a, targets_b))
            with amp_autocast():
                # TF noise blend (two-pass): first get model's feature predictions, then blend with GT
                tf_noise = getattr(args, 'tf_noise', 0.0)
                if tf_noise > 0 and _teacher_features is not None:
                    with torch.no_grad():
                        _gap = model.forward_features(input).flatten(2).mean(-1)
                        _feat_pred = torch.sigmoid(model.feature_fc_head(_gap))
                    _teacher_features = (1 - tf_noise) * _teacher_features + tf_noise * _feat_pred.detach()
                output = model(input, teacher_features=_teacher_features, teacher_lr=_teacher_lr, extra_features=clinical_features)
                if isinstance(output, (tuple, list)) and len(output) >= 2:
                    main_output = output[0]
                    feature_output = output[1] if len(output) >= 2 else None
                    lr_output = output[2] if len(output) >= 3 else None
                else:
                    main_output = output
                    feature_output = None
                    lr_output = None
                loss_main = mixup_criterion(loss_fn, main_output, targets_a, targets_b, lamb)
                # 只在特征标签存在时计算feature loss
                if feature_output is not None and feature_targets is not None and feature_loss_fn is not None:
                    loss_feat = feature_loss_fn(feature_output, feature_targets)
                    loss = loss_main + feature_loss_weight * loss_feat
                else:
                    loss = loss_main
                # LR等级预测loss
                if lr_output is not None and lr_label is not None and lr_loss_fn is not None:
                    loss_lr = lr_loss_fn(lr_output, lr_label)
                    loss = loss + lr_loss_weight * loss_lr
        else:
            with amp_autocast():
                # TF noise blend (two-pass): first get model's feature predictions, then blend with GT
                # This trains the fusion layer on a mix of GT and predicted features,
                # directly addressing the train-eval distribution shift.
                # Uses only backbone+feature_fc_head for efficiency (avoids full forward + double BN update).
                tf_noise = getattr(args, 'tf_noise', 0.0)
                if tf_noise > 0 and _teacher_features is not None:
                    with torch.no_grad():
                        _gap = model.forward_features(input).flatten(2).mean(-1)
                        _feat_pred = torch.sigmoid(model.feature_fc_head(_gap))
                    _teacher_features = (1 - tf_noise) * _teacher_features + tf_noise * _feat_pred.detach()
                output = model(input, teacher_features=_teacher_features, teacher_lr=_teacher_lr, extra_features=clinical_features)
                if isinstance(output, (tuple, list)) and len(output) >= 2:
                    main_output = output[0]
                    feature_output = output[1] if len(output) >= 2 else None
                    lr_output = output[2] if len(output) >= 3 else None
                else:
                    main_output = output
                    feature_output = None
                    lr_output = None
                loss_main = loss_fn(main_output, target)
                # 只在特征标签存在时计算feature loss
                if feature_output is not None and feature_targets is not None and feature_loss_fn is not None:
                    loss_feat = feature_loss_fn(feature_output, feature_targets)
                    loss = loss_main + feature_loss_weight * loss_feat
                else:
                    loss = loss_main
                # LR等级预测loss
                if lr_output is not None and lr_label is not None and lr_loss_fn is not None:
                    loss_lr = lr_loss_fn(lr_output, lr_label)
                    loss = loss + lr_loss_weight * loss_lr

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.local_rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('lr', lr), ('loss', losses_m.avg)])

@torch.no_grad()
def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    model.eval()
    predictions = []
    labels = []
    feature_predictions = []
    feature_targets_list = []
    lr_predictions = []
    lr_targets_list = []
    last_idx = len(loader) - 1
    for batch_idx, batch in enumerate(loader):
        last_batch = batch_idx == last_idx
        
        lr_label = None
        clinical_features = None
        if len(batch) == 5:
            input, target, feature_targets, lr_label, clinical_features = batch
            feature_targets = feature_targets.cuda()
            feature_targets_list.append(feature_targets)
            if isinstance(lr_label, torch.Tensor):
                lr_label = lr_label.cuda()
                lr_targets_list.append(lr_label)
            else:
                lr_targets_list.append(torch.tensor(lr_label, dtype=torch.long))
            if clinical_features is not None:
                clinical_features = clinical_features.cuda()
        elif len(batch) == 4:
            input, target, feature_targets, lr_label = batch
            feature_targets = feature_targets.cuda()
            feature_targets_list.append(feature_targets)
            if isinstance(lr_label, torch.Tensor):
                lr_label = lr_label.cuda()
                lr_targets_list.append(lr_label)
            else:
                lr_targets_list.append(torch.tensor(lr_label, dtype=torch.long))
        elif len(batch) == 3:
            input, target, feature_targets = batch
            feature_targets = feature_targets.cuda()
            feature_targets_list.append(feature_targets)
        else:
            input, target = batch
            feature_targets = None
        
        input = input.cuda()
        target = target.cuda()

        with amp_autocast():
            output = model(input, extra_features=clinical_features)

        if isinstance(output, (tuple, list)):
            predictions.append(output[0])
            if len(output) >= 2 and feature_targets is not None:
                feature_predictions.append(output[1])
            if len(output) >= 3 and output[2] is not None:
                lr_predictions.append(output[2])
        else:
            predictions.append(output)
        labels.append(target)

    evaluation_metrics = compute_metrics(predictions, labels, loss_fn, args)
    
    # 计算征象预测的平均准确率
    selected_features = []
    if feature_predictions and feature_targets_list:
        feat_preds = torch.cat(feature_predictions, dim=0).cpu().numpy()
        feat_targets = torch.cat(feature_targets_list, dim=0).cpu().numpy()
        
        # 所有征象均为二分类
        feat_binary_preds = (feat_preds > 0.5).astype(float)
        feat_acc = (feat_binary_preds == feat_targets).mean()
        evaluation_metrics['feature_acc'] = feat_acc
        
        # 计算每个征象的准确率，筛选 >80% 的
        feature_accs = []
        for i in range(feat_preds.shape[1]):
            acc_i = (feat_binary_preds[:, i] == feat_targets[:, i]).mean()
            evaluation_metrics[f'feature_{i}_acc'] = float(acc_i)
            feature_accs.append(float(acc_i))
            if acc_i >= 0.8:
                selected_features.append(i)
        
        evaluation_metrics['selected_feature_count'] = len(selected_features)
        evaluation_metrics['selected_feature_indices'] = selected_features
        
        # 保存征象准确率到文件
        if args.local_rank == 0 and args.output:
            feat_acc_path = os.path.join(args.output, f'feature_acc_epoch{getattr(args, "_current_epoch", 0)}.json')
            os.makedirs(args.output, exist_ok=True)
            feat_acc_data = {
                'feature_acc': float(evaluation_metrics['feature_acc']),
                'feature_accuracies': feature_accs,
                'selected_features': [int(i) for i in selected_features],
                'selected_feature_count': len(selected_features),
                'original_feature_indices': getattr(args, 'selected_features', None),
            }
            try:
                import json
                with open(feat_acc_path, 'w') as f:
                    json.dump(feat_acc_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                _logger.warning(f"Failed to save feature accuracy: {e}")

    # 计算LR等级预测准确率
    if lr_predictions and lr_targets_list:
        lr_preds = torch.cat(lr_predictions, dim=0).cpu()  # (N, 5) logits
        lr_true = torch.cat(lr_targets_list, dim=0).cpu()  # (N,) integer labels
        lr_pred_labels = lr_preds.argmax(dim=-1)  # (N,)
        lr_acc = (lr_pred_labels == lr_true).float().mean().item()
        evaluation_metrics['lr_acc'] = lr_acc
        # 每个LR等级的预测数量
        for i in range(lr_preds.shape[1]):
            cls_count = (lr_true == i).sum().item()
            cls_correct = ((lr_pred_labels == i) & (lr_true == i)).sum().item()
            cls_acc = cls_correct / cls_count if cls_count > 0 else 0.0
            evaluation_metrics[f'lr_class_{i}_acc'] = cls_acc
            evaluation_metrics[f'lr_class_{i}_count'] = cls_count

    if args.local_rank == 0:
        output_str = 'Test:\n'
        for key, value in evaluation_metrics.items():
            output_str += f'{key}: {value}\n'
        _logger.info(output_str)
        
        # 输出筛选出的高准确率征象
        if selected_features:
            _logger.info(f"Selected {len(selected_features)} features with accuracy >= 80%: {selected_features}")

    return evaluation_metrics

def compute_metrics(outputs, targets, loss_fn, args):
    
    outputs = torch.cat(outputs, dim=0).detach()
    targets = torch.cat(targets, dim=0).detach()

    if args.distributed:
        outputs = gather_data(outputs)
        targets = gather_data(targets)

    loss = loss_fn(outputs, targets).cpu().item()

    outputs = outputs.cpu().numpy()
    targets = targets.cpu().numpy()
    acc = ACC(outputs, targets)
    f1 = F1_score(outputs, targets)
    recall = Recall(outputs, targets)
    # specificity = Specificity(outputs, targets)
    precision = Precision(outputs, targets)
    kappa = Cohen_Kappa(outputs, targets)
    metrics = OrderedDict([
        ('loss', loss),
        ('acc', acc),
        ('f1', f1),
        ('recall', recall),
        ('precision', precision),
        ('kappa', kappa),
    ])
        
    return metrics


def gather_data(input):
    '''
    gather data from multi gpus
    '''
    output_list = [torch.zeros_like(input) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_list, input)
    output = torch.cat(output_list, dim=0)
    return output


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()