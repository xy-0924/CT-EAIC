#!/usr/bin/env python3
"""
脚本功能：处理预训练权重文件，移除与当前任务不匹配的层（如分类头），
生成适用于迁移学习的 partial 权重文件。
"""
import torch
import argparse
import os

def process_pretrained_weights(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"错误: 找不到输入文件 {input_path}")
        return

    print(f"正在加载权重: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu')
    
    # 兼容不同的保存格式 (有时是 dict, 有时直接是 state_dict)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    print(f"原始权重包含 {len(state_dict)} 个参数键值对。")

    # 定义需要移除的键名模式 (通常是分类头或辅助头)
    keys_to_remove = []
    for key in state_dict.keys():
        if 'head.' in key or 'auxiliary_head.' in key or 'pre_logits.' in key:
            keys_to_remove.append(key)

    print(f"检测到 {len(keys_to_remove)} 个不匹配的层，准备移除...")
    for key in keys_to_remove:
        del state_dict[key]
        print(f"  - 已移除: {key}")

    # 保存新的权重文件
    # 保持与原文件相同的格式包装
    if 'state_dict' in checkpoint:
        checkpoint['state_dict'] = state_dict
    elif 'model' in checkpoint:
        checkpoint['model'] = state_dict
    else:
        checkpoint = state_dict

    torch.save(checkpoint, output_path)
    print(f"\n处理完成！新权重已保存至: {output_path}")
    print(f"新权重包含 {len(state_dict)} 个参数键值对。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process pretrained weights for classification models')
    parser.add_argument('--input', type=str, default='classification_models/pretrained_weights/uniformer_small_k400_8x8.pth',
                        help='Path to the original pretrained weights')
    parser.add_argument('--output', type=str, default='classification_models/pretrained_weights/uniformer_small_k400_8x8_partial.pth',
                        help='Path to save the processed partial weights')
    
    args = parser.parse_args()
    process_pretrained_weights(args.input, args.output)
