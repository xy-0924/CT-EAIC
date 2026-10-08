#!/usr/bin/env python3
"""
将 kinetics-400 预训练权重的 3 通道 patch_embed1 扩展为 4 通道，
以适配 the four-phase CT classification input。
"""
import torch
import os

def expand_patch_embed_to_4ch(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"错误: 找不到输入文件 {input_path}")
        return

    print(f"正在加载权重: {input_path}")
    ckpt = torch.load(input_path, map_location='cpu')
    
    # 兼容不同的保存格式
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        wrap_key = 'state_dict'
    elif 'model' in ckpt:
        state_dict = ckpt['model']
        wrap_key = 'model'
    else:
        state_dict = ckpt
        wrap_key = None

    print(f"原始权重包含 {len(state_dict)} 个参数键值对。")

    # 处理 patch_embed1：将 3 通道平均扩展为 4 通道
    key = 'patch_embed1.proj.weight'
    if key in state_dict:
        w = state_dict[key]  # shape: [out_ch, in_ch, D, H, W]
        print(f"  发现 {key}: shape = {tuple(w.shape)}")
        if w.shape[1] == 3:
            # 第4个通道 = 前3个通道的均值
            w4 = torch.cat([w, w.mean(dim=1, keepdim=True)], dim=1)
            state_dict[key] = w4
            print(f"  已扩展为 4 通道: shape = {tuple(w4.shape)}")
        elif w.shape[1] == 4:
            print(f"  已经是 4 通道，无需处理。")
        else:
            print(f"  警告: 输入通道数为 {w.shape[1]}，不是预期的 3 或 4。")
    else:
        print(f"  警告: 未找到 {key}")

    # 同样处理 patch_embed2/3/4（如果有 in_chans 不匹配的情况，但通常不需要）
    for pe_key in ['patch_embed2.proj.weight', 'patch_embed3.proj.weight', 'patch_embed4.proj.weight']:
        if pe_key in state_dict:
            # 这些层的输入通道通常与上一层的输出通道匹配，不需要扩展
            pass

    # 移除分类头（确保与 partial 一致）
    keys_to_remove = [k for k in state_dict.keys() 
                      if 'head.' in k or 'auxiliary_head.' in k or 'pre_logits.' in k]
    print(f"\n检测到 {len(keys_to_remove)} 个分类头/辅助头，准备移除...")
    for k in keys_to_remove:
        del state_dict[k]
        print(f"  - 已移除: {k}")

    # 保存
    if wrap_key == 'state_dict':
        ckpt['state_dict'] = state_dict
    elif wrap_key == 'model':
        ckpt['model'] = state_dict
    else:
        ckpt = state_dict

    torch.save(ckpt, output_path)
    print(f"\n处理完成！新权重已保存至: {output_path}")
    print(f"新权重包含 {len(state_dict)} 个参数键值对。")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Expand k400 pretrained weights from 3ch to 4ch')
    parser.add_argument('--input', type=str, 
                        default='classification_models/pretrained_weights/uniformer_small_k400_8x8.pth',
                        help='Path to original 3-channel pretrained weights')
    parser.add_argument('--output', type=str,
                        default='classification_models/pretrained_weights/uniformer_small_k400_8x8_4ch.pth',
                        help='Path to save 4-channel weights')
    args = parser.parse_args()
    expand_patch_embed_to_4ch(args.input, args.output)
