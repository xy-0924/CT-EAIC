#!/usr/bin/env python3
"""
处理外部测试集标签文件：按自然顺序排列并补充缺失的 label2。

功能：
    1. 读取 case_name_mapping_test.txt 和 data2.xlsx
    2. 使用 case_name_mapping_test.txt 建立原始病例名与 converted_case_name 的映射
    3. 从 data2.xlsx 的 "LI-RADS等级" 列中提取 label2 信息
    4. 对于有多个ROI的case（images目录中有_1和_2），若data2中对应记录LR等级相同，则拆分为 case_xxx_1 和 case_xxx_2
    5. 按自然顺序排序后保存到 test.txt

使用示例：
    python process_labels_test.py
"""

import os
import pandas as pd
import re

def natural_sort_key(s):
    """自然排序键函数：case_1, case_2, ..., case_10, case_11, ..."""
    parts = re.split(r'(\d+)', s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

def normalize_label2(label2):
    """标准化label2，处理大小写和空格问题"""
    if pd.isna(label2) or str(label2).strip() == '':
        return ''
    
    label2_str = str(label2).strip()
    label2_str = label2_str.upper()
    label2_str = re.sub(r'\s+', ' ', label2_str)
    label2_str = label2_str.replace('LR  1/2', 'LR 1/2').replace('LR  5', 'LR 5').replace('LR  4', 'LR 4').replace('LR  3', 'LR 3').replace('LR  M', 'LR M')
    
    return label2_str

# 24个征象列名（用于提取并添加到输出）
SIGN_COLUMNS = [
    '动脉期高强化（0=无，1=有）',
    '非边缘动脉期高强化（0=无，1=有）',
    '环形动脉期高强化（0=无，1=有）',
    '非边缘廓清（0=无，1=有）',
    '周边\u201c廓清\u201d（0=无，1=有）',
    '晕状强化（0=无，1=有）',
    '强化包膜（0=无，1=有）',
    '非强化包膜（0=无，1=有）',
    '周边不连续结节状强化（0=无，1=有）',
    '渐进性强化（0=无，1=有）',
    '向心性增强（0=无，1=有）',
    '平行血池强化（0=无，1=有）',
    '\xa0均匀动脉期强化（0=无，1=有）',
    '均匀门脉期强化（0=无，1=有）',
    '均匀延迟期强化（0=无，1=有）',
    '坏死或严重缺血（0=无，1=有）',
    '瘤内出血（0=无，1=有）',
    '结中结 结构（0=无，1=有）',
    '马赛克结构 / 镶嵌样结构（0=无，1=有）',
    '延迟期中央强化（0=无，1=有）',
    '浸润性外观（0=无，1=有）',
    '门脉期周围低强化（0=无，1=有）',
    '病灶内脂肪（含量多于肝脏）（0=无，1=有）',
    '实性病灶内脂肪缺失（0=无，1=有）',
    '瘤内动脉（0=无，1=有）',
]


def main():
    # 读取 case_name_mapping_test.txt
    print("读取case_name_mapping_test.txt...")
    mapping_file = './case_name_mapping_test.txt'
    mapping_df = pd.read_csv(mapping_file, sep='\t', dtype=str)
    print(f"映射文件包含 {len(mapping_df)} 条记录")
    
    # 读取 data2.xlsx
    print("\n读取data2.xlsx...")
    data1_file = './classification_models/data/data2.xlsx'
    # 使用 header=2 读取以获取完整的24个征象列名
    data1_df = pd.read_excel(data1_file, header=2, dtype=str)
    # 将 LI-RADS等级 列从 Unnamed: 15 重命名
    data1_df = data1_df.rename(columns={'Unnamed: 15': 'LI-RADS等级'})
    print(f"data2.xlsx包含 {len(data1_df)} 条记录")
    
    # 打印征象列信息
    print(f"\n征象列数量: {len(SIGN_COLUMNS)}")
    missing_sign_cols = [c for c in SIGN_COLUMNS if c not in data1_df.columns]
    if missing_sign_cols:
        print(f"警告：data2.xlsx 中缺少以下征象列: {missing_sign_cols}")
    else:
        print("所有征象列均已找到")
    
    # 建立 original_case_name -> label2 映射（同时存储小写和清理后的版本）
    id_to_label2 = {}
    id_to_label2_lower = {}  # 小写版本
    # 同时建立 -> 征象字典 的映射
    id_to_signs = {}
    id_to_signs_lower = {}
    for idx, row in data1_df.iterrows():
        patient_id = str(row.iloc[0]).strip()
        li_rads = str(row['LI-RADS等级']).strip() if pd.notna(row['LI-RADS等级']) else ''
        
        if patient_id and patient_id != 'nan':
            label2_val = normalize_label2(li_rads)
            id_to_label2[patient_id] = label2_val
            id_to_label2_lower[patient_id.lower()] = label2_val

            # 提取征象值
            sign_vals = {}
            for col in SIGN_COLUMNS:
                if col in data1_df.columns:
                    v = row.get(col, '')
                    if pd.isna(v):
                        sign_vals[col] = ''
                    else:
                        sign_vals[col] = str(int(float(v))) if str(v).strip() not in ('', 'nan') else ''
                else:
                    sign_vals[col] = ''
            id_to_signs[patient_id] = sign_vals
            id_to_signs_lower[patient_id.lower()] = sign_vals

    print(f"\nID到label2的映射数量: {len(id_to_label2)}")
    
    def find_label2(name):
        """尝试多种方式查找label2"""
        if not name or name == 'nan':
            return ''
        # 1. 精确匹配
        if name in id_to_label2:
            return id_to_label2[name]
        # 2. 小写匹配
        if name.lower() in id_to_label2_lower:
            return id_to_label2_lower[name.lower()]
        # 3. 清理括号后匹配（全角/半角括号）
        clean = re.sub(r'[（(].*?[）)]', '', name).strip()
        if clean in id_to_label2:
            return id_to_label2[clean]
        if clean.lower() in id_to_label2_lower:
            return id_to_label2_lower[clean.lower()]
        # 4. 去掉"-S数字"或"-大小"等后缀
        clean2 = re.sub(r'\-.*', '', name).strip()
        if clean2 in id_to_label2:
            return id_to_label2[clean2]
        if clean2.lower() in id_to_label2_lower:
            return id_to_label2_lower[clean2.lower()]
        return ''
    
    def find_signs(name):
        """尝试多种方式查找征象字典"""
        if not name or name == 'nan':
            return {}
        # 1. 精确匹配
        if name in id_to_signs:
            return id_to_signs[name]
        # 2. 小写匹配
        if name.lower() in id_to_signs_lower:
            return id_to_signs_lower[name.lower()]
        # 3. 清理括号后匹配（全角/半角括号）
        clean = re.sub(r'[（(].*?[）)]', '', name).strip()
        if clean in id_to_signs:
            return id_to_signs[clean]
        if clean.lower() in id_to_signs_lower:
            return id_to_signs_lower[clean.lower()]
        # 4. 去掉"-S数字"或"-大小"等后缀
        clean2 = re.sub(r'\-.*', '', name).strip()
        if clean2 in id_to_signs:
            return id_to_signs[clean2]
        if clean2.lower() in id_to_signs_lower:
            return id_to_signs_lower[clean2.lower()]
        return {}
    
    # 检查 images 目录中哪些 case 有 _1 和 _2
    img_dir = './classification_models/data/images/'
    multi_cases = set()
    for d in os.listdir(img_dir):
        if d.endswith('_1') or d.endswith('_2'):
            base = d.rsplit('_', 1)[0]
            multi_cases.add(base)
    
    # 构建输出数据
    records = []
    split_count = 0
    unsplit_cases = []
    
    for idx, row in mapping_df.iterrows():
        original_name = str(row['original_case_name']).strip()
        converted_name = str(row['converted_case_name']).strip()
        label1 = str(row['group']).strip() if pd.notna(row['group']) else ''
        
        # 检查是否有多ROI
        if converted_name in multi_cases:
            # 去data2中查找匹配的记录
            clean = re.sub(r'[（(].*?[）)]', '', original_name).strip()
            matches = data1_df[data1_df.iloc[:, 0].str.strip().str.lower() == clean.lower()]
            if len(matches) == 0:
                matches = data1_df[data1_df.iloc[:, 0].str.strip().str.contains(clean, case=False, na=False)]
            
            lr_values = matches['LI-RADS等级'].dropna().unique()
            lr_values = [str(v).strip() for v in lr_values if str(v).strip()]
            lr_same = len(set(lr_values)) <= 1
            
            if lr_same and len(lr_values) > 0:
                # LR等级相同，拆分为 _1 和 _2
                label2 = normalize_label2(lr_values[0])
                signs = find_signs(original_name)
                rec = {
                    'casename': f"{converted_name}_1",
                    'label1': label1,
                    'label2': label2,
                }
                rec.update(signs)
                records.append(rec)
                rec = {
                    'casename': f"{converted_name}_2",
                    'label1': label1,
                    'label2': label2,
                }
                rec.update(signs)
                records.append(rec)
                split_count += 1
                continue
            elif len(lr_values) > 1:
                # LR等级不同，保持原样
                unsplit_cases.append((converted_name, original_name, lr_values))
        
        # 默认情况：不拆分
        label2 = find_label2(original_name)
        signs = find_signs(original_name)
        rec = {
            'casename': converted_name,
            'label1': label1,
            'label2': label2,
        }
        rec.update(signs)
        records.append(rec)
    
    all_df = pd.DataFrame(records)
    print(f"\n总记录数: {len(all_df)}")
    print(f"拆分为多ROI的case数: {split_count}")
    if unsplit_cases:
        print(f"因LR等级不同未拆分的case数: {len(unsplit_cases)}")
        for c, o, lrs in unsplit_cases:
            print(f"  {c} ({o}): LR={lrs}")
    
    # 按自然顺序排序
    all_df['sort_key'] = all_df['casename'].apply(natural_sort_key)
    all_df = all_df.sort_values('sort_key').drop('sort_key', axis=1)
    all_df = all_df.reset_index(drop=True)
    
    output_file_all = './classification_models/data/labels/test.txt'
    all_df.to_csv(output_file_all, sep='\t', index=False)
    print(f"test.txt已保存到: {output_file_all}")
    
    print("\n=== 统计信息 ===")
    print(f"总记录数: {len(all_df)}")
    remaining_missing = all_df['label2'].isna() | (all_df['label2'].str.strip() == '')
    print(f"缺失label2的记录数: {remaining_missing.sum()}")
    
    print("\nlabel2分布:")
    print(all_df['label2'].value_counts())

if __name__ == '__main__':
    main()
