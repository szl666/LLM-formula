#!/usr/bin/env python3
"""
清理C2DB特征文件，去除矩阵、向量、数字等大型特征
"""

import pandas as pd
import numpy as np
import re
import os

def should_remove_feature(column_name):
    """
    判断是否应该去除某个特征列
    """
    # 保留的重要列
    if column_name in ['material_id', 'formula', 'target']:
        return False
    
    # 转为小写进行匹配
    col_lower = column_name.lower()
    
    # 去除包含特定关键词的特征
    keywords_to_remove = ['eig', 'matrix', 'vector', 'fourier', 'soap', 'acsf']
    for keyword in keywords_to_remove:
        if keyword in col_lower:
            return True
    
    # 去除包含数字的特征（使用正则表达式）
    if re.search(r'\d', column_name):
        return True
    
    return False

def clean_c2db_features(input_file, output_file):
    """
    清理C2DB特征文件
    """
    print(f"Loading C2DB features from {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        return None
    
    df = pd.read_csv(input_file)
    
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {df.shape[1]}")
    
    # 找出要去除的列
    columns_to_remove = []
    columns_to_keep = []
    
    for col in df.columns:
        if should_remove_feature(col):
            columns_to_remove.append(col)
        else:
            columns_to_keep.append(col)
    
    print(f"\nFound {len(columns_to_remove)} columns to remove")
    print(f"Will keep {len(columns_to_keep)} columns")
    
    # 显示一些要去除的特征示例
    print("\nColumns to remove (first 20 examples):")
    for i, col in enumerate(columns_to_remove[:20]):
        reason = ""
        col_lower = col.lower()
        if 'eig' in col_lower:
            reason = "(contains 'eig')"
        elif 'matrix' in col_lower:
            reason = "(contains 'matrix')"
        elif 'vector' in col_lower:
            reason = "(contains 'vector')"
        elif 'fourier' in col_lower:
            reason = "(contains 'fourier')"
        elif 'soap' in col_lower:
            reason = "(contains 'soap')"
        elif 'acsf' in col_lower:
            reason = "(contains 'acsf')"
        elif re.search(r'\d', col):
            reason = "(contains numbers)"
        
        print(f"  {i+1}. {col} {reason}")
    
    if len(columns_to_remove) > 20:
        print(f"  ... and {len(columns_to_remove) - 20} more")
    
    # 显示保留的特征类型
    print(f"\nColumns to keep (first 20 examples):")
    for i, col in enumerate(columns_to_keep[:20]):
        print(f"  {i+1}. {col}")
    if len(columns_to_keep) > 20:
        print(f"  ... and {len(columns_to_keep) - 20} more")
    
    # 去除指定的特征列
    df_cleaned = df[columns_to_keep]
    
    print(f"\nAfter cleaning:")
    print(f"New shape: {df_cleaned.shape}")
    print(f"Columns removed: {len(columns_to_remove)}")
    print(f"Columns remaining: {df_cleaned.shape[1]}")
    print(f"Reduction: {df.shape[1]} → {df_cleaned.shape[1]} ({len(columns_to_remove)} columns removed)")
    print(f"Reduction percentage: {(len(columns_to_remove) / df.shape[1] * 100):.1f}%")
    
    # 检查文件大小估算
    original_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    new_size_mb = df_cleaned.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"\nMemory usage:")
    print(f"Original: {original_size_mb:.1f} MB")
    print(f"New: {new_size_mb:.1f} MB")
    print(f"Reduction: {((original_size_mb - new_size_mb) / original_size_mb * 100):.1f}%")
    
    # 保存清理后的文件
    print(f"\nSaving cleaned data to {output_file}...")
    df_cleaned.to_csv(output_file, index=False)
    
    print(f"✓ C2DB feature cleaning completed!")
    print(f"✓ Cleaned file saved: {output_file}")
    
    return df_cleaned

def analyze_remaining_features(df_cleaned):
    """
    分析剩余特征的类型
    """
    remaining_columns = [col for col in df_cleaned.columns if col not in ['material_id', 'formula', 'target']]
    
    # 按特征类型分类
    magpie_features = [col for col in remaining_columns if col.startswith('MagpieData')]
    oxidation_features = [col for col in remaining_columns if 'oxidation' in col.lower()]
    ionic_features = [col for col in remaining_columns if 'ionic' in col.lower()]
    coordination_features = [col for col in remaining_columns if any(x in col.lower() for x in ['cn_', 'coordination'])]
    bond_features = [col for col in remaining_columns if any(x in col.lower() for x in ['bond', 'bd_'])]
    cluster_features = [col for col in remaining_columns if 'cluster' in col.lower()]
    other_features = [col for col in remaining_columns if not any([
        col.startswith('MagpieData'),
        'oxidation' in col.lower(),
        'ionic' in col.lower(),
        any(x in col.lower() for x in ['cn_', 'coordination']),
        any(x in col.lower() for x in ['bond', 'bd_']),
        'cluster' in col.lower()
    ])]
    
    print(f"\nRemaining feature breakdown:")
    print(f"  - MagpieData composition features: {len(magpie_features)}")
    print(f"  - Oxidation state features: {len(oxidation_features)}")
    print(f"  - Ionic character features: {len(ionic_features)}")
    print(f"  - Coordination features: {len(coordination_features)}")
    print(f"  - Bond features: {len(bond_features)}")
    print(f"  - Cluster features: {len(cluster_features)}")
    print(f"  - Other features: {len(other_features)}")
    print(f"  - Total features: {len(remaining_columns)}")
    print(f"  - Plus metadata: material_id, formula, target")
    
    if other_features:
        print(f"\nOther features (first 10):")
        for i, col in enumerate(other_features[:10]):
            print(f"    {i+1}. {col}")
        if len(other_features) > 10:
            print(f"    ... and {len(other_features) - 10} more")

def main():
    """
    主函数 - 清理PBE和HSE特征文件
    """
    print("="*70)
    print("C2DB FEATURE CLEANING")
    print("Removing: eig, matrix, vector, fourier, soap, acsf, and numeric features")
    print("="*70)
    
    # 清理PBE特征文件
    pbe_input = 'c2db_pbe_features.csv'
    pbe_output = 'c2db_pbe_features_minimal.csv'
    
    if os.path.exists(pbe_input):
        print(f"\n{'='*50}")
        print("CLEANING PBE FEATURES")
        print(f"{'='*50}")
        
        df_pbe_cleaned = clean_c2db_features(pbe_input, pbe_output)
        
        if df_pbe_cleaned is not None:
            analyze_remaining_features(df_pbe_cleaned)
    else:
        print(f"PBE feature file not found: {pbe_input}")
    
    # 清理HSE特征文件
    hse_input = 'c2db_hse_features.csv'
    hse_output = 'c2db_hse_features_minimal.csv'
    
    if os.path.exists(hse_input):
        print(f"\n{'='*50}")
        print("CLEANING HSE FEATURES")
        print(f"{'='*50}")
        
        df_hse_cleaned = clean_c2db_features(hse_input, hse_output)
        
        if df_hse_cleaned is not None:
            analyze_remaining_features(df_hse_cleaned)
    else:
        print(f"HSE feature file not found: {hse_input}")
    
    print("\n" + "="*70)
    print("C2DB FEATURE CLEANING COMPLETED!")
    print("="*70)
    
    if os.path.exists(pbe_output):
        print(f"PBE cleaned features: {pbe_output}")
    if os.path.exists(hse_output):
        print(f"HSE cleaned features: {hse_output}")
    
    print("\nNext step: Run symbolic regression on cleaned features")

if __name__ == "__main__":
    main()
