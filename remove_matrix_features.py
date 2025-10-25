#!/usr/bin/env python3
"""
去除bandgap特征文件中的矩阵特征以减小文件大小
"""

import pandas as pd
import numpy as np

def remove_matrix_features(input_file, output_file):
    """
    去除包含'matrix'的特征列
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {df.shape[1]}")
    
    # 找出包含'matrix'的列
    matrix_columns = [col for col in df.columns if 'matrix' in col.lower()]
    print(f"Found {len(matrix_columns)} matrix feature columns")
    
    # 显示前几个矩阵特征列名
    print("Matrix feature columns (first 10):")
    for i, col in enumerate(matrix_columns[:10]):
        print(f"  {i+1}. {col}")
    if len(matrix_columns) > 10:
        print(f"  ... and {len(matrix_columns) - 10} more")
    
    # 去除矩阵特征列
    df_cleaned = df.drop(columns=matrix_columns)
    
    print(f"\nAfter removing matrix features:")
    print(f"New shape: {df_cleaned.shape}")
    print(f"Columns removed: {len(matrix_columns)}")
    print(f"Columns remaining: {df_cleaned.shape[1]}")
    print(f"Reduction: {df.shape[1]} → {df_cleaned.shape[1]} ({len(matrix_columns)} columns removed)")
    
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
    
    print(f"✓ Matrix features removed successfully!")
    print(f"✓ Cleaned file saved: {output_file}")
    
    return df_cleaned

if __name__ == "__main__":
    input_file = "2d_materials_bandgap_features.csv"
    output_file = "2d_materials_bandgap_features_no_matrix.csv"
    
    print("="*60)
    print("REMOVING MATRIX FEATURES FROM BANDGAP DATASET")
    print("="*60)
    
    df_cleaned = remove_matrix_features(input_file, output_file)
    
    print("\n" + "="*60)
    print("MATRIX FEATURE REMOVAL COMPLETED!")
    print("="*60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Final dataset shape: {df_cleaned.shape}")
    
    # 显示剩余特征的类型分布
    remaining_columns = [col for col in df_cleaned.columns if col not in ['material_id', 'formula', 'target']]
    magpie_features = [col for col in remaining_columns if col.startswith('MagpieData')]
    other_features = [col for col in remaining_columns if not col.startswith('MagpieData')]
    
    print(f"\nRemaining feature breakdown:")
    print(f"  - MagpieData composition features: {len(magpie_features)}")
    print(f"  - Other structural features: {len(other_features)}")
    print(f"  - Total features: {len(remaining_columns)}")
    print(f"  - Plus metadata: material_id, formula, target")
