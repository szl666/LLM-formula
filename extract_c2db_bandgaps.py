#!/usr/bin/env python3
"""
从C2DB数据库提取PBE和HSE带隙数据
properties.csv包含性质，cif_files目录包含结构文件
"""

import pandas as pd
import numpy as np
import os
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
import warnings
warnings.filterwarnings('ignore')

def load_c2db_properties(properties_file='properties.csv'):
    """
    加载C2DB性质数据
    """
    print(f"Loading properties from {properties_file}...")
    df = pd.read_csv(properties_file)
    
    print(f"Total materials in C2DB: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # 检查带隙相关列
    bandgap_columns = [col for col in df.columns if 'gap' in col.lower()]
    print(f"Bandgap-related columns: {bandgap_columns}")
    
    return df

def analyze_bandgap_data(df):
    """
    分析带隙数据的分布
    """
    print("\n=== BANDGAP DATA ANALYSIS ===")
    
    # PBE带隙分析
    pbe_gap_col = 'gap'  # PBE带隙
    pbe_gap_dir_col = 'gap_dir'  # PBE直接带隙
    
    # HSE带隙分析
    hse_gap_col = 'gap_hse'  # HSE带隙
    hse_gap_dir_col = 'gap_dir_hse'  # HSE直接带隙
    
    print(f"\nPBE bandgap ({pbe_gap_col}):")
    pbe_valid = df[pbe_gap_col].notna()
    print(f"  Valid entries: {pbe_valid.sum()} / {len(df)} ({pbe_valid.sum()/len(df)*100:.1f}%)")
    if pbe_valid.sum() > 0:
        pbe_values = pd.to_numeric(df[pbe_gap_col][pbe_valid], errors='coerce')
        pbe_values = pbe_values.dropna()  # 去除转换失败的值
        if len(pbe_values) > 0:
            print(f"  Range: {pbe_values.min():.3f} - {pbe_values.max():.3f} eV")
            print(f"  Mean: {pbe_values.mean():.3f} eV")
            print(f"  Metals (gap=0): {(pbe_values == 0).sum()} ({(pbe_values == 0).sum()/len(pbe_values)*100:.1f}%)")
    
    print(f"\nPBE direct bandgap ({pbe_gap_dir_col}):")
    pbe_dir_valid = df[pbe_gap_dir_col].notna()
    print(f"  Valid entries: {pbe_dir_valid.sum()} / {len(df)} ({pbe_dir_valid.sum()/len(df)*100:.1f}%)")
    if pbe_dir_valid.sum() > 0:
        pbe_dir_values = pd.to_numeric(df[pbe_gap_dir_col][pbe_dir_valid], errors='coerce')
        pbe_dir_values = pbe_dir_values.dropna()
        if len(pbe_dir_values) > 0:
            print(f"  Range: {pbe_dir_values.min():.3f} - {pbe_dir_values.max():.3f} eV")
            print(f"  Mean: {pbe_dir_values.mean():.3f} eV")
    
    print(f"\nHSE bandgap ({hse_gap_col}):")
    hse_valid = df[hse_gap_col].notna()
    print(f"  Valid entries: {hse_valid.sum()} / {len(df)} ({hse_valid.sum()/len(df)*100:.1f}%)")
    if hse_valid.sum() > 0:
        hse_values = pd.to_numeric(df[hse_gap_col][hse_valid], errors='coerce')
        hse_values = hse_values.dropna()
        if len(hse_values) > 0:
            print(f"  Range: {hse_values.min():.3f} - {hse_values.max():.3f} eV")
            print(f"  Mean: {hse_values.mean():.3f} eV")
            print(f"  Metals (gap=0): {(hse_values == 0).sum()} ({(hse_values == 0).sum()/len(hse_values)*100:.1f}%)")
    
    print(f"\nHSE direct bandgap ({hse_gap_dir_col}):")
    hse_dir_valid = df[hse_gap_dir_col].notna()
    print(f"  Valid entries: {hse_dir_valid.sum()} / {len(df)} ({hse_dir_valid.sum()/len(df)*100:.1f}%)")
    if hse_dir_valid.sum() > 0:
        hse_dir_values = pd.to_numeric(df[hse_gap_dir_col][hse_dir_valid], errors='coerce')
        hse_dir_values = hse_dir_values.dropna()
        if len(hse_dir_values) > 0:
            print(f"  Range: {hse_dir_values.min():.3f} - {hse_dir_values.max():.3f} eV")
            print(f"  Mean: {hse_dir_values.mean():.3f} eV")
    
    return {
        'pbe_valid': pbe_valid,
        'hse_valid': hse_valid,
        'pbe_dir_valid': pbe_dir_valid,
        'hse_dir_valid': hse_dir_valid
    }

def extract_structures_and_bandgaps(df, cif_dir='cif_files'):
    """
    提取结构和带隙数据
    """
    print(f"\n=== EXTRACTING STRUCTURES AND BANDGAPS ===")
    
    # 检查cif文件目录
    if not os.path.exists(cif_dir):
        raise FileNotFoundError(f"CIF directory not found: {cif_dir}")
    
    cif_files = os.listdir(cif_dir)
    print(f"Found {len(cif_files)} CIF files in {cif_dir}")
    
    # 提取PBE数据
    pbe_data = extract_dataset(df, cif_dir, 'gap', 'PBE')
    
    # 提取HSE数据
    hse_data = extract_dataset(df, cif_dir, 'gap_hse', 'HSE')
    
    return pbe_data, hse_data

def extract_dataset(df, cif_dir, gap_column, gap_type):
    """
    提取特定类型的带隙数据集
    """
    print(f"\nExtracting {gap_type} dataset...")
    
    # 筛选有效的带隙数据
    valid_mask = df[gap_column].notna()
    df_valid = df[valid_mask].copy()
    
    # 确保带隙列是数值类型
    df_valid[gap_column] = pd.to_numeric(df_valid[gap_column], errors='coerce')
    df_valid = df_valid[df_valid[gap_column].notna()]
    
    print(f"Materials with valid {gap_type} bandgap: {len(df_valid)}")
    
    structures = []
    material_ids = []
    formulas = []
    bandgaps = []
    failed_count = 0
    
    for idx, row in df_valid.iterrows():
        material_id = row['ID']
        formula = row['formula']
        bandgap = row[gap_column]
        
        # 构建CIF文件路径
        cif_file = os.path.join(cif_dir, f"{material_id}.cif")
        
        if not os.path.exists(cif_file):
            failed_count += 1
            continue
        
        try:
            # 读取CIF文件
            parser = CifParser(cif_file)
            structure = parser.get_structures()[0]  # 取第一个结构
            
            structures.append(structure)
            material_ids.append(material_id)
            formulas.append(formula)
            bandgaps.append(float(bandgap))
            
            if len(structures) % 100 == 0:
                print(f"  Processed {len(structures)} structures...")
                
        except Exception as e:
            failed_count += 1
            continue
    
    print(f"Successfully extracted {len(structures)} {gap_type} structures")
    print(f"Failed to process: {failed_count} materials")
    
    # 统计带隙分布
    bandgaps_array = np.array(bandgaps)
    metals_count = sum(bg == 0.0 for bg in bandgaps)
    semiconductors_count = len(bandgaps) - metals_count
    
    print(f"{gap_type} bandgap statistics:")
    print(f"  - Metals (bandgap = 0): {metals_count} ({metals_count/len(bandgaps)*100:.1f}%)")
    print(f"  - Semiconductors/Insulators: {semiconductors_count} ({semiconductors_count/len(bandgaps)*100:.1f}%)")
    print(f"  - Bandgap range: {bandgaps_array.min():.3f} - {bandgaps_array.max():.3f} eV")
    print(f"  - Average bandgap: {bandgaps_array.mean():.3f} eV")
    
    return {
        'structures': structures,
        'material_ids': material_ids,
        'formulas': formulas,
        'bandgaps': bandgaps,
        'gap_type': gap_type
    }

def save_simple_datasets(pbe_data, hse_data):
    """
    保存简单的CSV数据集（不包含结构）
    """
    print(f"\n=== SAVING SIMPLE DATASETS ===")
    
    # 保存PBE数据
    pbe_df = pd.DataFrame({
        'material_id': pbe_data['material_ids'],
        'formula': pbe_data['formulas'],
        'bandgap_pbe': pbe_data['bandgaps']
    })
    pbe_file = 'c2db_pbe_bandgaps.csv'
    pbe_df.to_csv(pbe_file, index=False)
    print(f"PBE dataset saved: {pbe_file} ({len(pbe_df)} materials)")
    
    # 保存HSE数据
    hse_df = pd.DataFrame({
        'material_id': hse_data['material_ids'],
        'formula': hse_data['formulas'],
        'bandgap_hse': hse_data['bandgaps']
    })
    hse_file = 'c2db_hse_bandgaps.csv'
    hse_df.to_csv(hse_file, index=False)
    print(f"HSE dataset saved: {hse_file} ({len(hse_df)} materials)")
    
    return pbe_df, hse_df

def main():
    """
    主函数
    """
    print("="*70)
    print("C2DB BANDGAP DATA EXTRACTION")
    print("="*70)
    
    # 加载性质数据
    df = load_c2db_properties('properties.csv')
    
    # 分析带隙数据
    bandgap_info = analyze_bandgap_data(df)
    
    # 提取结构和带隙数据
    pbe_data, hse_data = extract_structures_and_bandgaps(df, 'cif_files')
    
    # 保存简单数据集
    pbe_df, hse_df = save_simple_datasets(pbe_data, hse_data)
    
    print("\n" + "="*70)
    print("C2DB BANDGAP EXTRACTION COMPLETED!")
    print("="*70)
    print(f"PBE dataset: {len(pbe_data['structures'])} materials")
    print(f"HSE dataset: {len(hse_data['structures'])} materials")
    print("\nOutput files:")
    print("- c2db_pbe_bandgaps.csv: PBE bandgap data")
    print("- c2db_hse_bandgaps.csv: HSE bandgap data")
    print("\nNext steps:")
    print("1. Use automatminer to extract features from structures")
    print("2. Remove matrix features")
    print("3. Run symbolic regression")
    
    # 返回数据供后续使用
    return pbe_data, hse_data

if __name__ == "__main__":
    pbe_data, hse_data = main()
