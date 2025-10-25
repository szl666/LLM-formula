#!/usr/bin/env python3
"""
使用automatminer对C2DB结构进行特征化
处理PBE和HSE带隙数据
"""

import pandas as pd
import numpy as np
import os
import pickle
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
import warnings
warnings.filterwarnings('ignore')

# Apply pandas compatibility patch for matminer
try:
    import pandas_patch
except ImportError:
    # If patch file not found, apply patch inline
    if not hasattr(pd.DataFrame, 'append'):
        def append_method(self, other, ignore_index=False, verify_integrity=False, sort=False):
            return pd.concat([self, other], ignore_index=ignore_index, 
                           verify_integrity=verify_integrity, sort=sort)
        pd.DataFrame.append = append_method
        print("Added pandas DataFrame.append compatibility method")
    else:
        print("pandas DataFrame.append method already exists")

from automatminer import MatPipe

def load_bandgap_data(csv_file):
    """
    加载带隙数据CSV文件
    """
    print(f"Loading bandgap data from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} materials")
    return df

def load_structures_from_cifs(df, cif_dir='cif_files'):
    """
    从CIF文件加载结构
    """
    print(f"Loading structures from {cif_dir}...")
    
    structures = []
    material_ids = []
    formulas = []
    bandgaps = []
    failed_count = 0
    
    for idx, row in df.iterrows():
        material_id = row['material_id']
        formula = row['formula']
        
        # 确定带隙列名
        if 'bandgap_pbe' in row:
            bandgap = row['bandgap_pbe']
        elif 'bandgap_hse' in row:
            bandgap = row['bandgap_hse']
        else:
            print(f"Warning: No bandgap column found for material {material_id}")
            continue
        
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
                print(f"  Loaded {len(structures)} structures...")
                
        except Exception as e:
            failed_count += 1
            continue
    
    print(f"Successfully loaded {len(structures)} structures")
    print(f"Failed to load: {failed_count} materials")
    
    return structures, material_ids, formulas, bandgaps

def create_features_with_automatminer(structures, material_ids, formulas, bandgaps, gap_type):
    """
    使用automatminer创建特征
    """
    print(f"\n=== CREATING FEATURES FOR {gap_type} DATASET ===")
    print(f"Processing {len(structures)} materials...")
    
    # 创建基础DataFrame
    df_base = pd.DataFrame({
        'material_id': material_ids,
        'formula': formulas,
        'structure': structures
    })
    
    print("Initializing MatPipe...")
    try:
        # 尝试使用express预设（更快）
        pipe = MatPipe.from_preset("express")
        pipe.ml_type = 'regression'
        print("✓ Using 'express' preset")
    except Exception as e:
        print(f"Error with express preset: {e}")
        print("Creating minimal MatPipe configuration...")
        from automatminer.featurization import AutoFeaturizer
        from automatminer.preprocessing import DataCleaner
        from matminer.featurizers.composition import ElementProperty
        from matminer.featurizers.structure import DensityFeatures
        
        # 创建最小配置
        featurizers = {
            "composition": [ElementProperty.from_preset("magpie")],
            "structure": [DensityFeatures()]
        }
        
        pipe = MatPipe(
            autofeaturizer=AutoFeaturizer(featurizers=featurizers),
            cleaner=DataCleaner(),
            reducer=None,
            learner=None
        )
        pipe.ml_type = 'regression'
        print("✓ Using minimal MatPipe configuration")
    
    # 添加临时目标列用于特征化
    df_temp = df_base.copy()
    df_temp['temp_target'] = bandgaps
    
    print("Running autofeaturizer...")
    try:
        df_features = pipe.autofeaturizer.fit_transform(df_temp, 'temp_target')
        print(f"Autofeaturizer completed. Shape: {df_features.shape}")
    except Exception as e:
        print(f"Error in autofeaturizer: {e}")
        return None
    
    print("Running cleaner...")
    try:
        df_features = pipe.cleaner.fit_transform(df_features, 'temp_target')
        print(f"Cleaner completed. Shape: {df_features.shape}")
    except Exception as e:
        print(f"Error in cleaner: {e}")
        return None
    
    # 移除临时目标列和结构列，保留material_id和formula
    feature_columns = [col for col in df_features.columns 
                      if col not in ['temp_target', 'structure']]
    df_features = df_features[feature_columns]
    
    # 添加真实的带隙目标
    # MatPipe可能会过滤掉一些材料，所以需要对齐
    num_final_materials = df_features.shape[0]
    num_original_materials = len(bandgaps)
    
    if num_final_materials != num_original_materials:
        print(f"Materials filtered by MatPipe: {num_original_materials - num_final_materials}")
        final_bandgaps = bandgaps[:num_final_materials]
    else:
        final_bandgaps = bandgaps
    
    df_features['target'] = final_bandgaps
    
    print(f"Generated {df_features.shape[1]-3} features")  # -3 for material_id, formula, target
    print(f"Final dataset: {df_features.shape}")
    
    return df_features

def save_features(df_features, gap_type):
    """
    保存特征数据
    """
    output_file = f'c2db_{gap_type.lower()}_features.csv'
    df_features.to_csv(output_file, index=False)
    print(f"Features saved: {output_file}")
    
    # 保存统计信息
    final_bandgaps = df_features['target'].values
    metals_count = sum(bg == 0.0 for bg in final_bandgaps)
    semiconductors_count = len(final_bandgaps) - metals_count
    
    print(f"\nFinal {gap_type} dataset statistics:")
    print(f"  - Total materials: {len(final_bandgaps)}")
    print(f"  - Metals (bandgap = 0): {metals_count} ({metals_count/len(final_bandgaps)*100:.1f}%)")
    print(f"  - Semiconductors/Insulators: {semiconductors_count} ({semiconductors_count/len(final_bandgaps)*100:.1f}%)")
    print(f"  - Bandgap range: {np.array(final_bandgaps).min():.3f} - {np.array(final_bandgaps).max():.3f} eV")
    print(f"  - Average bandgap: {np.array(final_bandgaps).mean():.3f} eV")
    
    return output_file

def process_dataset(csv_file, gap_type, cif_dir='cif_files'):
    """
    处理单个数据集
    """
    print(f"\n{'='*70}")
    print(f"PROCESSING {gap_type} DATASET")
    print(f"{'='*70}")
    
    # 加载带隙数据
    df = load_bandgap_data(csv_file)
    
    # 加载结构
    structures, material_ids, formulas, bandgaps = load_structures_from_cifs(df, cif_dir)
    
    if len(structures) == 0:
        print(f"No structures loaded for {gap_type} dataset!")
        return None
    
    # 创建特征
    df_features = create_features_with_automatminer(structures, material_ids, formulas, bandgaps, gap_type)
    
    if df_features is None:
        print(f"Feature creation failed for {gap_type} dataset!")
        return None
    
    # 保存特征
    output_file = save_features(df_features, gap_type)
    
    return output_file

def main():
    """
    主函数
    """
    print("="*70)
    print("C2DB FEATURE EXTRACTION WITH AUTOMATMINER")
    print("="*70)
    
    # 处理PBE数据集
    pbe_output = process_dataset('c2db_pbe_bandgaps.csv', 'PBE')
    
    # 处理HSE数据集
    hse_output = process_dataset('c2db_hse_bandgaps.csv', 'HSE')
    
    print("\n" + "="*70)
    print("C2DB FEATURE EXTRACTION COMPLETED!")
    print("="*70)
    
    if pbe_output:
        print(f"PBE features: {pbe_output}")
    if hse_output:
        print(f"HSE features: {hse_output}")
    
    print("\nNext steps:")
    print("1. Remove matrix features from the generated files")
    print("2. Run symbolic regression on the cleaned features")

if __name__ == "__main__":
    main()
