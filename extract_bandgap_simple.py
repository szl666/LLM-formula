#!/usr/bin/env python3
"""
从2Dmatpedia.json文件中提取结构和带隙信息，生成简洁的CSV文件
"""

import json
import pandas as pd
from pymatgen.core.structure import Structure
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

def load_and_process_bandgap_data(json_file):
    """
    从JSON文件中加载2D材料数据，提取结构和带隙信息
    """
    data_list = []
    
    print(f"Loading data from {json_file}...")
    
    with open(json_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 1000 == 0:
                print(f"Processed {line_num} lines...")
            
            try:
                data = json.loads(line.strip())
                
                # 检查是否包含必要的字段
                if 'structure' not in data or 'bandgap' not in data:
                    continue
                
                # 提取基本信息
                material_id = data.get('material_id', f'unknown_{line_num}')
                formula = data.get('formula_pretty', 'Unknown')
                
                # 提取带隙信息
                bandgap = data.get('bandgap', None)
                if bandgap is None:
                    continue
                
                # 提取金属性质
                bandstructure = data.get('bandstructure', {})
                is_metal = bandstructure.get('is_metal', None)
                if is_metal is None:
                    # 根据带隙判断金属性质
                    is_metal = (bandgap == 0.0)
                
                # 提取其他带结构信息
                cbm = bandstructure.get('cbm', None)  # 导带底
                vbm = bandstructure.get('vbm', None)  # 价带顶
                is_gap_direct = bandstructure.get('is_gap_direct', None)  # 是否直接带隙
                
                # 提取结构信息
                structure_dict = data['structure']
                try:
                    structure = Structure.from_dict(structure_dict)
                except Exception as e:
                    print(f"Warning: Failed to parse structure for {material_id}: {e}")
                    continue
                
                # 提取晶格参数
                lattice = structure.lattice
                
                # 提取空间群信息
                space_group_number = None
                space_group_symbol = None
                try:
                    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                    sga = SpacegroupAnalyzer(structure)
                    space_group_number = sga.get_space_group_number()
                    space_group_symbol = sga.get_space_group_symbol()
                except:
                    # 尝试从原始数据中获取空间群信息
                    spacegroup = data.get('spacegroup', {})
                    space_group_number = spacegroup.get('number', None)
                    space_group_symbol = spacegroup.get('symbol', None)
                
                # 提取元素信息
                elements = data.get('elements', [])
                nelements = data.get('nelements', len(elements))
                
                # 提取化学系统
                chemsys = data.get('chemsys', None)
                
                # 创建数据记录
                record = {
                    'material_id': material_id,
                    'formula': formula,
                    'bandgap_eV': bandgap,
                    'is_metal': is_metal,
                    'is_gap_direct': is_gap_direct,
                    'cbm': cbm,
                    'vbm': vbm,
                    'lattice_a': lattice.a,
                    'lattice_b': lattice.b,
                    'lattice_c': lattice.c,
                    'lattice_alpha': lattice.alpha,
                    'lattice_beta': lattice.beta,
                    'lattice_gamma': lattice.gamma,
                    'lattice_volume': lattice.volume,
                    'num_sites': len(structure.sites),
                    'density': structure.density,
                    'space_group_number': space_group_number,
                    'space_group_symbol': space_group_symbol,
                    'elements': ','.join(elements),
                    'nelements': nelements,
                    'chemsys': chemsys
                }
                
                data_list.append(record)
                
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON at line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue
    
    print(f"Successfully processed {len(data_list)} materials with complete bandgap data")
    
    return data_list

def create_bandgap_csv(data_list):
    """
    创建包含结构和带隙信息的CSV文件
    """
    print(f"Creating CSV file with {len(data_list)} materials...")
    
    # 创建DataFrame
    df = pd.DataFrame(data_list)
    
    # 保存到CSV文件
    output_file = '2d_materials_bandgap_simple.csv'
    df.to_csv(output_file, index=False)
    print(f"Bandgap dataset saved to: {output_file}")
    print(f"Dataset shape: {df.shape}")
    
    # 打印统计信息
    print("\nDataset statistics:")
    print(f"Total materials: {len(df)}")
    print(f"Metals (bandgap = 0): {sum(df['is_metal'])}")
    print(f"Semiconductors/Insulators (bandgap > 0): {sum(~df['is_metal'])}")
    print(f"Average bandgap: {df['bandgap_eV'].mean():.3f} eV")
    print(f"Bandgap range: {df['bandgap_eV'].min():.3f} - {df['bandgap_eV'].max():.3f} eV")
    
    # 按带隙分布统计
    print(f"\nBandgap distribution:")
    print(f"  Metals (0.0 eV): {sum(df['bandgap_eV'] == 0.0)}")
    print(f"  Small gap (0.0 < gap <= 0.5 eV): {sum((df['bandgap_eV'] > 0.0) & (df['bandgap_eV'] <= 0.5))}")
    print(f"  Medium gap (0.5 < gap <= 2.0 eV): {sum((df['bandgap_eV'] > 0.5) & (df['bandgap_eV'] <= 2.0))}")
    print(f"  Large gap (gap > 2.0 eV): {sum(df['bandgap_eV'] > 2.0)}")
    
    # 按元素数量统计
    print(f"\nElement count distribution:")
    element_counts = df['nelements'].value_counts().sort_index()
    for n_elem, count in element_counts.items():
        print(f"  {n_elem} elements: {count} materials")
    
    return df

def main():
    """
    主函数
    """
    json_file = '2Dmatpedia.json'
    
    print("=== 2D Materials Structure-Bandgap Data Extraction ===")
    
    # 加载和处理数据
    data_list = load_and_process_bandgap_data(json_file)
    
    # 创建CSV文件
    df = create_bandgap_csv(data_list)
    
    print("\n=== Extraction Complete ===")

if __name__ == "__main__":
    main()

