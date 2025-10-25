#!/usr/bin/env python3
"""
从2Dmatpedia.json文件中提取结构和带隙信息，生成CSV文件
"""

import json
import pandas as pd
from pymatgen.core.structure import Structure
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

def load_2d_materials_bandgap_data(json_file):
    """
    从JSON文件中加载2D材料数据，提取结构和带隙信息
    """
    materials = []
    material_ids = []
    formulas = []
    structures = []
    bandgaps = []
    is_metals = []
    
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
                is_metal = data.get('bandstructure', {}).get('is_metal', None)
                if is_metal is None:
                    # 根据带隙判断金属性质
                    is_metal = (bandgap == 0.0)
                
                # 提取结构信息
                structure_dict = data['structure']
                try:
                    structure = Structure.from_dict(structure_dict)
                except Exception as e:
                    print(f"Warning: Failed to parse structure for {material_id}: {e}")
                    continue
                
                # 存储数据
                materials.append(data)
                material_ids.append(material_id)
                formulas.append(formula)
                structures.append(structure)
                bandgaps.append(bandgap)
                is_metals.append(is_metal)
                
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON at line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue
    
    print(f"Successfully loaded {len(materials)} materials with complete bandgap data")
    
    return {
        'materials': materials,
        'material_ids': material_ids,
        'formulas': formulas,
        'structures': structures,
        'bandgaps': bandgaps,
        'is_metals': is_metals
    }

def create_bandgap_csv(data):
    """
    创建包含结构和带隙信息的CSV文件
    """
    structures = data['structures']
    material_ids = data['material_ids']
    formulas = data['formulas']
    bandgaps = data['bandgaps']
    is_metals = data['is_metals']
    
    print(f"Creating CSV file with {len(structures)} materials...")
    
    # 创建基础DataFrame
    df_data = {
        'material_id': material_ids,
        'formula': formulas,
        'bandgap_eV': bandgaps,
        'is_metal': is_metals,
        'structure': structures  # pymatgen Structure对象
    }
    
    # 提取结构的基本信息
    lattice_a = []
    lattice_b = []
    lattice_c = []
    lattice_alpha = []
    lattice_beta = []
    lattice_gamma = []
    lattice_volume = []
    num_sites = []
    density = []
    space_group = []
    
    for i, structure in enumerate(structures):
        if i % 100 == 0:
            print(f"Processing structure {i+1}/{len(structures)}...")
        
        try:
            # 晶格参数
            lattice = structure.lattice
            lattice_a.append(lattice.a)
            lattice_b.append(lattice.b)
            lattice_c.append(lattice.c)
            lattice_alpha.append(lattice.alpha)
            lattice_beta.append(lattice.beta)
            lattice_gamma.append(lattice.gamma)
            lattice_volume.append(lattice.volume)
            
            # 结构信息
            num_sites.append(len(structure.sites))
            density.append(structure.density)
            
            # 空间群信息
            try:
                from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                sga = SpacegroupAnalyzer(structure)
                sg_number = sga.get_space_group_number()
                space_group.append(sg_number)
            except:
                space_group.append(None)
                
        except Exception as e:
            print(f"Warning: Failed to extract structure info for material {i}: {e}")
            # 填充默认值
            lattice_a.append(None)
            lattice_b.append(None)
            lattice_c.append(None)
            lattice_alpha.append(None)
            lattice_beta.append(None)
            lattice_gamma.append(None)
            lattice_volume.append(None)
            num_sites.append(None)
            density.append(None)
            space_group.append(None)
    
    # 添加结构特征到DataFrame
    df_data.update({
        'lattice_a': lattice_a,
        'lattice_b': lattice_b,
        'lattice_c': lattice_c,
        'lattice_alpha': lattice_alpha,
        'lattice_beta': lattice_beta,
        'lattice_gamma': lattice_gamma,
        'lattice_volume': lattice_volume,
        'num_sites': num_sites,
        'density': density,
        'space_group': space_group
    })
    
    # 创建DataFrame
    df = pd.DataFrame(df_data)
    
    # 保存到CSV文件
    output_file = '2d_materials_structure_bandgap.csv'
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
    
    return df

def main():
    """
    主函数
    """
    json_file = '2Dmatpedia.json'
    
    print("=== 2D Materials Structure-Bandgap Data Extraction ===")
    
    # 加载数据
    data = load_2d_materials_bandgap_data(json_file)
    
    # 创建CSV文件
    df = create_bandgap_csv(data)
    
    print("\n=== Extraction Complete ===")

if __name__ == "__main__":
    main()

