#!/usr/bin/env python3
"""
从2Dmatpedia.json文件中提取结构，将每个结构单独保存到文件中
"""

import json
import os
from pymatgen.core.structure import Structure
import warnings
import pickle

# 忽略警告
warnings.filterwarnings('ignore')

def extract_and_save_structures(json_file, output_dir='2d_structures'):
    """
    从JSON文件中提取结构并单独保存
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    print(f"Loading structures from {json_file}...")
    
    successful_saves = 0
    failed_saves = 0
    
    with open(json_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 1000 == 0:
                print(f"Processed {line_num} lines, saved {successful_saves} structures...")
            
            try:
                data = json.loads(line.strip())
                
                # 检查是否包含结构信息
                if 'structure' not in data:
                    continue
                
                # 提取材料ID
                material_id = data.get('material_id', f'unknown_{line_num}')
                
                # 提取结构信息
                structure_dict = data['structure']
                try:
                    structure = Structure.from_dict(structure_dict)
                except Exception as e:
                    print(f"Warning: Failed to parse structure for {material_id}: {e}")
                    failed_saves += 1
                    continue
                
                # 保存结构到文件
                # 使用多种格式保存
                
                # 1. 保存为pickle文件 (Python对象)
                pickle_file = os.path.join(output_dir, f"{material_id}.pkl")
                with open(pickle_file, 'wb') as pf:
                    pickle.dump(structure, pf)
                
                # 2. 保存为JSON文件 (pymatgen字典格式)
                json_file_path = os.path.join(output_dir, f"{material_id}.json")
                with open(json_file_path, 'w') as jf:
                    json.dump(structure.as_dict(), jf, indent=2)
                
                # 3. 保存为POSCAR文件 (VASP格式)
                poscar_file = os.path.join(output_dir, f"{material_id}.vasp")
                structure.to(filename=poscar_file, fmt='poscar')
                
                # 4. 保存为CIF文件 (晶体学信息文件)
                cif_file = os.path.join(output_dir, f"{material_id}.cif")
                try:
                    structure.to(filename=cif_file, fmt='cif')
                except:
                    # CIF格式可能失败，跳过
                    pass
                
                successful_saves += 1
                
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON at line {line_num}: {e}")
                failed_saves += 1
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                failed_saves += 1
                continue
    
    print(f"\n=== Structure Extraction Complete ===")
    print(f"Successfully saved: {successful_saves} structures")
    print(f"Failed to save: {failed_saves} structures")
    print(f"Structures saved in directory: {output_dir}")
    print(f"File formats saved:")
    print(f"  - .pkl: Python pickle files (pymatgen Structure objects)")
    print(f"  - .json: JSON files (pymatgen dictionary format)")
    print(f"  - .vasp: POSCAR files (VASP format)")
    print(f"  - .cif: CIF files (Crystallographic Information Format)")
    
    return successful_saves, failed_saves

def create_structure_index(output_dir='2d_structures'):
    """
    创建结构文件的索引
    """
    index_file = os.path.join(output_dir, 'structure_index.txt')
    
    pkl_files = []
    json_files = []
    vasp_files = []
    cif_files = []
    
    for filename in os.listdir(output_dir):
        if filename.endswith('.pkl'):
            pkl_files.append(filename)
        elif filename.endswith('.json'):
            json_files.append(filename)
        elif filename.endswith('.vasp'):
            vasp_files.append(filename)
        elif filename.endswith('.cif'):
            cif_files.append(filename)
    
    with open(index_file, 'w') as f:
        f.write("2D Materials Structure Files Index\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total pickle files (.pkl): {len(pkl_files)}\n")
        f.write(f"Total JSON files (.json): {len(json_files)}\n")
        f.write(f"Total VASP files (.vasp): {len(vasp_files)}\n")
        f.write(f"Total CIF files (.cif): {len(cif_files)}\n\n")
        
        f.write("How to load structures:\n")
        f.write("-" * 30 + "\n")
        f.write("1. Python pickle files (.pkl):\n")
        f.write("   import pickle\n")
        f.write("   with open('material_id.pkl', 'rb') as f:\n")
        f.write("       structure = pickle.load(f)\n\n")
        
        f.write("2. JSON files (.json):\n")
        f.write("   from pymatgen.core.structure import Structure\n")
        f.write("   import json\n")
        f.write("   with open('material_id.json', 'r') as f:\n")
        f.write("       structure_dict = json.load(f)\n")
        f.write("   structure = Structure.from_dict(structure_dict)\n\n")
        
        f.write("3. VASP files (.vasp):\n")
        f.write("   from pymatgen.core.structure import Structure\n")
        f.write("   structure = Structure.from_file('material_id.vasp')\n\n")
        
        f.write("4. CIF files (.cif):\n")
        f.write("   from pymatgen.core.structure import Structure\n")
        f.write("   structure = Structure.from_file('material_id.cif')\n\n")
    
    print(f"Structure index created: {index_file}")

def main():
    """
    主函数
    """
    json_file = '2Dmatpedia.json'
    output_dir = '2d_structures'
    
    print("=== 2D Materials Structure Extraction ===")
    
    # 提取并保存结构
    successful_saves, failed_saves = extract_and_save_structures(json_file, output_dir)
    
    # 创建索引文件
    create_structure_index(output_dir)
    
    print(f"\n=== Summary ===")
    print(f"Total structures processed: {successful_saves + failed_saves}")
    print(f"Successfully saved: {successful_saves}")
    print(f"Failed to save: {failed_saves}")
    print(f"Success rate: {successful_saves/(successful_saves + failed_saves)*100:.1f}%")

if __name__ == "__main__":
    main()

