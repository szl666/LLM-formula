#!/usr/bin/env python3
"""
演示如何加载保存的结构文件
"""

import pickle
import json
import os
from pymatgen.core.structure import Structure

def demo_load_structures():
    """
    演示如何从不同格式加载结构
    """
    structure_dir = '2d_structures'
    
    # 选择一个示例材料ID
    example_id = '2dm-1'
    
    print("=== Structure Loading Demo ===")
    print(f"Example material ID: {example_id}")
    print()
    
    # 1. 从pickle文件加载
    print("1. Loading from pickle file (.pkl):")
    pkl_file = os.path.join(structure_dir, f"{example_id}.pkl")
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            structure_pkl = pickle.load(f)
        print(f"   ✓ Loaded from {pkl_file}")
        print(f"   Formula: {structure_pkl.composition.reduced_formula}")
        print(f"   Lattice: {structure_pkl.lattice}")
        print(f"   Number of sites: {len(structure_pkl.sites)}")
    else:
        print(f"   ✗ File not found: {pkl_file}")
    
    print()
    
    # 2. 从JSON文件加载
    print("2. Loading from JSON file (.json):")
    json_file = os.path.join(structure_dir, f"{example_id}.json")
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            structure_dict = json.load(f)
        structure_json = Structure.from_dict(structure_dict)
        print(f"   ✓ Loaded from {json_file}")
        print(f"   Formula: {structure_json.composition.reduced_formula}")
        print(f"   Lattice: {structure_json.lattice}")
        print(f"   Number of sites: {len(structure_json.sites)}")
    else:
        print(f"   ✗ File not found: {json_file}")
    
    print()
    
    # 3. 从VASP文件加载
    print("3. Loading from VASP file (.vasp):")
    vasp_file = os.path.join(structure_dir, f"{example_id}.vasp")
    if os.path.exists(vasp_file):
        structure_vasp = Structure.from_file(vasp_file)
        print(f"   ✓ Loaded from {vasp_file}")
        print(f"   Formula: {structure_vasp.composition.reduced_formula}")
        print(f"   Lattice: {structure_vasp.lattice}")
        print(f"   Number of sites: {len(structure_vasp.sites)}")
    else:
        print(f"   ✗ File not found: {vasp_file}")
    
    print()
    
    # 4. 从CIF文件加载
    print("4. Loading from CIF file (.cif):")
    cif_file = os.path.join(structure_dir, f"{example_id}.cif")
    if os.path.exists(cif_file):
        try:
            structure_cif = Structure.from_file(cif_file)
            print(f"   ✓ Loaded from {cif_file}")
            print(f"   Formula: {structure_cif.composition.reduced_formula}")
            print(f"   Lattice: {structure_cif.lattice}")
            print(f"   Number of sites: {len(structure_cif.sites)}")
        except Exception as e:
            print(f"   ✗ Failed to load CIF: {e}")
    else:
        print(f"   ✗ File not found: {cif_file}")
    
    print()
    
    # 验证所有格式加载的结构是否相同
    print("5. Verifying structure consistency:")
    if os.path.exists(pkl_file) and os.path.exists(json_file) and os.path.exists(vasp_file):
        # 比较不同格式的结构
        structures_match = True
        
        # 比较晶格参数
        lat_pkl = structure_pkl.lattice
        lat_json = structure_json.lattice
        lat_vasp = structure_vasp.lattice
        
        tolerance = 1e-6
        if (abs(lat_pkl.a - lat_json.a) > tolerance or 
            abs(lat_pkl.a - lat_vasp.a) > tolerance):
            structures_match = False
        
        if structures_match:
            print("   ✓ All formats contain consistent structure data")
        else:
            print("   ✗ Structure data inconsistency detected")
    
    print()
    
    # 展示结构的详细信息
    if os.path.exists(pkl_file):
        print("6. Detailed structure information:")
        structure = structure_pkl
        print(f"   Material ID: {example_id}")
        print(f"   Chemical formula: {structure.composition.reduced_formula}")
        print(f"   Space group: {structure.get_space_group_info()}")
        print(f"   Lattice parameters:")
        print(f"     a = {structure.lattice.a:.4f} Å")
        print(f"     b = {structure.lattice.b:.4f} Å") 
        print(f"     c = {structure.lattice.c:.4f} Å")
        print(f"     α = {structure.lattice.alpha:.2f}°")
        print(f"     β = {structure.lattice.beta:.2f}°")
        print(f"     γ = {structure.lattice.gamma:.2f}°")
        print(f"   Volume: {structure.lattice.volume:.2f} Ų")
        print(f"   Density: {structure.density:.3f} g/cm³")
        print(f"   Number of atoms: {len(structure.sites)}")
        print(f"   Elements: {list(structure.composition.elements)}")

def list_available_materials():
    """
    列出前10个可用的材料ID
    """
    structure_dir = '2d_structures'
    
    pkl_files = [f for f in os.listdir(structure_dir) if f.endswith('.pkl')]
    material_ids = [f.replace('.pkl', '') for f in pkl_files]
    material_ids.sort()
    
    print("\n=== Available Materials (first 10) ===")
    for i, mat_id in enumerate(material_ids[:10]):
        print(f"{i+1:2d}. {mat_id}")
    
    print(f"\nTotal materials available: {len(material_ids)}")
    
    return material_ids

def main():
    """
    主函数
    """
    demo_load_structures()
    list_available_materials()
    
    print("\n=== Usage Examples ===")
    print("To load any structure in Python:")
    print("```python")
    print("import pickle")
    print("with open('2d_structures/2dm-1.pkl', 'rb') as f:")
    print("    structure = pickle.load(f)")
    print("print(structure)")
    print("```")
    print()
    print("Or use JSON format:")
    print("```python")
    print("from pymatgen.core.structure import Structure")
    print("import json")
    print("with open('2d_structures/2dm-1.json', 'r') as f:")
    print("    structure_dict = json.load(f)")
    print("structure = Structure.from_dict(structure_dict)")
    print("```")

if __name__ == "__main__":
    main()

