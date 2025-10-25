#!/usr/bin/env python3
"""
Script to extract 2D material structures and their exfoliation/decomposition energies 
from 2Dmatpedia.json and create feature-engineered datasets.
"""

import json
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from automatminer import MatPipe
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_2d_materials_data(json_file):
    """
    Load 2D materials data from JSON file.
    Each line contains a separate JSON object.
    """
    materials = []
    
    print(f"Loading data from {json_file}...")
    with open(json_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    material = json.loads(line)
                    materials.append(material)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue
    
    print(f"Loaded {len(materials)} materials from {json_file}")
    return materials

def extract_structures_and_energies(materials):
    """
    Extract pymatgen structures and energy data from materials list.
    """
    structures = []
    material_ids = []
    formulas = []
    exfoliation_energies = []
    decomposition_energies = []
    
    print("Extracting structures and energies...")
    
    for i, material in enumerate(materials):
        try:
            # Extract structure
            structure_dict = material.get('structure')
            if structure_dict is None:
                print(f"Material {i}: No structure data found")
                continue
                
            # Convert to pymatgen Structure
            structure = Structure.from_dict(structure_dict)
            
            # Extract energies
            exf_energy = material.get('exfoliation_energy_per_atom')
            dec_energy = material.get('decomposition_energy_per_atom')
            
            # Skip if missing energy data
            if exf_energy is None or dec_energy is None:
                print(f"Material {i}: Missing energy data (exf: {exf_energy}, dec: {dec_energy})")
                continue
            
            # Store data
            structures.append(structure)
            material_ids.append(material.get('material_id', f'material_{i}'))
            formulas.append(material.get('formula_pretty', structure.formula))
            exfoliation_energies.append(float(exf_energy))
            decomposition_energies.append(float(dec_energy))
            
            if len(structures) % 100 == 0:
                print(f"Processed {len(structures)} materials...")
                
        except Exception as e:
            print(f"Error processing material {i}: {e}")
            continue
    
    print(f"Successfully extracted {len(structures)} materials with complete data")
    
    return {
        'structures': structures,
        'material_ids': material_ids,
        'formulas': formulas,
        'exfoliation_energies': exfoliation_energies,
        'decomposition_energies': decomposition_energies
    }

def create_feature_datasets(data):
    """
    Create feature-engineered datasets using MatPipe.
    """
    structures = data['structures']
    material_ids = data['material_ids']
    formulas = data['formulas']
    exf_energies = data['exfoliation_energies']
    dec_energies = data['decomposition_energies']
    
    # Create base dataframe
    df_base = pd.DataFrame({
        'material_id': material_ids,
        'formula': formulas,
        'structure': structures
    })
    
    print("Starting feature extraction with MatPipe...")
    print("This may take a while for large datasets...")
    
    # Initialize MatPipe with heavy preset for comprehensive features
    pipe = MatPipe.from_preset("heavy")
    pipe.ml_type = 'regression'
    
    # Create exfoliation energy dataset
    print("\nCreating exfoliation energy dataset...")
    df_exf = df_base.copy()
    df_exf['target'] = exf_energies
    
    try:
        # Feature engineering
        df_exf_features = pipe.autofeaturizer.fit_transform(df_exf, 'target')
        df_exf_features = pipe.cleaner.fit_transform(df_exf_features, 'target')
        
        # Save exfoliation energy dataset
        df_exf_features.to_csv('2d_materials_exfoliation_energy_features.csv', index=False)
        print(f"Exfoliation energy dataset saved: {df_exf_features.shape}")
        
    except Exception as e:
        print(f"Error creating exfoliation energy dataset: {e}")
        df_exf_features = None
    
    # Create decomposition energy dataset
    print("\nCreating decomposition energy dataset...")
    df_dec = df_base.copy()
    df_dec['target'] = dec_energies
    
    try:
        # Note: We can reuse the same featurizer since structures are the same
        df_dec_features = df_exf_features.copy() if df_exf_features is not None else None
        
        if df_dec_features is not None:
            # Replace target column
            df_dec_features['target'] = dec_energies
            # Save decomposition energy dataset
            df_dec_features.to_csv('2d_materials_decomposition_energy_features.csv', index=False)
            print(f"Decomposition energy dataset saved: {df_dec_features.shape}")
        else:
            # If exfoliation failed, try fresh featurization
            df_dec_features = pipe.autofeaturizer.fit_transform(df_dec, 'target')
            df_dec_features = pipe.cleaner.fit_transform(df_dec_features, 'target')
            df_dec_features.to_csv('2d_materials_decomposition_energy_features.csv', index=False)
            print(f"Decomposition energy dataset saved: {df_dec_features.shape}")
            
    except Exception as e:
        print(f"Error creating decomposition energy dataset: {e}")
        df_dec_features = None
    
    return df_exf_features, df_dec_features

def main():
    """
    Main function to process 2Dmatpedia data and create feature datasets.
    """
    json_file = '2Dmatpedia.json'
    
    # Load materials data
    materials = load_2d_materials_data(json_file)
    
    if not materials:
        print("No materials data loaded. Exiting.")
        return
    
    # Extract structures and energies
    data = extract_structures_and_energies(materials)
    
    if not data['structures']:
        print("No valid structures extracted. Exiting.")
        return
    
    # Create feature datasets
    df_exf, df_dec = create_feature_datasets(data)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total materials processed: {len(data['structures'])}")
    print(f"Exfoliation energy range: {min(data['exfoliation_energies']):.4f} to {max(data['exfoliation_energies']):.4f}")
    print(f"Decomposition energy range: {min(data['decomposition_energies']):.4f} to {max(data['decomposition_energies']):.4f}")
    
    if df_exf is not None:
        print(f"Exfoliation energy features: {df_exf.shape[1]-1} features")
        print("Saved: 2d_materials_exfoliation_energy_features.csv")
    
    if df_dec is not None:
        print(f"Decomposition energy features: {df_dec.shape[1]-1} features")
        print("Saved: 2d_materials_decomposition_energy_features.csv")
    
    print("\nFeature extraction completed!")

if __name__ == "__main__":
    main()
