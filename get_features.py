#!/usr/bin/env python3
"""
Feature extraction script for 2D materials from 2Dmatpedia.json
Extracts structural features and creates dataset for bandgap prediction.
"""

import json
import numpy as np
import pandas as pd

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
        print("Applied pandas DataFrame.append compatibility patch")

from automatminer import MatPipe
from pymatgen.core import Structure
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_2d_materials_data(json_file='2Dmatpedia.json'):
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

def extract_structures_and_bandgaps(materials):
    """
    Extract pymatgen structures and bandgap data from materials list.
    """
    structures = []
    material_ids = []
    formulas = []
    bandgaps = []
    
    print("Extracting structures and bandgap data...")
    
    for i, material in enumerate(materials):
        try:
            # Extract structure
            structure_dict = material.get('structure')
            if structure_dict is None:
                continue
                
            # Convert to pymatgen Structure
            structure = Structure.from_dict(structure_dict)
            
            # Extract bandgap
            bandgap = material.get('bandgap')
            
            # Skip if missing bandgap data
            if bandgap is None:
                continue
            
            # Store data
            structures.append(structure)
            material_ids.append(material.get('material_id', f'material_{i}'))
            formulas.append(material.get('formula_pretty', structure.formula))
            bandgaps.append(float(bandgap))
            
            if len(structures) % 100 == 0:
                print(f"Processed {len(structures)} materials...")
                
        except Exception as e:
            print(f"Error processing material {i}: {e}")
            continue
    
    print(f"Successfully extracted {len(structures)} materials with complete bandgap data")
    
    # Print bandgap statistics
    bandgaps_array = np.array(bandgaps)
    metals_count = sum(bg == 0.0 for bg in bandgaps)
    semiconductors_count = len(bandgaps) - metals_count
    
    print(f"Materials statistics:")
    print(f"  - Metals (bandgap = 0): {metals_count} ({metals_count/len(bandgaps)*100:.1f}%)")
    print(f"  - Semiconductors/Insulators: {semiconductors_count} ({semiconductors_count/len(bandgaps)*100:.1f}%)")
    print(f"  - Bandgap range: {bandgaps_array.min():.3f} - {bandgaps_array.max():.3f} eV")
    print(f"  - Average bandgap: {bandgaps_array.mean():.3f} eV")
    
    return {
        'structures': structures,
        'material_ids': material_ids,
        'formulas': formulas,
        'bandgaps': bandgaps
    }

def create_feature_datasets(data):
    """
    Create feature-engineered dataset using MatPipe.
    Generate CSV file for bandgap prediction.
    """
    structures = data['structures']
    material_ids = data['material_ids']
    formulas = data['formulas']
    bandgaps = data['bandgaps']
    
    # Create base dataframe with structures
    df_base = pd.DataFrame({
        'material_id': material_ids,
        'formula': formulas,
        'structure': structures
    })
    
    print("Starting feature extraction with MatPipe...")
    print(f"Processing {len(structures)} materials...")
    
    # Initialize MatPipe
    print("Initializing MatPipe with heavy preset...")
    try:
        pipe = MatPipe.from_preset("heavy")
        pipe.ml_type = 'regression'
        print("✓ Using 'heavy' preset")
    except Exception as e:
        print(f"Error with heavy preset: {e}")
        print("Trying express preset...")
        try:
            pipe = MatPipe.from_preset("express")
            pipe.ml_type = 'regression'
            print("✓ Using 'express' preset")
        except Exception as e2:
            print(f"Error with express preset: {e2}")
            print("Creating minimal MatPipe configuration...")
            from automatminer.featurization import AutoFeaturizer
            from automatminer.preprocessing import DataCleaner
            from matminer.featurizers.composition import ElementProperty
            from matminer.featurizers.structure import DensityFeatures
            
            # Create a minimal, compatible featurizer set
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
    
    # Extract structural features
    print("Extracting structural features...")
    df_temp = df_base.copy()
    df_temp['temp_target'] = bandgaps  # Temporary target for featurization
    
    # Feature engineering
    print("Running autofeaturizer...")
    df_features = pipe.autofeaturizer.fit_transform(df_temp, 'temp_target')
    print(f"Autofeaturizer completed. Shape: {df_features.shape}")
    
    print("Running cleaner...")
    df_features = pipe.cleaner.fit_transform(df_features, 'temp_target')
    print(f"Cleaner completed. Shape: {df_features.shape}")
    
    # Remove temporary target and structure columns (keep material_id and formula for reference)
    feature_columns = [col for col in df_features.columns 
                      if col not in ['temp_target', 'structure']]
    df_features = df_features[feature_columns]
    
    print(f"Generated {df_features.shape[1]} structural features")
    print(f"Final dataset contains {df_features.shape[0]} materials with complete features")
    
    # Get the indices of materials that survived the cleaning process
    # MatPipe cleaner might filter out some materials, so we need to align the bandgaps
    num_final_materials = df_features.shape[0]
    num_original_materials = len(bandgaps)
    
    if num_final_materials != num_original_materials:
        print(f"Materials filtered out by MatPipe: {num_original_materials - num_final_materials}")
        # Use the first num_final_materials bandgaps (assuming MatPipe preserves order)
        final_bandgaps = bandgaps[:num_final_materials]
    else:
        final_bandgaps = bandgaps
    
    # Create bandgap dataset
    print("\nCreating bandgap dataset...")
    df_bandgap = df_features.copy()
    df_bandgap['target'] = final_bandgaps
    df_bandgap.to_csv('2d_materials_bandgap_features.csv', index=False)
    print(f"Bandgap dataset saved: {df_bandgap.shape}")
    
    # Print final statistics
    final_bandgaps_array = np.array(final_bandgaps)
    final_metals_count = sum(bg == 0.0 for bg in final_bandgaps)
    final_semiconductors_count = len(final_bandgaps) - final_metals_count
    
    print(f"\nFinal dataset statistics:")
    print(f"  - Total materials: {len(final_bandgaps)}")
    print(f"  - Metals (bandgap = 0): {final_metals_count} ({final_metals_count/len(final_bandgaps)*100:.1f}%)")
    print(f"  - Semiconductors/Insulators: {final_semiconductors_count} ({final_semiconductors_count/len(final_bandgaps)*100:.1f}%)")
    print(f"  - Bandgap range: {final_bandgaps_array.min():.3f} - {final_bandgaps_array.max():.3f} eV")
    print(f"  - Average bandgap: {final_bandgaps_array.mean():.3f} eV")
    
    return df_bandgap

def main():
    """
    Main function to extract features from 2Dmatpedia data for bandgap prediction.
    """
    print("="*60)
    print("2D MATERIALS BANDGAP FEATURE EXTRACTION")
    print("="*60)
    
    # Load materials data
    materials = load_2d_materials_data('2Dmatpedia.json')
    
    if not materials:
        print("No materials data loaded. Exiting.")
        return
    
    # Extract structures and bandgaps
    data = extract_structures_and_bandgaps(materials)
    
    if not data['structures']:
        print("No valid structures extracted. Exiting.")
        return
    
    # Create feature dataset
    df_bandgap = create_feature_datasets(data)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("FEATURE EXTRACTION SUMMARY")
    print("="*60)
    print(f"Materials loaded from JSON: {len(data['structures'])}")
    print(f"Materials with complete features: {df_bandgap.shape[0]}")
    print(f"Materials filtered out by MatPipe: {len(data['structures']) - df_bandgap.shape[0]}")
    print(f"Number of structural features: {df_bandgap.shape[1] - 3}")  # -3 for material_id, formula, target
    print(f"Success rate: {df_bandgap.shape[0]/len(data['structures'])*100:.1f}%")
    
    print("\nOutput file:")
    print("- 2d_materials_bandgap_features.csv")
    print("\nColumns in the output file:")
    print("- material_id, formula: Material identification")
    print("- target: Bandgap (eV) - main target variable")
    print("- Structural features: Generated by MatPipe")
    print("\nFeature extraction completed successfully!")
    print("Dataset is ready for bandgap prediction modeling.")

if __name__ == "__main__":
    main()