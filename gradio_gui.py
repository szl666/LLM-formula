#!/usr/bin/env python3
"""
Advanced Gradio GUI Interface for LLM-Feynman
Supports CSV/Excel file upload and automated scientific formula discovery workflow
"""

import os
import sys

# Set environment variables for English interface (only analytics, avoid locale issues)
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

import gradio as gr
import pandas as pd
import numpy as np
import json
import traceback
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# Add current directory to path to import LLM-Feynman modules
current_dir = os.path.dirname(os.path.abspath(__file__))
llm_feynman_dir = os.path.join(current_dir, 'llm-feynman')

# Add both current dir and llm-feynman dir to path
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if llm_feynman_dir not in sys.path:
    sys.path.insert(0, llm_feynman_dir)

try:
    # Change to llm-feynman directory for imports
    original_cwd = os.getcwd()
    os.chdir(llm_feynman_dir)
    
    from main import LLMFeynman
    from core.symbolic_regression import Formula
    
    # Change back to original directory
    os.chdir(original_cwd)
    
    print("✅ LLM-Feynman modules loaded successfully")
except ImportError as e:
    print(f"❌ Warning: Unable to import LLM-Feynman module: {e}")
    print(f"   Python path: {sys.path}")
    print(f"   Current directory: {os.getcwd()}")
    print(f"   LLM-Feynman directory: {llm_feynman_dir}")
    import traceback
    traceback.print_exc()
    LLMFeynman = None
    Formula = None
except Exception as e:
    print(f"❌ Unexpected error loading LLM-Feynman: {e}")
    import traceback
    traceback.print_exc()
    LLMFeynman = None
    Formula = None

def safe_import_from_llm_feynman(module_path, class_names):
    """
    Safely import from llm-feynman directory
    
    Args:
        module_path: Module path relative to llm-feynman (e.g., 'models.openai_model')
        class_names: List of class names to import or a single class name string
    
    Returns:
        Imported class(es) or None if import fails
    """
    if isinstance(class_names, str):
        class_names = [class_names]
    
    try:
        # Ensure llm-feynman directory is in path
        if llm_feynman_dir not in sys.path:
            sys.path.insert(0, llm_feynman_dir)
        
        # Import module
        module = __import__(module_path, fromlist=class_names)
        
        # Get classes
        results = [getattr(module, name) for name in class_names]
        return results[0] if len(results) == 1 else results
    except Exception as e:
        print(f"Error importing {class_names} from {module_path}: {e}")
        return None

class LLMFeynmanGUI:
    """LLM-Feynman GUI Main Class"""
    
    def __init__(self):
        self.llm_feynman = None
        self.current_data = None
        self.results = None
        self.feature_columns = []
        self.target_column = None
        self.template_llm = None  # LLM instance for template generation
        
    def load_data(self, file_obj) -> Tuple[str, str, List[str]]:
        """Load CSV or Excel file"""
        try:
            if file_obj is None:
                return "", "❌ Please upload a data file", []
            
            # Get file path
            if hasattr(file_obj, 'name'):
                file_path = file_obj.name
            elif isinstance(file_obj, str):
                file_path = file_obj
            else:
                return "", "❌ Invalid file object", []
            
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Read data
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                return "", "❌ Unsupported file format. Please upload CSV or Excel file", []
            
            self.current_data = df
            
            # Generate data preview
            preview_html = self._generate_data_preview(df)
            
            # Return column selection options
            columns = list(df.columns)
            
            success_msg = f"✅ Data loaded successfully! Shape: {df.shape}"
            
            return preview_html, success_msg, columns
            
        except Exception as e:
            error_msg = f"❌ Failed to load data: {str(e)}"
            return "", error_msg, []
    
    def _generate_data_preview(self, df: pd.DataFrame) -> str:
        """Generate data preview HTML"""
        try:
            # Basic statistics
            info_html = f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h3 style="margin: 0; color: white;">📊 Data Overview</h3>
                <p style="margin: 10px 0; color: white;">
                    <strong>Shape:</strong> {df.shape[0]} rows × {df.shape[1]} columns<br>
                    <strong>Columns:</strong> {', '.join(df.columns.tolist())}<br>
                    <strong>Data Types:</strong> {df.dtypes.value_counts().to_dict()}
                </p>
            </div>
            """
            
            # Data preview table
            preview_df = df.head(10)
            table_html = preview_df.to_html(classes='preview-table', table_id='data-preview')
            
            # Add styles
            styled_html = f"""
            <style>
                .preview-table {{
                    border-collapse: collapse;
                    width: 100%;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .preview-table th {{
                    background: linear-gradient(135deg, #4CAF50, #45a049);
                    color: white;
                    font-weight: bold;
                    padding: 12px;
                    text-align: center;
                }}
                .preview-table td {{
                    padding: 10px;
                    text-align: center;
                    border-bottom: 1px solid #ddd;
                }}
                .preview-table tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .preview-table tr:hover {{
                    background-color: #f5f5f5;
                }}
            </style>
            {info_html}
            <h4>📋 Data Preview (First 10 rows)</h4>
            {table_html}
            """
            
            return styled_html
            
        except Exception as e:
            return f"<p style='color: red;'>Preview generation failed: {str(e)}</p>"
    
    def update_feature_columns(self, target_column: str) -> Tuple[str, str, str]:
        """Update feature column list"""
        try:
            if self.current_data is None or not target_column:
                return "❌ Please load data and select target column first", "{}", "{}"
            
            self.target_column = target_column
            self.feature_columns = [col for col in self.current_data.columns if col != target_column]
            
            msg = f"✅ Target column: {target_column}\n🔧 Feature columns: {', '.join(self.feature_columns)}"
            
            # Auto-generate templates
            meaning_template = self.generate_feature_meaning_template(self.feature_columns)
            dimension_template = self.generate_feature_dimension_template(self.feature_columns)
            
            return msg, meaning_template, dimension_template
            
        except Exception as e:
            return f"❌ Update failed: {str(e)}", "{}", "{}"
    
    def generate_feature_meaning_template(self, feature_columns: List[str]) -> str:
        """Generate feature meaning template"""
        if not feature_columns:
            return "{}"
        
        template = {}
        for col in feature_columns:
            template[col] = f"Physical meaning of {col}"
        
        return json.dumps(template, indent=2, ensure_ascii=False)
    
    def generate_feature_dimension_template(self, feature_columns: List[str]) -> str:
        """Generate feature dimension template"""
        if not feature_columns:
            return "{}"
        
        template = {}
        for col in feature_columns:
            template[col] = "unit"  # User needs to replace with actual unit
        
        return json.dumps(template, indent=2, ensure_ascii=False)
    
    def llm_generate_meaning_template(self, target_meaning: str, target_dimension: str) -> str:
        """Auto-generate feature physical meaning template using LLM"""
        if not self.feature_columns or not target_meaning:
            return "❌ Please select target column and fill in target variable's physical meaning first"
        
        try:
            # Import prompt templates using safe import
            FeatureEngineeringPrompts = safe_import_from_llm_feynman(
                'templates.prompt_templates', 
                'FeatureEngineeringPrompts'
            )
            
            if FeatureEngineeringPrompts is None:
                return "❌ Failed to import prompt templates"
            
            # Initialize temporary LLM model for template generation
            if not hasattr(self, 'template_llm') or self.template_llm is None:
                return "❌ Please configure LLM model first"
            
            # Use LLM to generate feature suggestions, passing specific data structure information
            prompt = FeatureEngineeringPrompts.feature_recommendation_prompt(
                target_property=target_meaning,
                feature_columns=self.feature_columns,
                target_column=self.target_column
            )
            
            response = self.template_llm.generate(prompt)
            
            # Parse response
            meanings = self._parse_llm_meanings_response(response)
            
            return json.dumps(meanings, indent=2, ensure_ascii=False)
            
        except Exception as e:
            return f"❌ LLM generation failed: {str(e)}"
    
    def llm_generate_dimension_template(self, target_meaning: str, target_dimension: str) -> str:
        """Auto-generate feature dimension template using LLM"""
        if not self.feature_columns or not target_meaning:
            return "❌ Please select target column and fill in target variable's physical meaning first"
        
        try:
            # Import prompt templates using safe import
            FeatureEngineeringPrompts = safe_import_from_llm_feynman(
                'templates.prompt_templates', 
                'FeatureEngineeringPrompts'
            )
            
            if FeatureEngineeringPrompts is None:
                return "❌ Failed to import prompt templates"
            
            # Initialize temporary LLM model for template generation
            if not hasattr(self, 'template_llm') or self.template_llm is None:
                return "❌ Please configure LLM model first"
            
            # Use LLM to generate feature suggestions, passing specific data structure information
            prompt = FeatureEngineeringPrompts.feature_recommendation_prompt(
                target_property=target_meaning,
                feature_columns=self.feature_columns,
                target_column=self.target_column
            )
            
            response = self.template_llm.generate(prompt)
            
            # Parse response
            dimensions = self._parse_llm_dimensions_response(response)
            
            return json.dumps(dimensions, indent=2, ensure_ascii=False)
            
        except Exception as e:
            return f"❌ LLM generation failed: {str(e)}"
    
    def _parse_llm_meanings_response(self, response: str) -> Dict[str, str]:
        """Parse physical meanings from LLM response"""
        lines = response.strip().split('\n')
        meanings = {}
        
        try:
            if len(lines) >= 2:
                # First line is feature name, second line is meaning
                descriptors = [desc.strip() for desc in lines[0].split(',')]
                meanings_list = [meaning.strip() for meaning in lines[1].split(',')]
                
                # Match existing feature column names
                for i, col in enumerate(self.feature_columns):
                    if i < len(meanings_list):
                        meanings[col] = meanings_list[i]
                    elif len(descriptors) > 0:
                        # Use generic meaning
                        meanings[col] = f"Physical property related to {col}"
                    else:
                        meanings[col] = f"Physical meaning of {col}"
            else:
                # Fall back to simple template
                for col in self.feature_columns:
                    meanings[col] = f"Physical meaning of {col}"
                    
        except Exception:
            # Use simple template when parsing fails
            for col in self.feature_columns:
                meanings[col] = f"Physical meaning of {col}"
        
        return meanings
    
    def _parse_llm_dimensions_response(self, response: str) -> Dict[str, str]:
        """Parse dimensions from LLM response"""
        lines = response.strip().split('\n')
        dimensions = {}
        
        try:
            if len(lines) >= 3:
                # Third line is dimensions
                units_list = [unit.strip() for unit in lines[2].split(',')]
                
                # Match existing feature column names
                for i, col in enumerate(self.feature_columns):
                    if i < len(units_list):
                        dimensions[col] = units_list[i]
                    else:
                        dimensions[col] = "unit"
            else:
                # Fall back to simple template
                for col in self.feature_columns:
                    dimensions[col] = "unit"
                    
        except Exception:
            # Use simple template when parsing fails
            for col in self.feature_columns:
                dimensions[col] = "unit"
        
        return dimensions
    
    def setup_template_llm(self, model_type: str, model_name: str, api_key: str, base_url: str = "") -> str:
        """Setup LLM model for template generation"""
        try:
            if model_type == "o3" or model_type == "openai":
                OpenAIModel = safe_import_from_llm_feynman('models.openai_model', 'OpenAIModel')
                if OpenAIModel is None:
                    return "❌ Failed to import OpenAIModel"
                
                if model_type == "o3":
                    self.template_llm = OpenAIModel(
                        model_name=model_name,
                        api_key=api_key,
                        base_url=base_url
                    )
                else:  # openai
                    self.template_llm = OpenAIModel(
                        model_name=model_name,
                        api_key=api_key
                    )
            else:  # huggingface
                HuggingFaceModel = safe_import_from_llm_feynman('models.hf_model', 'HuggingFaceModel')
                if HuggingFaceModel is None:
                    return "❌ Failed to import HuggingFaceModel"
                self.template_llm = HuggingFaceModel(model_name=model_name)
            
            return "✅ LLM template generator configured"
            
        except Exception as e:
            return f"❌ LLM configuration failed: {str(e)}"
    
    def run_discovery(self, 
                     target_column: str,
                     target_meaning: str,
                     target_dimension: str,
                     context_description: str,
                     feature_meanings_json: str,
                     feature_dimensions_json: str,
                     model_type: str,
                     model_name: str,
                     api_key: str,
                     base_url: str,
                     n_initial_formulas: int,
                     max_iterations: int,
                     formulas_per_iteration: int,
                     enable_dimensional_analysis: bool,
                     include_interpretation: bool,
                     progress=gr.Progress()) -> Tuple[str, str, str]:
        """Run formula discovery workflow"""
        
        try:
            if self.current_data is None:
                return "❌ Please load data first", "", ""
            
            if not target_column:
                return "❌ Please select target column", "", ""
            
            progress(0.1, desc="Initializing LLM model...")
            
            # Initialize LLM-Feynman
            if LLMFeynman is None:
                return "❌ LLM-Feynman module not loaded correctly", "", ""
            
            # Parse feature meanings and dimensions
            try:
                feature_meanings = json.loads(feature_meanings_json) if feature_meanings_json.strip() else {}
                feature_dimensions = json.loads(feature_dimensions_json) if feature_dimensions_json.strip() else {}
            except json.JSONDecodeError as e:
                return f"❌ JSON format error: {str(e)}", "", ""
            
            # Prepare data
            X = self.current_data.drop(columns=[target_column])
            y = self.current_data[target_column]
            
            progress(0.2, desc="Initializing LLM-Feynman...")
            
            # Initialize model based on model type
            if model_type == "o3":
                self.llm_feynman = LLMFeynman(
                    model_type="openai",  # O3 uses OpenAI-compatible API
                    model_name=model_name,
                    api_key=api_key,
                    base_url=base_url
                )
            elif model_type == "openai":
                self.llm_feynman = LLMFeynman(
                    model_type="openai",
                    model_name=model_name,
                    api_key=api_key
                )
            else:  # huggingface
                self.llm_feynman = LLMFeynman(
                    model_type="huggingface",
                    model_name=model_name
                )
            
            progress(0.3, desc="Starting formula discovery...")
            
            # Run discovery workflow
            formulas = self.llm_feynman.discover_formulas(
                X=X,
                y=y,
                feature_meanings=feature_meanings or None,
                feature_dimensions=feature_dimensions or None,
                target_meaning=target_meaning or None,
                target_dimension=target_dimension or None,
                n_initial_formulas=n_initial_formulas,
                max_iterations=max_iterations,
                formulas_per_iteration=formulas_per_iteration,
                enable_dimensional_analysis=enable_dimensional_analysis,
                include_interpretation=include_interpretation,
                verbose_loss=True
            )
            
            progress(0.9, desc="Generating results...")
            
            self.results = formulas
            
            # Generate results display
            results_html = self._generate_results_html(formulas)
            download_json = self._generate_download_data(formulas)
            
            progress(1.0, desc="Complete!")
            
            success_msg = f"✅ Discovered {len(formulas)} formulas!"
            
            return success_msg, results_html, download_json
            
        except Exception as e:
            error_msg = f"❌ Discovery workflow failed: {str(e)}\n\nDetailed error:\n{traceback.format_exc()}"
            return error_msg, "", ""
    
    def _generate_results_html(self, formulas: List[Formula]) -> str:
        """Generate results display HTML"""
        try:
            if not formulas:
                return "<p>No formulas discovered</p>"
            
            # Sort formulas (by R² score)
            sorted_formulas = sorted(formulas, key=lambda f: f.r2, reverse=True)
            
            html_parts = []
            
            # Title
            html_parts.append("""
            <div style="background: linear-gradient(135deg, #FF6B6B, #4ECDC4); 
                        color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h2 style="margin: 0; color: white;">🔬 Discovered Scientific Formulas</h2>
                <p style="margin: 10px 0; color: white;">
                    Mathematical formulas automatically discovered using LLM-Feynman algorithm
                </p>
            </div>
            """)
            
            # Formula list
            for i, formula in enumerate(sorted_formulas[:10]):  # Display top 10 best formulas
                
                # Calculate performance level
                if formula.r2 >= 0.95:
                    performance_color = "#4CAF50"
                    performance_text = "Excellent"
                elif formula.r2 >= 0.90:
                    performance_color = "#FF9800"
                    performance_text = "Good"
                elif formula.r2 >= 0.80:
                    performance_color = "#FF5722"
                    performance_text = "Fair"
                else:
                    performance_color = "#9E9E9E"
                    performance_text = "Poor"
                
                formula_html = f"""
                <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0; 
                           background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <h4 style="margin: 0; color: #333;">🧮 Formula #{i+1}</h4>
                        <span style="background: {performance_color}; color: white; padding: 4px 8px; 
                                   border-radius: 4px; font-size: 12px;">{performance_text}</span>
                    </div>
                    
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 6px; margin: 10px 0; 
                               border-left: 4px solid #007bff;">
                        <code style="font-size: 16px; font-weight: bold; color: #d63384;">
                            {formula.expression}
                        </code>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); 
                               gap: 10px; margin: 10px 0;">
                        <div style="text-align: center; padding: 8px; background: #e3f2fd; border-radius: 4px;">
                            <strong>R² Score</strong><br>
                            <span style="font-size: 18px; color: #1976d2;">{formula.r2:.4f}</span>
                        </div>
                        <div style="text-align: center; padding: 8px; background: #f3e5f5; border-radius: 4px;">
                            <strong>MAE</strong><br>
                            <span style="font-size: 18px; color: #7b1fa2;">{formula.mae:.4f}</span>
                        </div>
                        <div style="text-align: center; padding: 8px; background: #e8f5e8; border-radius: 4px;">
                            <strong>Complexity</strong><br>
                            <span style="font-size: 18px; color: #388e3c;">{formula.complexity}</span>
                        </div>
                        <div style="text-align: center; padding: 8px; background: #fff3e0; border-radius: 4px;">
                            <strong>Interpretability</strong><br>
                            <span style="font-size: 18px; color: #f57c00;">{formula.interpretability:.3f}</span>
                        </div>
                    </div>
                </div>
                """
                html_parts.append(formula_html)
            
            # Summary statistics
            best_formula = sorted_formulas[0]
            summary_html = f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 20px; border-radius: 10px; margin-top: 20px;">
                <h3 style="margin: 0; color: white;">📈 Discovery Summary</h3>
                <div style="margin: 15px 0;">
                    <p style="color: white; margin: 5px 0;">
                        <strong>Best Formula:</strong> {best_formula.expression}
                    </p>
                    <p style="color: white; margin: 5px 0;">
                        <strong>Highest R² Score:</strong> {best_formula.r2:.4f}
                    </p>
                    <p style="color: white; margin: 5px 0;">
                        <strong>Total Formulas Discovered:</strong> {len(formulas)}
                    </p>
                </div>
            </div>
            """
            html_parts.append(summary_html)
            
            return "".join(html_parts)
            
        except Exception as e:
            return f"<p style='color: red;'>Result generation failed: {str(e)}</p>"
    
    def _generate_download_data(self, formulas: List[Formula]) -> str:
        """Generate download data"""
        try:
            if not formulas:
                return ""
            
            # Prepare export data
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "total_formulas": len(formulas),
                "formulas": []
            }
            
            for i, formula in enumerate(formulas):
                formula_data = {
                    "id": i + 1,
                    "expression": formula.expression,
                    "r2_score": float(formula.r2),
                    "mae": float(formula.mae),
                    "complexity": int(formula.complexity),
                    "interpretability": float(formula.interpretability),
                    "loss": float(formula.loss)
                }
                export_data["formulas"].append(formula_data)
            
            return json.dumps(export_data, indent=2, ensure_ascii=False)
            
        except Exception as e:
            return f'{{"error": "Export failed: {str(e)}"}}'

def create_gui():
    """Create Gradio interface"""
    
    gui = LLMFeynmanGUI()
    
    # Define advanced CSS styles
    css = """
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        min-height: 100vh;
    }
    
    .step-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        border: 1px solid rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
    }
    
    .gradio-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    .gradio-dropdown {
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 12px !important;
        background: white !important;
        transition: all 0.3s ease !important;
    }
    
    /* Override Chinese file upload text */
    .upload-text::before {
        content: 'Drop file here or click to browse' !important;
    }
    
    .file-preview {
        font-size: 14px !important;
    }
    """
    
    with gr.Blocks(css=css, title="LLM-Feynman Scientific Formula Discovery System") as interface:
        
        # Title and introduction
        gr.HTML("""
        <div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; border-radius: 16px; margin-bottom: 30px;">
            <h1 style="font-size: 3em; margin-bottom: 16px; font-weight: 700;">🔬 LLM-Feynman</h1>
            <h2 style="font-size: 1.5em; margin-bottom: 12px; font-weight: 400; opacity: 0.9;">Scientific Formula Discovery System</h2>
            <p style="font-size: 1.1em; margin-bottom: 8px; opacity: 0.8;">Automated scientific formula discovery platform based on Large Language Models</p>
        </div>
        """)
        
        # Step 1: Data Upload
        with gr.Group(elem_classes=["step-card"]):
            gr.HTML("""
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; 
                           width: 40px; height: 40px; border-radius: 50%; display: flex; 
                           align-items: center; justify-content: center; margin-right: 15px; 
                           font-weight: bold; font-size: 18px;">1</div>
                <h2 style="margin: 0; color: #374151; font-size: 1.5em;">📂 Data Upload</h2>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    gr.HTML("""
                    <div style="background: #e8f4f8; padding: 10px; border-radius: 8px; margin-bottom: 10px; text-align: center;">
                        <p style="margin: 0; color: #1976d2; font-size: 14px;">
                            📎 <strong>Drop your file here or click below to browse</strong>
                        </p>
                    </div>
                    """)
                    file_input = gr.File(
                        label="📁 Upload CSV or Excel file",
                        file_types=[".csv", ".xlsx", ".xls"],
                        type="filepath"
                    )
                    load_btn = gr.Button("🔄 Load Data", variant="primary", size="lg")
                    
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div style="background: #f8fafc; padding: 20px; border-radius: 12px; border-left: 4px solid #667eea;">
                        <h4 style="color: #374151; margin-top: 0;">📋 Data Requirements</h4>
                        <ul style="color: #6b7280; line-height: 1.6;">
                            <li>✅ CSV and Excel formats supported</li>
                            <li>📊 Data should contain feature columns and target column</li>
                            <li>📈 Recommended data size > 20 rows</li>
                            <li>🔢 Primarily numerical data</li>
                        </ul>
                    </div>
                    """)
        
        # Data preview and status
        data_status = gr.Textbox(label="Status", interactive=False)
        data_preview = gr.HTML(label="Data Preview")
        
        # Step 2: Column Selection
        with gr.Group(elem_classes=["step-card"]):
            gr.HTML("""
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; 
                           width: 40px; height: 40px; border-radius: 50%; display: flex; 
                           align-items: center; justify-content: center; margin-right: 15px; 
                           font-weight: bold; font-size: 18px;">2</div>
                <h2 style="margin: 0; color: #374151; font-size: 1.5em;">🎯 Target Column Selection</h2>
            </div>
            """)
            
            with gr.Row():
                target_dropdown = gr.Dropdown(
                    label="🎯 Select target column (variable to predict)",
                    choices=[],
                    interactive=False,
                    info="Please load data file first"
                )
                update_features_btn = gr.Button("✅ Confirm Selection", variant="secondary", size="lg")
        
        feature_status = gr.Textbox(label="Feature Column Status", interactive=False)
        
        # Step 3: Physical Meaning Configuration
        with gr.Group(elem_classes=["step-card"]):
            gr.HTML("""
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; 
                           width: 40px; height: 40px; border-radius: 50%; display: flex; 
                           align-items: center; justify-content: center; margin-right: 15px; 
                           font-weight: bold; font-size: 18px;">3</div>
                <h2 style="margin: 0; color: #374151; font-size: 1.5em;">🧪 Physical Meaning and Dimension Configuration</h2>
            </div>
            """)
            
            # Feature-target relationship explanation
            gr.HTML("""
            <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #4CAF50;">
                <h4 style="color: #2E7D32; margin-top: 0;">📊 Data Relationship Explanation</h4>
                <p style="color: #1B5E20; line-height: 1.6; margin-bottom: 8px;">
                    <strong>Feature Columns (Input Variables):</strong> Input data columns used to predict the target variable, serving as independent variables in the formula<br>
                    <strong>Target Column (Output Variable):</strong> Output data column to be predicted, serving as the dependent variable in the formula
                </p>
                <p style="color: #2E7D32; font-size: 14px; margin: 0;">
                    💡 Tip: Accurately describing the physical meaning and dimensions of each variable helps the LLM generate formulas that better conform to physical laws
                </p>
            </div>
            """)
        
        with gr.Row():
            with gr.Column():
                target_meaning = gr.Textbox(
                    label="Physical meaning of target variable",
                    placeholder="e.g.: voltage, temperature, reaction rate, etc.",
                    lines=2
                )
                target_dimension = gr.Textbox(
                    label="Dimension of target variable",
                    placeholder="e.g.: V, K, mol/s, etc.",
                    lines=1
                )
                
            with gr.Column():
                context_description = gr.Textbox(
                    label="Research Background Description",
                    placeholder="Describe your research objectives and expected physical laws to discover",
                    lines=4
                )
        
        with gr.Row():
            with gr.Column():
                feature_meanings = gr.Code(
                    label="Physical meanings of feature variables (JSON format)",
                    language="json",
                    value='{"feature1": "physical meaning 1", "feature2": "physical meaning 2"}',
                    lines=8
                )
                
            with gr.Column():
                feature_dimensions = gr.Code(
                    label="Dimensions of feature variables (JSON format)",
                    language="json",
                    value='{"feature1": "unit 1", "feature2": "unit 2"}',
                    lines=8
                )
        
        with gr.Row():
            generate_meaning_btn = gr.Button("📝 Generate Meaning Template", variant="secondary")
            generate_dimension_btn = gr.Button("📐 Generate Dimension Template", variant="secondary")
            
        with gr.Row():
            ai_generate_meaning_btn = gr.Button("🤖 AI Generate Meanings", variant="primary")
            ai_generate_dimension_btn = gr.Button("🤖 AI Generate Dimensions", variant="primary")
        
        # Step 4: Model Configuration
        with gr.Group(elem_classes=["step-card"]):
            gr.HTML("""
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; 
                           width: 40px; height: 40px; border-radius: 50%; display: flex; 
                           align-items: center; justify-content: center; margin-right: 15px; 
                           font-weight: bold; font-size: 18px;">4</div>
                <h2 style="margin: 0; color: #374151; font-size: 1.5em;">⚙️ Model and Algorithm Configuration</h2>
            </div>
            """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🤖 LLM Model Settings")
                model_type = gr.Dropdown(
                    label="🤖 Model Type",
                    choices=["huggingface", "openai", "o3"],
                    value="o3",
                    interactive=True
                )
                
                # Show different configurations based on model type
                with gr.Group(visible=False) as hf_config:
                    model_name_hf = gr.Textbox(
                        label="📝 HuggingFace Model Name",
                        value="meta-llama/Llama-2-7b-chat-hf",
                        placeholder="e.g.: meta-llama/Llama-2-7b-chat-hf"
                    )
                    
                with gr.Group(visible=False) as openai_config:
                    model_name_openai = gr.Dropdown(
                        label="📝 OpenAI Model",
                        choices=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                        value="gpt-4",
                        interactive=True
                    )
                    openai_api_key = gr.Textbox(
                        label="🔑 OpenAI API Key",
                        type="password",
                        placeholder="Enter your OpenAI API key"
                    )
                
                # O3 API configuration (displayed by default)
                with gr.Group(visible=True) as o3_config:
                    model_name_o3 = gr.Dropdown(
                        label="📝 O3 Model",
                        choices=[
                            "claude-sonnet-4-20250514",
                            "gpt-4o",
                            "gpt-4o-mini",
                            "o1-preview",
                            "o1-mini"
                        ],
                        value="claude-sonnet-4-20250514",
                        interactive=True
                    )
                    api_key = gr.Textbox(
                        label="🔑 O3 API Key",
                        value="sk-wqy8ahJLPeT4cTvmCbDcD1B9AeBf4fAcA610649b882fE5Fd",
                        type="password"
                    )
                    base_url = gr.Textbox(
                        label="🌐 Base URL",
                        value="https://api.o3.fan/v1"
                    )
                
            with gr.Column():
                gr.Markdown("### 🔧 Algorithm Parameters")
                n_initial_formulas = gr.Slider(
                    label="Number of initial formulas",
                    minimum=50,
                    maximum=500,
                    value=100,
                    step=50
                )
                max_iterations = gr.Slider(
                    label="Maximum iterations",
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=10
                )
                formulas_per_iteration = gr.Slider(
                    label="Formulas per iteration",
                    minimum=10,
                    maximum=50,
                    value=20,
                    step=5
                )
        
        with gr.Row():
            enable_dimensional_analysis = gr.Checkbox(
                label="Enable dimensional analysis",
                value=True
            )
            include_interpretation = gr.Checkbox(
                label="Include formula interpretation",
                value=True
            )
        
        # Step 5: Run Discovery
        with gr.Group(elem_classes=["step-card"]):
            gr.HTML("""
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; 
                           width: 40px; height: 40px; border-radius: 50%; display: flex; 
                           align-items: center; justify-content: center; margin-right: 15px; 
                           font-weight: bold; font-size: 18px;">5</div>
                <h2 style="margin: 0; color: #374151; font-size: 1.5em;">🚀 Run Formula Discovery</h2>
            </div>
            """)
            
            with gr.Row():
                run_btn = gr.Button("🔍 Start Discovery", variant="primary", size="lg")
            
        # Results display
        gr.Markdown("## 📊 Discovery Results")
        
        run_status = gr.Textbox(label="Run Status", interactive=False)
        results_display = gr.HTML(label="Discovered Formulas")
        
        # Results download
        with gr.Row():
            with gr.Column(scale=3):
                download_data = gr.Code(
                    label="Download Data (JSON format)",
                    language="json",
                    lines=10
                )
            with gr.Column(scale=1):
                gr.Markdown("### 💾 Export Options")
                export_format = gr.Radio(
                    choices=["JSON", "CSV"],
                    value="JSON",
                    label="Export Format",
                    interactive=True
                )
                download_file = gr.File(
                    label="Download File",
                    visible=True
                )
                create_download_btn = gr.Button("📁 Generate Download File", variant="secondary")
                download_status = gr.Textbox(label="Download Status", interactive=False)
        
        # Event handler functions
        def load_data_event(file_obj):
            """File loading event handler"""
            if file_obj is None:
                return "", "❌ Please upload a data file", gr.update(choices=[], interactive=False, info="Please upload data file first")
            
            try:
                preview_html, success_msg, choices = gui.load_data(file_obj)
                
                # Ensure choices is a list
                if not isinstance(choices, list):
                    choices = []
                
                # Use gr.update to update dropdown
                if len(choices) > 0:
                    dropdown_update = gr.update(
                        choices=choices, 
                        value=None, 
                        interactive=True,
                        info=f"Loaded {len(choices)} columns, please select target column"
                    )
                else:
                    dropdown_update = gr.update(
                        choices=[], 
                        value=None, 
                        interactive=False,
                        info="Data loading failed, please check file format"
                    )
                
                return preview_html, success_msg, dropdown_update
                
            except Exception as e:
                return "", f"❌ Loading failed: {str(e)}", gr.update(choices=[], interactive=False, info="Loading failed")
        
        def target_change_event(target_col):
            """Target column selection change event"""
            if not target_col:
                return "❌ Please select target column first", "{}", "{}"
            try:
                msg, meaning_template, dimension_template = gui.update_feature_columns(target_col)
                print(f"DEBUG: Target column update complete, feature columns: {gui.feature_columns}")
                return msg, meaning_template, dimension_template
            except Exception as e:
                print(f"DEBUG: Target column update error: {e}")
                return f"❌ Update failed: {str(e)}", "{}", "{}"
        
        def generate_meaning_wrapper():
            """Generate meaning template wrapper function"""
            try:
                if not gui.feature_columns:
                    return "❌ Please select target column first"
                result = gui.generate_feature_meaning_template(gui.feature_columns)
                return result
            except Exception as e:
                return f"❌ Generation failed: {str(e)}"
        
        def generate_dimension_wrapper():
            """Generate dimension template wrapper function"""
            try:
                if not gui.feature_columns:
                    return "❌ Please select target column first"
                result = gui.generate_feature_dimension_template(gui.feature_columns)
                return result
            except Exception as e:
                return f"❌ Generation failed: {str(e)}"
        
        def ai_generate_meaning_wrapper(target_meaning, target_dimension, model_type, 
                                       model_name_hf, model_name_openai, model_name_o3, 
                                       openai_api_key, api_key, base_url):
            """AI smart generation meaning template wrapper function"""
            try:
                # Get current model configuration
                model_name, current_api_key, current_base_url = get_current_model_config(
                    model_type, model_name_hf, model_name_openai, model_name_o3, 
                    openai_api_key, api_key, base_url
                )
                
                # Setup template LLM
                setup_status = gui.setup_template_llm(model_type, model_name, current_api_key, current_base_url)
                if "❌" in setup_status:
                    return setup_status
                
                # Generate template
                result = gui.llm_generate_meaning_template(target_meaning, target_dimension)
                return result
            except Exception as e:
                return f"❌ AI generation failed: {str(e)}"
        
        def ai_generate_dimension_wrapper(target_meaning, target_dimension, model_type, 
                                         model_name_hf, model_name_openai, model_name_o3, 
                                         openai_api_key, api_key, base_url):
            """AI smart generation dimension template wrapper function"""
            try:
                # Get current model configuration
                model_name, current_api_key, current_base_url = get_current_model_config(
                    model_type, model_name_hf, model_name_openai, model_name_o3, 
                    openai_api_key, api_key, base_url
                )
                
                # Setup template LLM
                setup_status = gui.setup_template_llm(model_type, model_name, current_api_key, current_base_url)
                if "❌" in setup_status:
                    return setup_status
                
                # Generate template
                result = gui.llm_generate_dimension_template(target_meaning, target_dimension)
                return result
            except Exception as e:
                return f"❌ AI generation failed: {str(e)}"
        
        def model_config_wrapper(model_type):
            """Model configuration switch wrapper function"""
            if model_type == "huggingface":
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            elif model_type == "openai":
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            else:  # o3
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
        
        def get_current_model_config(model_type, model_name_hf, model_name_openai, model_name_o3, openai_api_key, api_key, base_url):
            """Get current selected model configuration"""
            if model_type == "huggingface":
                return model_name_hf, "", ""
            elif model_type == "openai":
                return model_name_openai, openai_api_key, ""
            else:  # o3
                return model_name_o3, api_key, base_url
        
        def run_discovery_wrapper(target_dropdown, target_meaning, target_dimension, context_description,
                                feature_meanings, feature_dimensions, model_type, 
                                model_name_hf, model_name_openai, model_name_o3, openai_api_key, api_key, base_url,
                                n_initial_formulas, max_iterations, formulas_per_iteration,
                                enable_dimensional_analysis, include_interpretation, progress=gr.Progress()):
            """Run discovery wrapper function"""
            # Get current model configuration
            model_name, current_api_key, current_base_url = get_current_model_config(
                model_type, model_name_hf, model_name_openai, model_name_o3, 
                openai_api_key, api_key, base_url
            )
            
            return gui.run_discovery(
                target_dropdown, target_meaning, target_dimension, context_description,
                feature_meanings, feature_dimensions, model_type, model_name, 
                current_api_key, current_base_url, n_initial_formulas, max_iterations, 
                formulas_per_iteration, enable_dimensional_analysis, include_interpretation, progress
            )
        
        # Event bindings
        file_input.upload(
            fn=load_data_event,
            inputs=[file_input],
            outputs=[data_preview, data_status, target_dropdown]
        )
        
        load_btn.click(
            fn=load_data_event,
            inputs=[file_input],
            outputs=[data_preview, data_status, target_dropdown]
        )
        
        target_dropdown.change(
            fn=target_change_event,
            inputs=[target_dropdown],
            outputs=[feature_status, feature_meanings, feature_dimensions]
        )
        
        update_features_btn.click(
            fn=target_change_event,
            inputs=[target_dropdown],
            outputs=[feature_status, feature_meanings, feature_dimensions]
        )
        
        model_type.change(
            fn=model_config_wrapper,
            inputs=[model_type],
            outputs=[hf_config, openai_config, o3_config]
        )
        
        generate_meaning_btn.click(
            fn=generate_meaning_wrapper,
            outputs=[feature_meanings]
        )
        
        generate_dimension_btn.click(
            fn=generate_dimension_wrapper,
            outputs=[feature_dimensions]
        )
        
        ai_generate_meaning_btn.click(
            fn=ai_generate_meaning_wrapper,
            inputs=[
                target_meaning, target_dimension, model_type,
                model_name_hf, model_name_openai, model_name_o3, 
                openai_api_key, api_key, base_url
            ],
            outputs=[feature_meanings]
        )
        
        ai_generate_dimension_btn.click(
            fn=ai_generate_dimension_wrapper,
            inputs=[
                target_meaning, target_dimension, model_type,
                model_name_hf, model_name_openai, model_name_o3, 
                openai_api_key, api_key, base_url
            ],
            outputs=[feature_dimensions]
        )
        
        run_btn.click(
            fn=run_discovery_wrapper,
            inputs=[
                target_dropdown, target_meaning, target_dimension, context_description,
                feature_meanings, feature_dimensions, model_type,
                model_name_hf, model_name_openai, model_name_o3, openai_api_key, api_key, base_url,
                n_initial_formulas, max_iterations, formulas_per_iteration,
                enable_dimensional_analysis, include_interpretation
            ],
            outputs=[run_status, results_display, download_data]
        )
        
        # Add usage instructions
        with gr.Accordion("📖 Usage Instructions", open=False):
            gr.Markdown("""
            ### 📋 Workflow
            
            1. **Data Upload**: Upload a CSV or Excel file containing numerical data
            2. **Column Selection**: Select the target column to predict, other columns automatically become feature columns
            3. **Physical Configuration**: Input physical meanings and dimensions of variables to improve formula interpretability
            4. **Model Configuration**: Select LLM model and algorithm parameters
            5. **Run Discovery**: Click the start button and wait for the algorithm to automatically discover mathematical formulas
            
            ### 🎯 Best Practices
            
            - **Data Quality**: Ensure good data quality with no excessive missing values
            - **Physical Meaning**: Detailed descriptions of variable physical meanings help discover better formulas
            - **Dimensional Consistency**: Correct dimensional information helps the algorithm perform dimensional analysis
            - **Parameter Tuning**: Adjust algorithm parameters based on data complexity
            
            ### ⚠️ Notes
            
            - Running time depends on data complexity and parameter settings
            - It is recommended to test with smaller parameters first
            - Some models may require GPU support
            - Results can be downloaded in JSON format for further analysis
            """)
    
    return interface

if __name__ == "__main__":
    # Create and launch GUI
    interface = create_gui()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )