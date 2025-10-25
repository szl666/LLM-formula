
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from models.base_model import BaseLLMModel
from models.hf_model import HuggingFaceModel
from models.openai_model import OpenAIModel
from core.data_preprocessing import DataPreprocessor
from core.feature_engineering import FeatureEngineer
from core.symbolic_regression import SymbolicRegressor, Formula
from core.formula_interpretation import FormulaInterpreter
from utils.plotting import plot_results
from utils.metrics import evaluate_formulas

class LLMFeynman:
    """
    Main LLM-Feynman class for automated scientific discovery
    """
  
    def __init__(self, 
                 model_type: str = "huggingface",
                 model_name: str = "meta-llama/Llama-2-7b-chat-hf",
                 task_type: str = "regression",
                 **model_kwargs):
        """
        Initialize LLM-Feynman
      
        Args:
            model_type: Type of model ("huggingface", "openai")
            model_name: Name of the model
            task_type: Type of task ("regression" or "classification")
            **model_kwargs: Additional model arguments
        """
        self.task_type = task_type
        self.model_kwargs = model_kwargs
      
        # Initialize LLM model
        if model_type.lower() == "huggingface":
            self.llm_model = HuggingFaceModel(model_name, **model_kwargs)
        elif model_type.lower() == "openai":
            self.llm_model = OpenAIModel(model_name, **model_kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
      
        # Initialize modules
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer(self.llm_model, task_type)
        self.symbolic_regressor = SymbolicRegressor(self.llm_model, task_type)
        self.formula_interpreter = FormulaInterpreter(self.llm_model)
      
        # Results storage
        self.results = {}
      
    def discover_formulas(self, 
                         X: pd.DataFrame, 
                         y: pd.Series,
                         feature_meanings: Optional[Dict[str, str]] = None,
                         feature_dimensions: Optional[Dict[str, str]] = None,
                         target_meaning: Optional[str] = None,
                         target_dimension: Optional[str] = None,
                         preprocessing_config: Optional[Dict[str, Any]] = None,
                         feature_engineering_config: Optional[Dict[str, Any]] = None,
                         symbolic_regression_config: Optional[Dict[str, Any]] = None,
                         include_interpretation: bool = True,
                         enable_dimensional_analysis: bool = False,
                         n_initial_formulas: int = 100,
                         initial_formula_batches: int = 5,
                         max_iterations: int = 50,
                         formulas_per_iteration: int = 20,
                         top_formulas_for_history: int = 20,
                         verbose_loss: bool = False) -> List[Formula]:
        """
        Discover mathematical formulas from data
      
        Args:
            X: Feature matrix
            y: Target values
            feature_meanings: Physical meanings of features
            feature_dimensions: Dimensions of features
            target_meaning: Physical meaning of target
            target_dimension: Dimension of target
            preprocessing_config: Configuration for preprocessing
            feature_engineering_config: Configuration for feature engineering
            symbolic_regression_config: Configuration for symbolic regression
            include_interpretation: Whether to include MCTS-based interpretation
            enable_dimensional_analysis: Whether to enable dimensional analysis in self-evaluation
            n_initial_formulas: Total number of initial formulas to generate (default: 100)
            initial_formula_batches: Number of batches for initial formula generation (default: 5)
            max_iterations: Maximum number of iterations for optimization (default: 50)
            formulas_per_iteration: Number of formulas per iteration (default: 20)
            top_formulas_for_history: Number of top formulas to include in history (default: 20)
            verbose_loss: Whether to output detailed loss function components for each formula
           
         Returns:
             List of discovered formulas
        """
        print("Starting LLM-Feynman discovery process...")
      
        # Set default configurations
        preprocessing_config = preprocessing_config or {}
        feature_engineering_config = feature_engineering_config or {}
        symbolic_regression_config = symbolic_regression_config or {}
      
        # Module I: Data preprocessing and feature engineering
        print("\n=== Module I: Data Preprocessing ===")
        X_processed, y_processed, metadata = self.preprocessor.preprocess(
            X, y, feature_meanings, feature_dimensions, 
            target_meaning, target_dimension, **preprocessing_config
        )
      
        print("\n=== Module I: Feature Engineering ===")
        X_engineered, metadata = self.feature_engineer.engineer_features(
            X_processed, y_processed, metadata, **feature_engineering_config
        )
      
        # Module II: Symbolic regression with self-evaluation
        print("\n=== Module II: Symbolic Regression ===")
        # Add dimensional analysis configuration to symbolic regression config
        symbolic_regression_config['enable_dimensional_analysis'] = enable_dimensional_analysis
        # Add parallel processing configuration
        if 'max_workers' not in symbolic_regression_config:
            symbolic_regression_config['max_workers'] = 4
        # Add new configuration parameters
        symbolic_regression_config.update({
            'n_initial_formulas': n_initial_formulas,
            'initial_formula_batches': initial_formula_batches,
            'max_iterations': max_iterations,
            'formulas_per_iteration': formulas_per_iteration,
            'top_formulas_for_history': top_formulas_for_history,
            'verbose_loss': verbose_loss
        })
        discovered_formulas = self.symbolic_regressor.symbolic_regression(
            X_engineered, y_processed, metadata, **symbolic_regression_config
        )
      
        # Store results
        self.results = {
            'original_data': (X, y),
            'processed_data': (X_processed, y_processed),
            'engineered_data': (X_engineered, y_processed),
            'metadata': metadata,
            'formulas': discovered_formulas,
            'preprocessing_config': preprocessing_config,
            'feature_engineering_config': feature_engineering_config,
            'symbolic_regression_config': symbolic_regression_config
        }
      
        # Module III: Formula interpretation (optional)
        if include_interpretation and discovered_formulas:
            print("\n=== Module III: Formula Interpretation ===")
            interpretation_results = self.interpret_formulas(discovered_formulas)
            self.results['interpretations'] = interpretation_results
      
        print(f"\n=== Discovery Complete ===")
        print(f"Found {len(discovered_formulas)} formulas on Pareto frontier")
        if include_interpretation:
            print("Formula interpretations generated using MCTS")
      
        return discovered_formulas
  
    def interpret_formulas(self, formulas: Optional[List[Formula]] = None) -> Dict[str, Any]:
        """
        Interpret discovered formulas using LLM-guided Monte Carlo tree search
      
        Args:
            formulas: List of formulas to interpret (defaults to discovered formulas)
          
        Returns:
            Comprehensive interpretation results
        """
        if formulas is None:
            if not self.results or not self.results.get('formulas'):
                print("No formulas to interpret. Run discover_formulas first.")
                return {}
            formulas = self.results['formulas']
      
        if not self.results:
            print("No data context available. Run discover_formulas first.")
            return {}
      
        print("Starting formula interpretation with MCTS...")
      
        X_engineered, y_processed = self.results['engineered_data']
        metadata = self.results['metadata']
      
        interpretation_results = self.formula_interpreter.interpret_formulas(
            formulas, X_engineered, y_processed, metadata
        )
      
        # Store interpretation results
        if 'interpretations' not in self.results:
            self.results['interpretations'] = {}
        self.results['interpretations'].update(interpretation_results)
      
        return interpretation_results
  
    def evaluate_formulas(self, formulas: List[Formula], X_test: Optional[pd.DataFrame] = None, 
                         y_test: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Evaluate formulas on test data"""
        if X_test is None or y_test is None:
            if not self.results or 'engineered_data' not in self.results:
                print("No data available for evaluation. Run discover_formulas first.")
                return {}
            X_test, y_test = self.results['engineered_data']
      
        evaluation_results = evaluate_formulas(formulas, X_test, y_test, self.task_type)
        return evaluation_results
  
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """Plot discovery results"""
        if not self.results:
            print("No results to plot. Run discover_formulas first.")
            return
      
        plot_results(self.results, save_path)
  
    def get_best_formula(self, metric: str = "loss") -> Optional[Formula]:
        """Get the best formula based on specified metric"""
        if not self.results or not self.results.get('formulas'):
            print("No formulas available. Run discover_formulas first.")
            return None
      
        formulas = self.results['formulas']
      
        if metric == "loss":
            return min(formulas, key=lambda f: f.loss)
        elif metric == "r2":
            return max(formulas, key=lambda f: f.r2)
        elif metric == "interpretability":
            return max(formulas, key=lambda f: f.interpretability)
        elif metric == "simplicity":
            return min(formulas, key=lambda f: f.complexity)
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'loss', 'r2', 'interpretability', or 'simplicity'.")
  
    def get_formula_interpretation(self, formula_id: Optional[int] = None) -> Optional[str]:
        """
        Get interpretation for a specific formula
      
        Args:
            formula_id: Index of formula to get interpretation for (default: best formula)
          
        Returns:
            Formula interpretation text
        """
        if not self.results or 'interpretations' not in self.results:
            print("No interpretations available. Run discover_formulas with include_interpretation=True.")
            return None
      
        interpretations = self.results['interpretations']
      
        if formula_id is None:
            # Get interpretation for best formula
            best_formula = self.get_best_formula()
            if not best_formula:
                return None
          
            # Find corresponding interpretation
            for key, interp_result in interpretations.get('individual_interpretations', {}).items():
                if interp_result.get('formula') == best_formula:
                    return interp_result.get('final_interpretation', '')
        else:
            # Get interpretation for specific formula
            key = f"formula_{formula_id}"
            if key in interpretations.get('individual_interpretations', {}):
                return interpretations['individual_interpretations'][key].get('final_interpretation', '')
      
        return None
  
    def print_formula_summary(self, top_n: int = 5) -> None:
        """Print summary of top discovered formulas with interpretations"""
        if not self.results or not self.results.get('formulas'):
            print("No formulas to summarize. Run discover_formulas first.")
            return
      
        formulas = self.results['formulas']
        top_formulas = sorted(formulas, key=lambda f: f.r2, reverse=True)[:top_n]
      
        print(f"\n{'='*80}")
        print(f"TOP {len(top_formulas)} DISCOVERED FORMULAS")
        print(f"{'='*80}")
      
        for i, formula in enumerate(top_formulas, 1):
            print(f"\n{i}. Formula: {formula.expression}")
            print(f"   Performance: R² = {formula.r2:.4f}, MAE = {formula.mae:.4f}")
            print(f"   Complexity: {formula.complexity}, Interpretability: {formula.interpretability:.3f}")
            print(f"   Loss: {formula.loss:.4f}")
          
            # Get interpretation if available
            interpretation = self.get_formula_interpretation(i-1)
            if interpretation:
                print(f"   Interpretation: {interpretation[:200]}...")
            print("-" * 80)
  
    def export_formulas(self, filepath: str, format: str = "json", include_interpretations: bool = True) -> None:
        """Export discovered formulas to file"""
        if not self.results or not self.results.get('formulas'):
            print("No formulas to export.")
            return
      
        formulas_data = []
        for i, formula in enumerate(self.results['formulas']):
            formula_dict = {
                'id': i,
                'expression': formula.expression,
                'mae': formula.mae,
                'r2': formula.r2,
                'complexity': formula.complexity,
                'interpretability': formula.interpretability,
                'loss': formula.loss,
                'meaning': getattr(formula, 'meaning', ''),
                'dimension': getattr(formula, 'dimension', '')
            }
          
            # Add interpretation if available
            if include_interpretations:
                interpretation = self.get_formula_interpretation(i)
                formula_dict['interpretation'] = interpretation or ''
          
            formulas_data.append(formula_dict)
      
        # Add metadata
        export_data = {
            'formulas': formulas_data,
            'metadata': self.results.get('metadata', {}),
            'summary': {
                'total_formulas': len(formulas_data),
                'task_type': self.task_type,
                'best_r2': max(f['r2'] for f in formulas_data) if formulas_data else 0,
                'best_interpretability': max(f['interpretability'] for f in formulas_data) if formulas_data else 0
            }
        }
      
        # Add interpretation summary if available
        if include_interpretations and 'interpretations' in self.results:
            interp_summary = self.results['interpretations'].get('summary', {})
            export_data['interpretation_summary'] = interp_summary
      
        if format.lower() == "json":
            import json
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
        elif format.lower() == "csv":
            # Export formulas as CSV (interpretations in separate file if included)
            df = pd.DataFrame(formulas_data)
            df.to_csv(filepath, index=False)
          
            # Export interpretations separately if included
            if include_interpretations:
                interp_filepath = filepath.replace('.csv', '_interpretations.csv')
                interp_data = []
                for formula_dict in formulas_data:
                    if formula_dict.get('interpretation'):
                        interp_data.append({
                            'formula_id': formula_dict['id'],
                            'expression': formula_dict['expression'],
                            'interpretation': formula_dict['interpretation']
                        })
                if interp_data:
                    pd.DataFrame(interp_data).to_csv(interp_filepath, index=False)
                    print(f"Interpretations exported to {interp_filepath}")
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")
      
        print(f"Formulas exported to {filepath}")
  
    def save_results(self, filepath: str) -> None:
        """Save complete results to pickle file"""
        if not self.results:
            print("No results to save.")
            return
      
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Complete results saved to {filepath}")
  
    def load_results(self, filepath: str) -> None:
        """Load results from pickle file"""
        import pickle
        with open(filepath, 'rb') as f:
            self.results = pickle.load(f)
        print(f"Results loaded from {filepath}")
  
    def compare_formulas(self, metrics: List[str] = None) -> pd.DataFrame:
        """
        Compare discovered formulas across multiple metrics
      
        Args:
            metrics: List of metrics to compare (default: ['r2', 'mae', 'complexity', 'interpretability'])
          
        Returns:
            DataFrame with formula comparison
        """
        if not self.results or not self.results.get('formulas'):
            print("No formulas to compare. Run discover_formulas first.")
            return pd.DataFrame()
      
        if metrics is None:
            metrics = ['r2', 'mae', 'complexity', 'interpretability', 'loss']
      
        formulas = self.results['formulas']
        comparison_data = []
      
        for i, formula in enumerate(formulas):
            row = {'formula_id': i, 'expression': formula.expression}
          
            for metric in metrics:
                if hasattr(formula, metric):
                    row[metric] = getattr(formula, metric)
          
            comparison_data.append(row)
      
        df = pd.DataFrame(comparison_data)
      
        # Add rankings for each metric
        for metric in metrics:
            if metric in df.columns:
                ascending = metric in ['mae', 'complexity', 'loss']  # Lower is better for these
                df[f'{metric}_rank'] = df[metric].rank(ascending=ascending, method='min')
      
        return df.sort_values('r2', ascending=False)
  
    def get_pareto_front(self) -> List[Formula]:
        """Get formulas on the Pareto frontier (accuracy vs complexity)"""
        if not self.results or not self.results.get('formulas'):
            print("No formulas available. Run discover_formulas first.")
            return []
      
        formulas = self.results['formulas']
      
        # Sort by R² (descending)
        sorted_formulas = sorted(formulas, key=lambda f: f.r2, reverse=True)
      
        pareto_front = []
        min_complexity = float('inf')
      
        for formula in sorted_formulas:
            if formula.complexity < min_complexity:
                pareto_front.append(formula)
                min_complexity = formula.complexity
      
        return pareto_front
  
    def validate_formula(self, formula: Formula, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        """
        Validate a formula on validation data
      
        Args:
            formula: Formula to validate
            X_val: Validation features
            y_val: Validation targets
          
        Returns:
            Validation metrics
        """
        try:
            y_pred = formula.function(X_val)
          
            if self.task_type == "regression":
                from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
                mae = mean_absolute_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                mse = mean_squared_error(y_val, y_pred)
                rmse = np.sqrt(mse)
              
                return {
                    'mae': mae,
                    'r2': r2,
                    'mse': mse,
                    'rmse': rmse
                }
            else:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                y_pred_class = (y_pred > 0.5).astype(int) if len(np.unique(y_val)) == 2 else np.argmax(y_pred, axis=1)
              
                return {
                    'accuracy': accuracy_score(y_val, y_pred_class),
                    'precision': precision_score(y_val, y_pred_class, average='weighted'),
                    'recall': recall_score(y_val, y_pred_class, average='weighted'),
                    'f1': f1_score(y_val, y_pred_class, average='weighted')
                }
        except Exception as e:
            print(f"Validation failed: {e}")
            return {}