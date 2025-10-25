# llm_feynman/utils/metrics.py
import numpy as np
import pandas as pd
<<<<<<< HEAD
from typing import List, Dict, Any
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from ..core.symbolic_regression import Formula

def evaluate_formulas(formulas: List[Formula], X: pd.DataFrame, y: pd.Series, task_type: str) -> Dict[str, Any]:
    """Evaluate a list of formulas on test data"""
    
    results = {
        'individual_results': [],
        'summary_stats': {},
        'best_formula': None
=======
from typing import Dict, Any, List
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from core.symbolic_regression import Formula

def evaluate_formulas(formulas: List[Formula], X: pd.DataFrame, y: pd.Series, 
                     task_type: str = "regression") -> Dict[str, Any]:
    """Evaluate formulas on test data"""
    
    evaluation_results = {
        'individual_results': [],
        'summary': {}
>>>>>>> 10f4e4c (Initial import of core LLM-Feynman modules)
    }
    
    for i, formula in enumerate(formulas):
        try:
            y_pred = formula.function(X)
            
            if task_type == "regression":
                mae = mean_absolute_error(y, y_pred)
                r2 = r2_score(y, y_pred)
<<<<<<< HEAD
=======
                mse = mean_squared_error(y, y_pred)
>>>>>>> 10f4e4c (Initial import of core LLM-Feynman modules)
                
                result = {
                    'formula_id': i,
                    'expression': formula.expression,
                    'mae': mae,
                    'r2': r2,
<<<<<<< HEAD
                    'complexity': formula.complexity,
                    'interpretability': formula.interpretability
                }
            else:
                # Classification
                y_pred_class = (y_pred > 0.5).astype(int) if len(np.unique(y)) == 2 else np.argmax(y_pred, axis=1)
                
                accuracy = accuracy_score(y, y_pred_class)
                precision = precision_score(y, y_pred_class, average='weighted')
                recall = recall_score(y, y_pred_class, average='weighted')
                f1 = f1_score(y, y_pred_class, average='weighted')
                
                result = {
                    'formula_id': i,
                    'expression': formula.expression,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'complexity': formula.complexity,
                    'interpretability': formula.interpretability
                }
            
            results['individual_results'].append(result)
            
        except Exception as e:
            print(f"Warning: Evaluation failed for formula {i}: {e}")
    
    # Calculate summary statistics
    if results['individual_results']:
        if task_type == "regression":
            mae_values = [r['mae'] for r in results['individual_results']]
            r2_values = [r['r2'] for r in results['individual_results']]
            
            results['summary_stats'] = {
                'mean_mae': np.mean(mae_values),
                'std_mae': np.std(mae_values),
                'mean_r2': np.mean(r2_values),
                'std_r2': np.std(r2_values),
                'best_mae': np.min(mae_values),
                'best_r2': np.max(r2_values)
            }
            
            # Find best formula (highest R²)
            best_idx = np.argmax(r2_values)
            results['best_formula'] = results['individual_results'][best_idx]
        else:
            accuracy_values = [r['accuracy'] for r in results['individual_results']]
            
            results['summary_stats'] = {
                'mean_accuracy': np.mean(accuracy_values),
                'std_accuracy': np.std(accuracy_values),
                'best_accuracy': np.max(accuracy_values)
            }
            
            # Find best formula (highest accuracy)
            best_idx = np.argmax(accuracy_values)
            results['best_formula'] = results['individual_results'][best_idx]
    
    return results
=======
                    'mse': mse,
                    'rmse': np.sqrt(mse)
                }
            else:
                # Classification metrics would go here
                result = {
                    'formula_id': i,
                    'expression': formula.expression,
                    'accuracy': 0.0  # Placeholder
                }
            
            evaluation_results['individual_results'].append(result)
            
        except Exception as e:
            print(f"Evaluation failed for formula {i}: {e}")
    
    # Summary statistics
    if evaluation_results['individual_results']:
        results = evaluation_results['individual_results']
        evaluation_results['summary'] = {
            'num_formulas': len(results),
            'best_r2': max(r['r2'] for r in results if 'r2' in r),
            'best_mae': min(r['mae'] for r in results if 'mae' in r),
            'avg_r2': np.mean([r['r2'] for r in results if 'r2' in r]),
            'avg_mae': np.mean([r['mae'] for r in results if 'mae' in r])
        }
    
    return evaluation_results
>>>>>>> 10f4e4c (Initial import of core LLM-Feynman modules)
