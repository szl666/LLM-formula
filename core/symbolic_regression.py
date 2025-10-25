# llm_feynman/core/symbolic_regression.py
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Callable
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from dataclasses import dataclass
import re
import ast

@dataclass
class Formula:
    """Data class for storing formula information"""
    expression: str
    function: Callable
    mae: float
    r2: float
    complexity: int
    interpretability: float
    loss: float
    meaning: str = ""
    dimension: str = ""

class SymbolicRegressor:
    """Symbolic regression module for LLM-Feynman"""
    
    def __init__(self, llm_model, task_type: str = "regression"):
        """
        Args:
            llm_model: LLM model for formula generation
            task_type: Type of task ("regression" or "classification")
        """
        self.llm_model = llm_model
        self.task_type = task_type
        self.formulas = []
        self.best_formulas = []
        
    def symbolic_regression(self, X: pd.DataFrame, y: pd.Series, 
                          metadata: Dict[str, Any],
                          n_initial_formulas: int = 30,
                          n_iterations: int = 500,
                          n_new_formulas_per_iter: int = 10,
                          n_best_formulas: int = 30,
                          alpha: float = 1.0,
                          beta: float = 1.0,
                          gamma: float = 1.0) -> List[Formula]:
        """
        Perform symbolic regression with self-evaluation and multi-objective optimization
        
        Args:
            X: Feature matrix
            y: Target values
            metadata: Data metadata
            n_initial_formulas: Number of initial formulas to generate
            n_iterations: Number of optimization iterations
            n_new_formulas_per_iter: Number of new formulas per iteration
            n_best_formulas: Number of best formulas to keep for next iteration
            alpha: Weight for error term in loss function
            beta: Weight for complexity term in loss function
            gamma: Weight for interpretability term in loss function
            
        Returns:
            List of optimized formulas
        """
        # Step 1: Generate initial formulas
        print("Generating initial formulas...")
        initial_formulas = self._generate_initial_formulas(
            X, y, metadata, n_initial_formulas
        )
        
        # Step 2: Evaluate and score initial formulas
        evaluated_formulas = []
        for formula_expr in initial_formulas:
            formula = self._evaluate_formula(formula_expr, X, y, metadata, alpha, beta, gamma)
            if formula:
                evaluated_formulas.append(formula)
        
        self.formulas = evaluated_formulas
        
        # Step 3: Iterative optimization
        print(f"Starting iterative optimization for {n_iterations} iterations...")
        for iteration in range(n_iterations):
            if iteration % 50 == 0:
                print(f"Iteration {iteration}/{n_iterations}")
            
            # Select best formulas
            best_formulas = self._select_best_formulas(self.formulas, n_best_formulas)
            
            # Generate new formulas based on best ones
            new_formula_expressions = self._generate_new_formulas(
                best_formulas, X, y, metadata, n_new_formulas_per_iter
            )
            
            # Evaluate new formulas
            new_formulas = []
            for formula_expr in new_formula_expressions:
                formula = self._evaluate_formula(formula_expr, X, y, metadata, alpha, beta, gamma)
                if formula:
                    new_formulas.append(formula)
            
            # Add new formulas to the pool
            self.formulas.extend(new_formulas)
        
        # Step 4: Pareto frontier analysis
        print("Performing Pareto frontier analysis...")
        pareto_formulas = self._pareto_frontier_analysis(self.formulas)
        
        return pareto_formulas
    
    def _generate_initial_formulas(self, X: pd.DataFrame, y: pd.Series, 
                                  metadata: Dict[str, Any], n_formulas: int) -> List[str]:
        """Generate initial formulas using LLM"""
        from ..templates.prompt_templates import SymbolicRegressionPrompts, PromptManager
        
        # Format data points for prompt
        points_data = PromptManager.format_data_points(X, y)
        
        # Prepare variables and meanings
        variables_list = list(X.columns)
        meanings_list = [metadata.get('feature_meanings', {}).get(var, 'Unknown') for var in variables_list]
        target_meaning = metadata.get('target_meaning', 'target property')
        
        prompt = SymbolicRegressionPrompts.initial_formula_generation_prompt(
            points_data=points_data,
            target_meaning=target_meaning,
            variables_list=variables_list,
            meanings_list=meanings_list,
            num_variables=len(variables_list),
            num_formulas=n_formulas
        )
        
        try:
            response = self.llm_model.generate(prompt)
            formulas = PromptManager.extract_python_functions(response)
            return formulas[:n_formulas]
        except Exception as e:
            print(f"Warning: Formula generation failed: {e}")
            return []
    
    def _parse_formula_response(self, response: str) -> List[str]:
        """Parse LLM response to extract formula expressions"""
        formulas = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for function definitions
            if line.startswith('def ') and '(X):' in line:
                # Extract the return statement
                try:
                    # Simple parsing - look for return statement
                    if 'return ' in line:
                        formula = line.split('return ')[1].strip()
                        formulas.append(formula)
                except:
                    continue
            elif line.startswith('return '):
                formula = line.split('return ')[1].strip()
                formulas.append(formula)
            elif line.startswith('y = '):
                formula = line.split('y = ')[1].strip()
                formulas.append(formula)
        
        return formulas
    
    def _evaluate_formula(self, formula_expr: str, X: pd.DataFrame, y: pd.Series, 
                         metadata: Dict[str, Any], alpha: float, beta: float, gamma: float) -> Optional[Formula]:
        """Evaluate a single formula"""
        try:
            # Create function from expression
            func = self._create_function(formula_expr, X.columns)
            
            # Test the function
            y_pred = func(X)
            
            # Calculate metrics
            if self.task_type == "regression":
                mae = mean_absolute_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                error = mae
            else:
                # For classification
                y_pred_class = (y_pred > 0.5).astype(int) if len(np.unique(y)) == 2 else np.argmax(y_pred, axis=1)
                mae = 1 - accuracy_score(y, y_pred_class)
                r2 = accuracy_score(y, y_pred_class)
                error = mae
            
            # Calculate complexity
            complexity = self._calculate_complexity(formula_expr)
            
            # Get interpretability score from LLM
            interpretability = self._get_interpretability_score(
                formula_expr, metadata
            )
            
            # Calculate loss
            error_norm = self._normalize_metric(error, 'error')
            complexity_norm = self._normalize_metric(complexity, 'complexity')
            
            loss = alpha * error_norm + beta * complexity_norm + gamma * (1 - interpretability)
            
            return Formula(
                expression=formula_expr,
                function=func,
                mae=mae,
                r2=r2,
                complexity=complexity,
                interpretability=interpretability,
                loss=loss
            )
            
        except Exception as e:
            print(f"Warning: Formula evaluation failed for '{formula_expr}': {e}")
            return None
    
    def _create_function(self, formula_expr: str, feature_names: List[str]) -> Callable:
        """Create a callable function from formula expression"""
        
        # Clean the expression
        formula_expr = formula_expr.strip().rstrip(',').rstrip(';')
        
        # Create safe namespace
        safe_dict = {
            'np': np,
            'log': np.log,
            'sqrt': np.sqrt,
            'exp': np.exp,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'abs': np.abs,
            'pow': np.power,
            '__builtins__': {}
        }
        
        def formula_function(X):
            # Add feature columns to namespace
            local_dict = safe_dict.copy()
            for col in feature_names:
                local_dict[col] = X[col]
            
            # Evaluate the expression
            result = eval(formula_expr, local_dict)
            
            # Ensure result is a pandas Series
            if isinstance(result, (int, float)):
                result = pd.Series([result] * len(X), index=X.index)
            elif isinstance(result, np.ndarray):
                result = pd.Series(result, index=X.index)
            
            return result
        
        return formula_function
    
    def _calculate_complexity(self, formula_expr: str) -> int:
        """Calculate formula complexity based on number of operations and terms"""
        # Count operators
        operators = ['+', '-', '*', '/', '**', 'pow', 'log', 'sqrt', 'exp', 'sin', 'cos', 'tan']
        complexity = 0
        
        for op in operators:
            complexity += formula_expr.count(op)
        
        # Count parentheses (nested operations)
        complexity += formula_expr.count('(')
        
        # Count unique variables
        import re
        variables = re.findall(r'\b[A-Za-z_]\w*\b', formula_expr)
        unique_vars = set(variables) - {'np', 'log', 'sqrt', 'exp', 'sin', 'cos', 'tan', 'abs', 'pow'}
        complexity += len(unique_vars)
        
        return max(1, complexity)  # Minimum complexity of 1
    
    def _get_interpretability_score(self, formula_expr: str, metadata: Dict[str, Any]) -> float:
        """Get interpretability score from LLM"""
        from ..templates.prompt_templates import SymbolicRegressionPrompts
        
        # Prepare data for self-evaluation
        variables_list = list(metadata.get('feature_meanings', {}).keys())
        meanings_list = [metadata.get('feature_meanings', {}).get(var, 'Unknown') for var in variables_list]
        target_meaning = metadata.get('target_meaning', 'target property')
        
        prompt = SymbolicRegressionPrompts.formula_self_evaluation_prompt(
            mathematical_functions=[formula_expr],
            meanings_list=meanings_list,
            target_meaning=target_meaning
        )
        
        try:
            response = self.llm_model.generate(prompt)
            score = self._extract_score_from_response(response)
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"Warning: Interpretability scoring failed: {e}")
            return 0.5
    
    def _extract_score_from_response(self, response: str) -> float:
        """Extract numerical score from LLM response"""
        import re
        
        # Look for patterns like "Score: 0.8" or "0.8/1.0" or just "0.8"
        patterns = [
            r'score[:\s]*([0-9]*\.?[0-9]+)',
            r'([0-9]*\.?[0-9]+)/1\.?0?',
            r'([0-9]*\.?[0-9]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                try:
                    score = float(matches[0])
                    if 0 <= score <= 1:
                        return score
                    elif score <= 10:  # Assume 0-10 scale
                        return score / 10.0
                except ValueError:
                    continue
        
        return 0.5  # Default if no score found
    
    def _normalize_metric(self, value: float, metric_type: str) -> float:
        """Normalize metrics to [0, 1] range"""
        if metric_type == 'error':
            # For error metrics, higher is worse
            return min(1.0, value)  # Simple normalization
        elif metric_type == 'complexity':
            # For complexity, normalize based on typical range
            return min(1.0, value / 20.0)  # Assume max complexity of 20
        else:
            return value
    
    def _select_best_formulas(self, formulas: List[Formula], n_best: int) -> List[Formula]:
        """Select best formulas based on loss function"""
        sorted_formulas = sorted(formulas, key=lambda f: f.loss)
        return sorted_formulas[:n_best]
    
    def _generate_new_formulas(self, best_formulas: List[Formula], X: pd.DataFrame, 
                              y: pd.Series, metadata: Dict[str, Any], n_new: int) -> List[str]:
        """Generate new formulas based on best performing ones"""
        from ..templates.prompt_templates import SymbolicRegressionPrompts, PromptManager
        
        # Format data points
        points_data = PromptManager.format_data_points(X, y)
        
        # Format previous trajectory
        formulas_data = []
        for formula in best_formulas[:10]:
            formulas_data.append({
                'expression': formula.expression,
                'error': formula.mae,
                'interpretability': formula.interpretability
            })
        
        previous_trajectory = PromptManager.format_previous_trajectory(formulas_data)
        
        # Prepare variables and meanings
        variables_list = list(X.columns)
        meanings_list = [metadata.get('feature_meanings', {}).get(var, 'Unknown') for var in variables_list]
        target_meaning = metadata.get('target_meaning', 'target property')
        
        prompt = SymbolicRegressionPrompts.iterative_formula_generation_prompt(
            points_data=points_data,
            previous_trajectory=previous_trajectory,
            target_meaning=target_meaning,
            variables_list=variables_list,
            meanings_list=meanings_list,
            num_variables=len(variables_list),
            num_formulas=n_new
        )
        
        try:
            response = self.llm_model.generate(prompt)
            new_formulas = PromptManager.extract_python_functions(response)
            return new_formulas[:n_new]
        except Exception as e:
            print(f"Warning: New formula generation failed: {e}")
            return []
    
    def _pareto_frontier_analysis(self, formulas: List[Formula]) -> List[Formula]:
        """Identify formulas on the Pareto frontier (accuracy vs simplicity)"""
        if not formulas:
            return []
        
        # Sort by accuracy (R2 for regression, 1-MAE for classification)
        if self.task_type == "regression":
            sorted_formulas = sorted(formulas, key=lambda f: f.r2, reverse=True)
        else:
            sorted_formulas = sorted(formulas, key=lambda f: (1 - f.mae), reverse=True)
        
        pareto_front = []
        min_complexity = float('inf')
        
        for formula in sorted_formulas:
            if formula.complexity < min_complexity:
                pareto_front.append(formula)
                min_complexity = formula.complexity
        
        return pareto_front
