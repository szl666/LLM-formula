# llm_feynman/core/symbolic_regression.py
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Callable
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, log_loss
from dataclasses import dataclass
import re
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

@dataclass
class Formula:
    """Data class for storing formula information"""
    expression: str
    function: Callable
    mae: float
    r2: float
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    cross_entropy: float = 0.0
    complexity: int = 0
    interpretability: float = 0.0
    loss: float = 0.0
    meaning: str = ""
    dimension: str = ""
    # Loss function components for detailed output
    error_norm: float = 0.0
    complexity_norm: float = 0.0
    interpretability_norm: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0

class SymbolicRegressor:
    """Enhanced symbolic regression module for LLM-Feynman"""
    
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
        self.metric_bounds = {
            'error_min': float('inf'),
            'error_max': 0.0,
            'complexity_min': float('inf'),
            'complexity_max': 0.0
        }
    def symbolic_regression(self, X: pd.DataFrame, y: pd.Series, 
                           metadata: Dict[str, Any],
                           n_initial_formulas: int = 100,
                           alpha: float = 1.0,
                           beta: float = 1.0,
                           gamma: float = 1.0,
                           enable_dimensional_analysis: bool = False,
                           max_workers: int = 4,
                           initial_formula_batches: int = 5,
                           max_iterations: int = 50,
                           formulas_per_iteration: int = 20,
                           top_formulas_for_history: int = 20,
                           verbose_loss: bool = False) -> List[Formula]:
        """
        Enhanced symbolic regression with iterative generation and multi-objective optimization
        
        Args:
            X: Feature matrix
            y: Target values
            metadata: Data metadata
            n_initial_formulas: Total number of initial formulas to generate (default: 100)
            alpha: Weight for error term in loss function (default: 1.0)
            beta: Weight for complexity term in loss function (default: 1.0)
            gamma: Weight for interpretability term in loss function (default: 1.0)
            enable_dimensional_analysis: Whether to enable dimensional analysis (default: False)
            max_workers: Maximum number of parallel workers for evaluation (default: 4)
            initial_formula_batches: Number of batches for initial formula generation (default: 5)
            max_iterations: Maximum number of iterations for optimization (default: 50)
            formulas_per_iteration: Number of formulas per iteration (default: 20)
            top_formulas_for_history: Number of top formulas to include in history (default: 20)
            verbose_loss: Whether to output detailed loss function components for each formula (default: False)
            
        Returns:
            List of optimized formulas on Pareto frontier
        """
        print("Starting enhanced symbolic regression...")
        print(f"Configuration:")
        formulas_per_batch_calculated = n_initial_formulas // initial_formula_batches
        print(f"  - Initial generation: {n_initial_formulas} total formulas in {initial_formula_batches} batches")
        print(f"  - Each batch will generate: {formulas_per_batch_calculated} formulas")
        print(f"  - Iterative optimization: {max_iterations} iterations × {formulas_per_iteration} formulas per iteration")
        print(f"  - History size: top {top_formulas_for_history} formulas")
        print(f"  - Dimensional analysis: {'enabled' if enable_dimensional_analysis else 'disabled'}")
        print(f"  - Verbose loss output: {'enabled' if verbose_loss else 'disabled'}")
        
        start_time = time.time()
        
        # Step 1: Generate initial formulas in batches (NO HISTORY) - PARALLEL
        print(f"\n=== STEP 1: INITIAL FORMULA GENERATION (PARALLEL) ===")
        print(f"Generating {n_initial_formulas} initial formulas in {initial_formula_batches} parallel batches")
        print(f"Each batch will generate {formulas_per_batch_calculated} formulas")
        
        initial_formulas = self._generate_initial_formulas_parallel(
            X, y, metadata, initial_formula_batches, formulas_per_batch_calculated, max_workers
        )
        
        print(f"Total initial formulas generated: {len(initial_formulas)}")
        
        # Ensure we have enough initial formulas
        if len(initial_formulas) < n_initial_formulas:
            print(f"Warning: Only generated {len(initial_formulas)} formulas, need {n_initial_formulas}")
            print("Attempting to generate more formulas iteratively...")
            
            additional_formulas = self._generate_initial_formulas_iterative(
                X, y, metadata, n_initial_formulas - len(initial_formulas)
            )
            initial_formulas.extend(additional_formulas)
            print(f"Generated {len(additional_formulas)} additional formulas")
            print(f"Total initial formulas: {len(initial_formulas)}")
        
        # Step 2: Evaluate and score initial formulas with parallel processing
        print(f"\n=== STEP 2: INITIAL FORMULA EVALUATION ===")
        print("Evaluating initial formulas...")
        evaluated_formulas = self._evaluate_formulas_parallel(
            initial_formulas, X, y, metadata, alpha, beta, gamma, 
            enable_dimensional_analysis, max_workers
        )
        
        self.formulas = evaluated_formulas
        print(f"Successfully evaluated {len(evaluated_formulas)} formulas")
        
        # Output detailed loss function information if requested
        if verbose_loss and evaluated_formulas:
            print("\n" + "="*60)
            print("DETAILED LOSS FUNCTION ANALYSIS (Initial Formulas)")
            print("="*60)
            self._print_detailed_loss_analysis(evaluated_formulas, alpha, beta, gamma)
        
        # Update metric bounds for normalization
        self._update_metric_bounds(evaluated_formulas)
        
        # Step 3: Iterative optimization with progress tracking (WITH HISTORY)
        print(f"\n=== STEP 3: ITERATIVE OPTIMIZATION ===")
        print(f"Starting iterative optimization for {max_iterations} iterations...")
        print(f"Each iteration will generate {formulas_per_iteration} formulas using top {top_formulas_for_history} formulas as history")
        
        for iteration in range(max_iterations):
            if iteration % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Iteration {iteration + 1}/{max_iterations} (elapsed: {elapsed:.1f}s)")
            
            # Select best formulas based on loss function for history
            best_formulas = self._select_best_formulas(self.formulas, top_formulas_for_history)
            
            # Generate new formulas based on best ones (WITH HISTORY) - SEQUENTIAL
            print(f"  Iteration {iteration + 1}: Generating {formulas_per_iteration} new formulas...")
            new_formula_expressions = self._generate_new_formulas_with_history(
                best_formulas, X, y, metadata, formulas_per_iteration
            )
            
            # Evaluate new formulas in parallel
            new_formulas = self._evaluate_formulas_parallel(
                new_formula_expressions, X, y, metadata, alpha, beta, gamma,
                enable_dimensional_analysis, max_workers
            )
            
            # Add new formulas to the pool
            self.formulas.extend(new_formulas)
            print(f"  Iteration {iteration + 1}: Generated and evaluated {len(new_formulas)} new formulas")
             
            # Output detailed loss function information if requested
            if verbose_loss and new_formulas:
                print(f"\n--- Loss Analysis (Iteration {iteration + 1}) ---")
                self._print_detailed_loss_analysis(new_formulas, alpha, beta, gamma)
            
            # Update metric bounds
            self._update_metric_bounds(new_formulas)
            
            # Early stopping if no improvement
            if iteration > 20 and len(self.formulas) > 50:
                recent_improvement = self._check_improvement(iteration)
                if not recent_improvement:
                    print(f"Early stopping at iteration {iteration + 1} due to no improvement")
                    break
        
        # Step 4: Enhanced Pareto frontier analysis
        print(f"\n=== STEP 4: PARETO FRONTIER ANALYSIS ===")
        print("Performing enhanced Pareto frontier analysis...")
        pareto_formulas = self._enhanced_pareto_frontier_analysis(self.formulas)
        
        total_time = time.time() - start_time
        print(f"\n=== FINAL SUMMARY ===")
        print(f"Symbolic regression completed in {total_time:.1f} seconds")
        print(f"Total formulas generated: {len(self.formulas)}")
        print(f"Found {len(pareto_formulas)} formulas on Pareto frontier")
        
        return pareto_formulas
    
    def _generate_initial_formulas_in_batches(self, X: pd.DataFrame, y: pd.Series, 
                                            metadata: Dict[str, Any], num_batches: int, 
                                            formulas_per_batch: int) -> List[str]:
        """Generate initial formulas in multiple batches without history"""
        from templates.prompt_templates import SymbolicRegressionPrompts, PromptManager
        
        all_formulas = []
        
        # Format data points for prompt
        points_data = PromptManager.format_data_points(X, y)
        
        # Prepare variables and meanings
        variables_list = list(X.columns)
        meanings_list = [metadata.get('feature_meanings', {}).get(var, 'Unknown') for var in variables_list]
        target_meaning = metadata.get('target_meaning', 'target property')
        
        for batch in range(num_batches):
            print(f"  Generating batch {batch + 1}/{num_batches}...")
            
            # Generate formulas without previous trajectory (empty string)
            prompt = SymbolicRegressionPrompts.initial_formula_generation_prompt(
                points_data=points_data,
                target_meaning=target_meaning,
                variables_list=variables_list,
                meanings_list=meanings_list,
                num_variables=len(variables_list),
                num_formulas=formulas_per_batch
            )
            
            try:
                response = self.llm_model.generate(prompt)
                new_formulas = PromptManager.extract_python_functions(response)
                
                # Filter out duplicates and invalid formulas
                for formula in new_formulas:
                    if formula not in all_formulas and self._is_valid_formula(formula):
                        all_formulas.append(formula)
                        
            except Exception as e:
                print(f"Warning: Batch {batch + 1} generation failed: {e}")
        
        return all_formulas

    def _generate_initial_formulas_parallel(self, X: pd.DataFrame, y: pd.Series, 
                                          metadata: Dict[str, Any], num_batches: int, 
                                          formulas_per_batch: int, max_workers: int) -> List[str]:
        """Generate initial formulas in parallel batches without history"""
        from templates.prompt_templates import SymbolicRegressionPrompts, PromptManager
        
        # Format data points for prompt
        points_data = PromptManager.format_data_points(X, y)
        
        # Prepare variables and meanings
        variables_list = list(X.columns)
        meanings_list = [metadata.get('feature_meanings', {}).get(var, 'Unknown') for var in variables_list]
        target_meaning = metadata.get('target_meaning', 'target property')
        
        def generate_single_batch(batch_id):
            """Generate formulas for a single batch"""
            print(f"    Starting parallel batch {batch_id + 1}/{num_batches}...")
            
            # Generate formulas without previous trajectory (empty string)
            prompt = SymbolicRegressionPrompts.initial_formula_generation_prompt(
                points_data=points_data,
                target_meaning=target_meaning,
                variables_list=variables_list,
                meanings_list=meanings_list,
                num_variables=len(variables_list),
                num_formulas=formulas_per_batch
            )
            
            print(f"    Batch {batch_id + 1} prompt:")
            print(f"    {'='*50}")
            print(prompt)
            print(f"    {'='*50}")
            
            try:
                response = self.llm_model.generate(prompt)
                new_formulas = PromptManager.extract_python_functions(response)
                
                # Filter out duplicates and invalid formulas
                valid_formulas = []
                for formula in new_formulas:
                    if self._is_valid_formula(formula):
                        valid_formulas.append(formula)
                
                print(f"    Batch {batch_id + 1} generated {len(valid_formulas)} valid formulas")
                return valid_formulas
                        
            except Exception as e:
                print(f"    Warning: Batch {batch_id + 1} generation failed: {e}")
                return []
        
        # Generate formulas in parallel
        all_formulas = []
        if max_workers <= 1:
            # Sequential generation
            for batch in range(num_batches):
                batch_formulas = generate_single_batch(batch)
                all_formulas.extend(batch_formulas)
        else:
            # Parallel generation
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {
                    executor.submit(generate_single_batch, batch): batch
                    for batch in range(num_batches)
                }
                
                for future in as_completed(future_to_batch):
                    batch_formulas = future.result()
                    all_formulas.extend(batch_formulas)
        
        # Remove duplicates
        unique_formulas = []
        for formula in all_formulas:
            if formula not in unique_formulas:
                unique_formulas.append(formula)
        
        print(f"  Parallel generation completed: {len(unique_formulas)} unique formulas")
        return unique_formulas

    def _generate_initial_formulas_iterative(self, X: pd.DataFrame, y: pd.Series, 
                                           metadata: Dict[str, Any], target_n_formulas: int) -> List[str]:
        """Generate initial formulas with iterative mechanism until target number is reached"""
        from templates.prompt_templates import SymbolicRegressionPrompts, PromptManager
        
        all_formulas = []
        max_attempts = 10  # Prevent infinite loops
        attempt = 0
        
        while len(all_formulas) < target_n_formulas and attempt < max_attempts:
            attempt += 1
            
            # Calculate how many more formulas we need
            remaining = target_n_formulas - len(all_formulas)
            batch_size = min(remaining + 5, 10)  # Generate a few extra to account for failures
            
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
                num_formulas=batch_size
            )
            
            try:
                response = self.llm_model.generate(prompt)
                new_formulas = PromptManager.extract_python_functions(response)
                
                # Filter out duplicates and invalid formulas
                for formula in new_formulas:
                    if formula not in all_formulas and self._is_valid_formula(formula):
                        all_formulas.append(formula)
                        if len(all_formulas) >= target_n_formulas:
                            break
                            
            except Exception as e:
                print(f"Warning: Formula generation attempt {attempt} failed: {e}")
        
        return all_formulas[:target_n_formulas]
    
    def _is_valid_formula(self, formula: str) -> bool:
        """Check if formula is valid and safe to evaluate"""
        if not formula or len(formula.strip()) < 3:
            return False
        
        # Check for dangerous operations
        dangerous_patterns = [
            r'import\s+',
            r'exec\s*\(',
            r'eval\s*\(',
            r'__\w+__',
            r'open\s*\(',
            r'file\s*\(',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, formula, re.IGNORECASE):
                return False
        
        # Check for potentially problematic operations that might cause NaN
        problematic_patterns = [
            r'sqrt\s*\(\s*-',  # sqrt of negative number
            r'log\s*\(\s*[^)]*[<=]',  # log of non-positive number
        ]
        
        for pattern in problematic_patterns:
            if re.search(pattern, formula, re.IGNORECASE):
                return False
        
        return True
    
    def _evaluate_formulas_parallel(self, formula_expressions: List[str], X: pd.DataFrame, 
                                  y: pd.Series, metadata: Dict[str, Any], 
                                  alpha: float, beta: float, gamma: float,
                                  enable_dimensional_analysis: bool, max_workers: int) -> List[Formula]:
        """Evaluate formulas in parallel for better performance"""
        evaluated_formulas = []
        
        if max_workers <= 1:
            # Sequential evaluation
            for formula_expr in formula_expressions:
                formula = self._evaluate_formula(formula_expr, X, y, metadata, 
                                               alpha, beta, gamma, enable_dimensional_analysis)
                if formula:
                    evaluated_formulas.append(formula)
        else:
            # Parallel evaluation
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_formula = {
                    executor.submit(self._evaluate_formula, formula_expr, X, y, metadata,
                                  alpha, beta, gamma, enable_dimensional_analysis): formula_expr
                    for formula_expr in formula_expressions
                }
                
                for future in as_completed(future_to_formula):
                    formula = future.result()
                    if formula:
                        evaluated_formulas.append(formula)
        
        return evaluated_formulas
    
    def _evaluate_formula(self, formula_expr: str, X: pd.DataFrame, y: pd.Series, 
                         metadata: Dict[str, Any], alpha: float, beta: float, gamma: float, 
                         enable_dimensional_analysis: bool = False) -> Optional[Formula]:
        """Enhanced formula evaluation with complete metrics"""
        try:
            # Create function from expression
            func = self._create_function(formula_expr, X.columns)
            
            # Test the function with coefficient fitting
            y_pred = func(X, y)
            
            # Calculate comprehensive metrics
            if self.task_type == "regression":
                mae = mean_absolute_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                accuracy = 0.0  # Not applicable for regression
                precision = 0.0
                recall = 0.0
                f1 = 0.0
                cross_entropy = 0.0
                error = mae
            else:
                # For classification
                if len(np.unique(y)) == 2:
                    # Binary classification
                    y_pred_class = (y_pred > 0.5).astype(int)
                    y_pred_proba = np.clip(y_pred, 1e-15, 1-1e-15)
                else:
                    # Multi-class classification
                    y_pred_class = np.argmax(y_pred, axis=1)
                    y_pred_proba = y_pred
                
                mae = 1 - accuracy_score(y, y_pred_class)
                r2 = accuracy_score(y, y_pred_class)
                accuracy = accuracy_score(y, y_pred_class)
                precision = precision_score(y, y_pred_class, average='weighted', zero_division=0)
                recall = recall_score(y, y_pred_class, average='weighted', zero_division=0)
                f1 = f1_score(y, y_pred_class, average='weighted', zero_division=0)
                cross_entropy = log_loss(y, y_pred_proba)
                error = mae
            
            # Calculate complexity
            complexity = self._calculate_complexity(formula_expr)
            
            # Get interpretability score from LLM
            interpretability = self._get_interpretability_score(
                formula_expr, metadata, enable_dimensional_analysis
            )
            
            # Calculate normalized loss function
            error_norm = self._normalize_metric(error, 'error')
            complexity_norm = self._normalize_metric(complexity, 'complexity')
            interpretability_norm = 1 - interpretability
            
            loss = alpha * error_norm + beta * complexity_norm + gamma * interpretability_norm
            
            return Formula(
                expression=formula_expr,
                function=func,
                mae=mae,
                r2=r2,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                cross_entropy=cross_entropy,
                complexity=complexity,
                interpretability=interpretability,
                loss=loss,
                error_norm=error_norm,
                complexity_norm=complexity_norm,
                interpretability_norm=interpretability_norm,
                alpha=alpha,
                beta=beta,
                gamma=gamma
            )
            
        except Exception as e:
            print(f"Warning: Formula evaluation failed for '{formula_expr}': {e}")
            return None
    
    def _create_function(self, formula_expr: str, feature_names: List[str]) -> Callable:
        """Create a callable function from formula expression with coefficient fitting"""
        
        # Clean the expression
        formula_expr = formula_expr.strip().rstrip(',').rstrip(';')
        
        # Preprocess formula to add abs() for sqrt and log functions
        formula_expr = self._add_abs_for_safe_operations(formula_expr)
        
        # Check if formula contains coefficient 'c'
        has_coefficient = 'c' in formula_expr
        
        if has_coefficient:
            # Create a function that fits the coefficient
            def formula_function(X, y=None):
                if y is None:
                    # If no y provided, use default coefficient value of 1.0
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
                        'c': 1.0,
                        '__builtins__': {}
                    }
                    
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
                else:
                    # Fit the coefficient using linear regression
                    try:
                        # Create a function that computes the formula without 'c'
                        # Replace 'c' with a placeholder that we'll multiply later
                        formula_without_c = formula_expr.replace('c', '1.0')
                        
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
                        
                        # Add feature columns to namespace
                        local_dict = safe_dict.copy()
                        for col in feature_names:
                            local_dict[col] = X[col]
                        
                        # Compute formula without coefficient
                        base_result = eval(formula_without_c, local_dict)
                        
                        # Ensure result is a pandas Series
                        if isinstance(base_result, (int, float)):
                            base_result = pd.Series([base_result] * len(X), index=X.index)
                        elif isinstance(base_result, np.ndarray):
                            base_result = pd.Series(base_result, index=X.index)
                        
                        # Fit coefficient using linear regression
                        # y = c * base_result
                        # c = (y * base_result) / (base_result * base_result)
                        numerator = (y * base_result).sum()
                        denominator = (base_result * base_result).sum()
                        
                        if abs(denominator) > 1e-10:  # Avoid division by zero
                            fitted_c = numerator / denominator
                        else:
                            fitted_c = 1.0
                        
                        # Apply fitted coefficient
                        result = fitted_c * base_result
                        
                        # Check for NaN or infinite values
                        if isinstance(result, pd.Series):
                            if result.isna().any() or np.isinf(result).any():
                                raise ValueError("Formula produces NaN or infinite values")
                        elif np.isnan(result) or np.isinf(result):
                            raise ValueError("Formula produces NaN or infinite values")
                        
                        return result
                        
                    except Exception as e:
                        # Fallback to default coefficient
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
                            'c': 1.0,
                            '__builtins__': {}
                        }
                        
                        local_dict = safe_dict.copy()
                        for col in feature_names:
                            local_dict[col] = X[col]
                        
                        result = eval(formula_expr, local_dict)
                        
                        # Check for NaN or infinite values
                        if isinstance(result, pd.Series):
                            if result.isna().any() or np.isinf(result).any():
                                raise ValueError("Formula produces NaN or infinite values")
                        elif isinstance(result, np.ndarray):
                            if np.isnan(result).any() or np.isinf(result).any():
                                raise ValueError("Formula produces NaN or infinite values")
                        elif np.isnan(result) or np.isinf(result):
                            raise ValueError("Formula produces NaN or infinite values")
                        
                        if isinstance(result, (int, float)):
                            result = pd.Series([result] * len(X), index=X.index)
                        elif isinstance(result, np.ndarray):
                            result = pd.Series(result, index=X.index)
                        
                        return result
        else:
            # No coefficient to fit, create simple function
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
            
            def formula_function(X, y=None):
                # Add feature columns to namespace
                local_dict = safe_dict.copy()
                for col in feature_names:
                    local_dict[col] = X[col]
                
                # Evaluate the expression
                result = eval(formula_expr, local_dict)
                
                # Check for NaN or infinite values
                if isinstance(result, pd.Series):
                    if result.isna().any() or np.isinf(result).any():
                        raise ValueError("Formula produces NaN or infinite values")
                elif isinstance(result, np.ndarray):
                    if np.isnan(result).any() or np.isinf(result).any():
                        raise ValueError("Formula produces NaN or infinite values")
                elif np.isnan(result) or np.isinf(result):
                    raise ValueError("Formula produces NaN or infinite values")
                
                # Ensure result is a pandas Series
                if isinstance(result, (int, float)):
                    result = pd.Series([result] * len(X), index=X.index)
                elif isinstance(result, np.ndarray):
                    result = pd.Series(result, index=X.index)
                
                return result
        
        return formula_function
    
    def _add_abs_for_safe_operations(self, formula_expr: str) -> str:
        """Add abs() wrapper around arguments of sqrt, log, and fractional powers to prevent NaN"""
        import re
        
        # Pattern to match sqrt(expression) and add abs() around the expression
        formula_expr = re.sub(r'sqrt\s*\(\s*([^)]+)\s*\)', r'sqrt(abs(\1))', formula_expr)
        
        # Pattern to match log(expression) and add abs() around the expression
        formula_expr = re.sub(r'log\s*\(\s*([^)]+)\s*\)', r'log(abs(\1) + 1e-10)', formula_expr)
        
        # Pattern to match expression**0.5 and convert to sqrt(abs(expression))
        formula_expr = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\*\*\s*0\.5', r'sqrt(abs(\1))', formula_expr)
        
        # Pattern to match expression**1.5 and convert to expression * sqrt(abs(expression))
        formula_expr = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\*\*\s*1\.5', r'\1 * sqrt(abs(\1))', formula_expr)
        
        # Pattern to match expression**2.5 and convert to expression**2 * sqrt(abs(expression))
        formula_expr = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\*\*\s*2\.5', r'\1**2 * sqrt(abs(\1))', formula_expr)
        
        return formula_expr
    
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
    
    def _get_interpretability_score(self, formula_expr: str, metadata: Dict[str, Any], enable_dimensional_analysis: bool = False) -> float:
        """Get interpretability score from LLM"""
        from templates.prompt_templates import SymbolicRegressionPrompts
        
        # Prepare data for self-evaluation
        variables_list = list(metadata.get('feature_meanings', {}).keys())
        meanings_list = [metadata.get('feature_meanings', {}).get(var, 'Unknown') for var in variables_list]
        target_meaning = metadata.get('target_meaning', 'target property')
        
        # Get dimensional information if available
        feature_dimensions = metadata.get('feature_dimensions', {})
        target_dimension = metadata.get('target_dimension', 'unknown')
        
        if enable_dimensional_analysis and feature_dimensions and target_dimension != 'unknown':
            # Use dimensional analysis prompt
            prompt = SymbolicRegressionPrompts.formula_self_evaluation_with_dimensional_analysis_prompt(
                mathematical_functions=[formula_expr],
                meanings_list=meanings_list,
                target_meaning=target_meaning,
                feature_dimensions=feature_dimensions,
                target_dimension=target_dimension
            )
        else:
            # Use standard prompt
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
        
        # Clean the response - remove any non-numeric content except the score
        response = response.strip()
        
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
        
        # If no pattern found, try to extract any number from the response
        numbers = re.findall(r'([0-9]*\.?[0-9]+)', response)
        if numbers:
            try:
                score = float(numbers[0])
                if 0 <= score <= 1:
                    return score
                elif score <= 10:
                    return score / 10.0
            except ValueError:
                pass
        
        return 0.5  # Default if no score found
    
    def _normalize_metric(self, value: float, metric_type: str) -> float:
        """Enhanced normalization with dynamic bounds"""
        if metric_type == 'error':
            if self.metric_bounds['error_max'] > self.metric_bounds['error_min']:
                return (value - self.metric_bounds['error_min']) / (self.metric_bounds['error_max'] - self.metric_bounds['error_min'])
            else:
                return 0.5
        elif metric_type == 'complexity':
            if self.metric_bounds['complexity_max'] > self.metric_bounds['complexity_min']:
                return (value - self.metric_bounds['complexity_min']) / (self.metric_bounds['complexity_max'] - self.metric_bounds['complexity_min'])
            else:
                return 0.5
        else:
            return value
    
    def _select_best_formulas(self, formulas: List[Formula], n_best: int) -> List[Formula]:
        """Select best formulas based on loss function"""
        sorted_formulas = sorted(formulas, key=lambda f: f.loss)
        return sorted_formulas[:n_best]
    
    def _generate_new_formulas(self, best_formulas: List[Formula], X: pd.DataFrame, 
                              y: pd.Series, metadata: Dict[str, Any], n_new: int) -> List[str]:
        """Generate new formulas based on best performing ones"""
        from templates.prompt_templates import SymbolicRegressionPrompts, PromptManager
        
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

    def _generate_new_formulas_with_history(self, best_formulas: List[Formula], X: pd.DataFrame, 
                                           y: pd.Series, metadata: Dict[str, Any], n_new: int) -> List[str]:
        """Generate new formulas based on best performing ones with history"""
        from templates.prompt_templates import SymbolicRegressionPrompts, PromptManager
        
        # Format data points
        points_data = PromptManager.format_data_points(X, y)
        
        # Format previous trajectory with top formulas
        formulas_data = []
        for formula in best_formulas:
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
        
        print(f"    Iteration prompt:")
        print(f"    {'='*50}")
        print(prompt)
        print(f"    {'='*50}")
        
        try:
            response = self.llm_model.generate(prompt)
            new_formulas = PromptManager.extract_python_functions(response)
            print(f"    Generated {len(new_formulas)} formulas from iteration")
            return new_formulas[:n_new]
        except Exception as e:
            print(f"Warning: New formula generation with history failed: {e}")
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
    
    def _update_metric_bounds(self, formulas: List[Formula]):
        """Update metric bounds for normalization"""
        if not formulas:
            return
        
        errors = [f.mae for f in formulas]
        complexities = [f.complexity for f in formulas]
        
        self.metric_bounds['error_min'] = min(self.metric_bounds['error_min'], min(errors))
        self.metric_bounds['error_max'] = max(self.metric_bounds['error_max'], max(errors))
        self.metric_bounds['complexity_min'] = min(self.metric_bounds['complexity_min'], min(complexities))
        self.metric_bounds['complexity_max'] = max(self.metric_bounds['complexity_max'], max(complexities))
    
    def _check_improvement(self, current_iteration: int) -> bool:
        """Check if there's been recent improvement in formula quality"""
        if len(self.formulas) < 20:
            return True
        
        # Get best formulas from last 50 iterations
        recent_formulas = sorted(self.formulas, key=lambda f: f.loss)[:20]
        avg_loss = np.mean([f.loss for f in recent_formulas])
        
        # Compare with previous window
        if hasattr(self, '_previous_avg_loss'):
            improvement = self._previous_avg_loss - avg_loss
            self._previous_avg_loss = avg_loss
            return improvement > 0.01  # 1% improvement threshold
        else:
            self._previous_avg_loss = avg_loss
            return True

    def _enhanced_pareto_frontier_analysis(self, formulas: List[Formula]) -> List[Formula]:
        """Enhanced Pareto frontier analysis for multi-objective optimization"""
        if not formulas:
            return []
        
        # Define objectives: minimize error, minimize complexity, maximize interpretability
        objectives = []
        for formula in formulas:
            if self.task_type == "regression":
                error_obj = formula.mae
            else:
                error_obj = 1 - formula.accuracy
            
            complexity_obj = formula.complexity
            interpretability_obj = 1 - formula.interpretability  # Convert to minimization
            
            objectives.append((error_obj, complexity_obj, interpretability_obj))
        
        # Find Pareto optimal solutions
        pareto_indices = self._find_pareto_optimal(objectives)
        pareto_formulas = [formulas[i] for i in pareto_indices]
        
        # Sort by primary objective (error)
        pareto_formulas.sort(key=lambda f: f.mae if self.task_type == "regression" else (1 - f.accuracy))
        
        return pareto_formulas
    
    def _find_pareto_optimal(self, objectives: List[Tuple[float, ...]]) -> List[int]:
        """Find Pareto optimal solutions using dominance comparison"""
        n = len(objectives)
        pareto_indices = []
        
        for i in range(n):
            is_dominated = False
            for j in range(n):
                if i != j:
                    # Check if j dominates i
                    dominates = True
                    for k in range(len(objectives[i])):
                        if objectives[j][k] > objectives[i][k]:
                            dominates = False
                            break
                    
                    if dominates:
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_indices.append(i)
        
        return pareto_indices
    
    def _print_detailed_loss_analysis(self, formulas: List[Formula], alpha: float, beta: float, gamma: float):
         """Print detailed loss function analysis for formulas"""
         if not formulas:
             return
         
         # Sort formulas by loss for better readability
         sorted_formulas = sorted(formulas, key=lambda f: f.loss)
         
         print(f"Loss Function: L = {alpha}×Error + {beta}×Complexity + {gamma}×Interpretability")
         print(f"{'Formula':<50} {'Total Loss':<10} {'Error':<8} {'Complexity':<10} {'Interpretability':<15}")
         print("-" * 100)
         
         for i, formula in enumerate(sorted_formulas[:10]):  # Show top 10 formulas
             error_term = alpha * formula.error_norm
             complexity_term = beta * formula.complexity_norm
             interpretability_term = gamma * formula.interpretability_norm
             
             # Truncate formula expression for display
             expr_display = formula.expression[:47] + "..." if len(formula.expression) > 50 else formula.expression
             
             print(f"{expr_display:<50} {formula.loss:<10.4f} {error_term:<8.4f} {complexity_term:<10.4f} {interpretability_term:<15.4f}")
         
         if len(formulas) > 10:
             print(f"... and {len(formulas) - 10} more formulas")
         
         # Summary statistics
         losses = [f.loss for f in formulas]
         error_terms = [alpha * f.error_norm for f in formulas]
         complexity_terms = [beta * f.complexity_norm for f in formulas]
         interpretability_terms = [gamma * f.interpretability_norm for f in formulas]
         
         print(f"\nSummary Statistics:")
         print(f"Average Total Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
         print(f"Average Error Term: {np.mean(error_terms):.4f} ± {np.std(error_terms):.4f}")
         print(f"Average Complexity Term: {np.mean(complexity_terms):.4f} ± {np.std(complexity_terms):.4f}")
         print(f"Average Interpretability Term: {np.mean(interpretability_terms):.4f} ± {np.std(interpretability_terms):.4f}")
         print(f"Best Total Loss: {min(losses):.4f}")
         print(f"Worst Total Loss: {max(losses):.4f}")
