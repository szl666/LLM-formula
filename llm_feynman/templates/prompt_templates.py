
from typing import Dict, List, Any, Optional

class FeatureEngineeringPrompts:
    """Prompt templates for feature engineering"""
    
    @staticmethod
    def feature_recommendation_prompt(target_property: str, feature_columns: Optional[List[str]] = None, target_column: Optional[str] = None) -> str:
        """Generate prompt for automatic feature recommendation with data structure context"""
        
        if feature_columns and target_column:
            # 如果提供了具体的数据结构信息
            feature_list = ", ".join(feature_columns)
            data_context = f"""
DATA STRUCTURE CONTEXT:
- Target column (what we want to predict): {target_column} (represents: {target_property})
- Feature columns (input variables for prediction): {feature_list}
- Total features: {len(feature_columns)}

You need to provide physical meanings and units specifically for these {len(feature_columns)} feature columns: {feature_list}
"""
            requirements = f"""Requirements:
Provide physical meanings and units for exactly these {len(feature_columns)} feature columns: {feature_list}
Match the exact order of feature columns provided above.
Follow the specified formatting for meanings and units."""
        else:
            # 原始的通用模式
            data_context = ""
            requirements = """Requirements:
List at least 20 relevant material descriptors.
Ensure descriptors cover various material aspects (e.g., structure, composition).
Follow the specified formatting for descriptors, meanings, and units."""
        
        return f"""You are a large language model-driven knowledge extraction agent. Based on the target material property {target_property}, recommend relevant material descriptors.{data_context}

Ensure that the recommended descriptors cover different aspects of the material, such as structure, composition, electronic properties, thermodynamic properties, etc. Additionally, format your response as follows:

First Line: List the descriptors separated by commas.
Second Line: Provide the physical meaning of each descriptor, separated by commas.
Third Line: Specify the units for each descriptor, separated by commas.

Subsequent Lines: For each descriptor, include: - A brief explanation of why it is related to the target property.

{requirements}"""

    @staticmethod
    def feature_name_matching_prompt(material_descriptors: List[str], 
                                   matminer_features: List[str]) -> str:
        """Generate prompt for automatic feature name matching (Figure S3)"""
        
        descriptors_str = ", ".join(material_descriptors)
        matminer_str = ", ".join(matminer_features)
        
        return f"""{descriptors_str}

Given the above list of material descriptors and their physical meanings, identify the corresponding feature names from the Matminer feature library.

{matminer_str}

Instructions:
1. Match Each Descriptor: For each material descriptor provided, find the most appropriate feature name from the Matminer feature library that corresponds to its physical meaning.
2. Handle No Matches: If a descriptor does not have a direct match in the Matminer library, indicate "No match" for that descriptor.
3. Output Format: Provide the matched feature names in a single line, separated by commas, maintaining the order of the input descriptors.
4. Use Only Matminer Features: Ensure that only feature names from the Matminer feature library are used in the output."""

    @staticmethod
    def additional_feature_recommendation_prompt(existing_descriptors: List[str], 
                                               performance_metrics: Dict[str, float]) -> str:
        """Generate prompt for additional feature recommendation (Figure S4)"""
        
        descriptors_str = ", ".join(existing_descriptors)
        performance_str = ", ".join([f"{k}: {v:.4f}" for k, v in performance_metrics.items()])
        
        return f"""{descriptors_str}: {performance_str}

Based on the above list of existing material descriptors and their corresponding performance metrics, recommend additional material features that could potentially improve the descriptor performance. Ensure that the recommended features cover different aspects of the material properties, such as structure, composition, electronic properties, thermodynamic properties, etc.

Instructions:
1. Analyze Existing Features: Review the provided list of existing descriptors and their performance metrics to identify potential gaps or areas lacking sufficient representation.
2. Recommend New Features: Suggest at least 10 additional material features from the Matminer feature library that are likely to capture complementary information and improve overall performance.
3. Additionally, format your response as follows:

First Line: List the descriptors separated by commas.
Second Line: Provide the physical meaning of each descriptor, separated by commas.
Third Line: Specify the units for each descriptor, separated by commas."""

class SymbolicRegressionPrompts:
    """Prompt templates for symbolic regression"""
    
    @staticmethod
    def initial_formula_generation_prompt(points_data: str, target_meaning: str, 
                                        variables_list: List[str], meanings_list: List[str],
                                        num_variables: int, num_formulas: int = 5) -> str:
        """Generate prompt for initial formula generation (Figure S5)"""
        
        variables_str = ", ".join(variables_list)
        meanings_str = ", ".join(meanings_list)
        
        return f"""I want you to act as a mathematical formula generator.

DATA STRUCTURE:
Each data point is formatted as: ({variables_str}, target)
- Feature variables (inputs): {variables_str}
- Target variable (output to predict): {target_meaning}

Data points:
{points_data}

FEATURE AND TARGET MEANINGS AND DIMENSIONS:
{chr(10).join(f"- {var}: {meaning}" for var, meaning in zip(variables_list, meanings_list))}
- Target: {target_meaning}

Your task is to give me EXACTLY {num_formulas} new potential formulas that can accurately model the relationship between the input variables and the target.

CRITICAL OUTPUT REQUIREMENTS:
- Output EXACTLY {num_formulas} formulas, no more, no less
- Output ONLY the formulas, nothing else
- Do not include any explanations, comments, or additional text
- Do not include any markdown formatting
- Do not include any code blocks or ```python``` tags
- Output each formula on a separate line
- Use the exact format: def f(x1,x2,...): return formula

Your options are:
- Independent variable symbols: {variables_str}
- A coefficient symbol: c (there is no need to write a number - write this generic coefficient instead).
- Basic operators: +, -, *, /, **, sqrt, exp, log, abs

Remember that the formulas you generate should always have at most {num_variables} variables {variables_str}.

The functions should have parametric form, using "c" in place of any constant or coefficient. The coefficients will be optimized to fit the data. Make absolutely sure that the formulas you generate are completely different from the ones already given to you.

The formulas should be a python function in the format of:
def f(x1,x2,...)
    return formula

Remember that you can combine the simple building blocks (operations, constants, variables) in any way you want to generate accurate and simple formulas.

IMPORTANT: Use Python syntax for mathematical operations:
- Use ** for exponentiation (e.g., x**2, not x^2)
- Use sqrt() for square root (e.g., sqrt(x), not √x)
- Use exp() for exponential (e.g., exp(x), not e^x)
- Use log() for natural logarithm (e.g., log(x), not ln(x))

IMPORTANT: For safe mathematical operations:
- Use abs() around arguments of sqrt() to ensure positive values (e.g., sqrt(abs(x)))
- Use abs() around arguments of log() to ensure positive values (e.g., log(abs(x) + 1e-10))
- Use abs() for fractional powers to avoid complex numbers (e.g., x**1.5 becomes x * sqrt(abs(x)))

The formulas obtained from the combination of variables need to have physical dimensions as consistent as possible with the target y.

Don't be afraid to experiment!

Output EXACTLY {num_formulas} formulas:"""

    @staticmethod
    def formula_self_evaluation_prompt(mathematical_functions: List[str], 
                                     meanings_list: List[str], target_meaning: str,
                                     previous_trajectory: Optional[str] = None) -> str:
        """Generate prompt for formula self-evaluation (Figure S6)"""
        
        functions_str = "\n".join(mathematical_functions)
        meanings_str = ", ".join(meanings_list)
        
        trajectory_section = f"""
You will also be provided with previous formulas and their associated errors:
{previous_trajectory}

Note that errors are listed in order of fit values, where lower values indicate better performance.
""" if previous_trajectory else ""
        
        return f"""Meanings of the variables: {meanings_str}

I want you to act as a mathematical formula evaluator with expertise in physical chemistry.
Your task is to analyze the provided input mathematical formulas {{mathematical_functions}} using domain-specific criteria to assess whether they align with established physical chemistry principles and demonstrate strong extrapolation capabilities with respect to the target {target_meaning}.

To perform the evaluation, you should:
- Assess whether the formulas are consistent with physical chemistry principles, including concepts such as energy conservation, thermodynamic laws, and reaction kinetics.
- Evaluate the extrapolation capabilities of the formulas based on their physical chemistry relevance and potential to generalize beyond the given data range.
- Ensure that the formulas exhibit physical dimensional consistency with the target variable.
{trajectory_section}

CRITICAL OUTPUT REQUIREMENTS:
- Output ONLY a single score between 0.0 and 1.0
- Do not include any explanations, comments, or additional text
- Do not include any markdown formatting
- Do not include any code blocks
- Output only the numerical score

Mathematical functions to evaluate:
{functions_str}

Score:"""

    @staticmethod
    def formula_self_evaluation_with_dimensional_analysis_prompt(mathematical_functions: List[str], 
                                                              meanings_list: List[str], target_meaning: str,
                                                              feature_dimensions: Dict[str, str], target_dimension: str,
                                                              previous_trajectory: Optional[str] = None) -> str:
        """Generate prompt for formula self-evaluation with dimensional analysis"""
        
        functions_str = "\n".join(mathematical_functions)
        meanings_str = ", ".join(meanings_list)
        
        # Create dimensional information string
        dimensions_info = []
        for var, meaning in zip(meanings_list, feature_dimensions.keys()):
            dim = feature_dimensions.get(meaning, 'unknown')
            dimensions_info.append(f"{var}: {dim}")
        dimensions_str = ", ".join(dimensions_info)
        
        trajectory_section = f"""
You will also be provided with previous formulas and their associated errors:
{previous_trajectory}

Note that errors are listed in order of fit values, where lower values indicate better performance.
""" if previous_trajectory else ""
        
        return f"""Meanings of the variables: {meanings_str}
Dimensions of the variables: {dimensions_str}
Target dimension: {target_dimension}

I want you to act as a mathematical formula evaluator with expertise in physical chemistry and dimensional analysis.
Your task is to analyze the provided input mathematical formulas {{mathematical_functions}} using domain-specific criteria to assess whether they align with established physical chemistry principles and demonstrate strong extrapolation capabilities with respect to the target {target_meaning}.

To perform the evaluation, you should:
- Assess whether the formulas are consistent with physical chemistry principles, including concepts such as energy conservation, thermodynamic laws, and reaction kinetics.
- Evaluate the extrapolation capabilities of the formulas based on their physical chemistry relevance and potential to generalize beyond the given data range.
- CRITICAL: Perform detailed dimensional analysis to check if the formula's dimensional units match the target dimension {target_dimension}.
- If the formula's dimensional units do not match the target dimension, assign a significantly lower score (0.1-0.3).
- If the formula's dimensional units match the target dimension, consider it favorably in scoring.
- Ensure that the formulas exhibit physical dimensional consistency with the target variable.
{trajectory_section}

CRITICAL OUTPUT REQUIREMENTS:
- Output ONLY a single score between 0.0 and 1.0
- Do not include any explanations, comments, or additional text
- Do not include any markdown formatting
- Do not include any code blocks
- Output only the numerical score

Mathematical functions to evaluate:
{functions_str}

Score:"""

    @staticmethod
    def iterative_formula_generation_prompt(points_data: str, previous_trajectory: str,
                                          target_meaning: str, variables_list: List[str],
                                          meanings_list: List[str], num_variables: int,
                                          num_formulas: int = 5) -> str:
        """Generate prompt for iterative formula generation (Figure S7)"""
        
        variables_str = ", ".join(variables_list)
        meanings_str = ", ".join(meanings_list)
        
        return f"""I want you to act as a mathematical formula generator.

DATA STRUCTURE:
Each data point is formatted as: ({variables_str}, target)
- Feature variables (inputs): {variables_str}
- Target variable (output to predict): {target_meaning}

Data points:
{points_data}

FEATURE AND TARGET MEANINGS AND DIMENSIONS:
{chr(10).join(f"- {var}: {meaning}" for var, meaning in zip(variables_list, meanings_list))}
- Target: {target_meaning}

Below are some previous formulas and their error and self-evaluation scores(alignment with physical chemistry principles and capability of extrapolation) of them, which are arranged in order of their fit values, with the highest values coming first, and lower is better.
{previous_trajectory}

Your task is to give me EXACTLY {num_formulas} new potential formulas that are different from all the ones reported below, and have a lower error value than all of the functions below.

CRITICAL OUTPUT REQUIREMENTS:
- Output EXACTLY {num_formulas} formulas, no more, no less
- Output ONLY the formulas, nothing else
- Do not include any explanations, comments, or additional text
- Do not include any markdown formatting
- Do not include any code blocks or ```python``` tags
- Output each formula on a separate line
- Use the exact format: def f(x1,x2,...): return formula

Your options are:
- Independent variable symbols: {variables_str}
- A coefficient symbol: c (there is no need to write a number - write this generic coefficient instead).
- Basic operators: +, -, *, /, **, sqrt, exp, log, abs

Remember that the formulas you generate should always have at most {num_variables} variables {variables_str}.

The functions should have parametric form, using "c" in place of any constant or coefficient. The coefficients will be optimized to fit the data. Make absolutely sure that the formulas you generate are completely different from the ones already given to you.

The formulas should be a python function in the format of:
def f(x1,x2,...)
    return formula

Remember that you can combine the simple building blocks (operations, constants, variables) in any way you want to generate accurate and simple formulas.

IMPORTANT: For safe mathematical operations:
- Use abs() around arguments of sqrt() to ensure positive values (e.g., sqrt(abs(x)))
- Use abs() around arguments of log() to ensure positive values (e.g., log(abs(x) + 1e-10))
- Use abs() for fractional powers to avoid complex numbers (e.g., x**1.5 becomes x * sqrt(abs(x)))

The formulas obtained from the combination of variables need to have physical dimensions as consistent as possible with the target y.

Don't be afraid to experiment!

Output EXACTLY {num_formulas} formulas:"""

class InterpretationPrompts:
    """Prompt templates for formula interpretation using MCTS"""
    
    @staticmethod
    def initial_interpretation_prompt(formula: str, feature_meanings: Dict[str, str],
                                    target_meaning: str, feature_dimensions: Dict[str, str],
                                    target_dimension: str) -> str:
        """Generate initial interpretation prompt"""
        
        features_info = "\n".join([
            f"- {feature}: {feature_meanings.get(feature, 'Unknown')} [{feature_dimensions.get(feature, 'Unknown')}]"
            for feature in feature_meanings.keys()
        ])
        
        return f"""
You are a scientific expert providing physical interpretation of mathematical formulas.

Formula: {formula}
Target: {target_meaning} [{target_dimension}]

Available features:
{features_info}

Please provide a comprehensive physical interpretation of this formula, explaining:
1. The underlying physical mechanisms
2. How each term contributes to the target property
3. The physical meaning of mathematical operations
4. Why this relationship makes sense scientifically

Focus on clear, scientifically accurate explanations that connect the mathematics to physical reality.

Interpretation:
"""
    
    @staticmethod
    def refined_interpretation_prompt(formula: str, parent_interpretation: str,
                                    refinement_aspect: str, feature_meanings: Dict[str, str],
                                    target_meaning: str) -> str:
        """Generate refined interpretation prompt"""
        
        aspect_prompts = {
            "physical_mechanisms": "Focus on the detailed physical mechanisms and processes",
            "dimensional_analysis": "Perform detailed dimensional analysis and unit consistency",
            "limiting_cases": "Analyze limiting cases and boundary conditions",
            "parameter_sensitivity": "Examine parameter sensitivity and scaling relationships",
            "theoretical_foundations": "Connect to established theoretical foundations and principles"
        }
        
        aspect_instruction = aspect_prompts.get(refinement_aspect, "Provide additional insights")
        
        return f"""
Building upon the previous interpretation, provide a refined analysis of this formula.

Formula: {formula}
Target: {target_meaning}

Previous interpretation:
{parent_interpretation}

Refinement focus: {aspect_instruction}

Provide a refined interpretation that builds upon the previous analysis while focusing specifically on the {refinement_aspect} aspect. Add new insights and deeper understanding.

Refined interpretation:
"""
    
    @staticmethod
    def interpretation_quality_prompt(formula: str, interpretation: str,
                                    feature_meanings: Dict[str, str], target_meaning: str) -> str:
        """Generate interpretation quality evaluation prompt"""
        
        return f"""
Evaluate the quality of this scientific interpretation on a scale from 0.0 to 1.0.

Formula: {formula}
Target: {target_meaning}

Interpretation to evaluate:
{interpretation}

Evaluation criteria:
- Physical accuracy and correctness (0.3 weight)
- Clarity and comprehensiveness (0.2 weight)  
- Scientific depth and insight (0.2 weight)
- Connection between math and physics (0.2 weight)
- Practical relevance and applicability (0.1 weight)

Provide a score from 0.0 (poor) to 1.0 (excellent) and brief reasoning.

Quality score: [0.0-1.0]
Reasoning: [brief explanation]
"""
    
    @staticmethod
    def synthesis_prompt(formula: str, interpretations: List[str],
                        feature_meanings: Dict[str, str], target_meaning: str,
                        formula_metrics: Dict[str, float]) -> str:
        """Generate synthesis prompt for final interpretation"""
        
        interpretations_text = "\n\n".join([
            f"Interpretation {i+1}:\n{interp}" 
            for i, interp in enumerate(interpretations)
        ])
        
        return f"""
Synthesize the following interpretations into a comprehensive, unified explanation.

Formula: {formula}
Target: {target_meaning}
Performance: R² = {formula_metrics['r2']:.3f}, MAE = {formula_metrics['mae']:.3f}, Complexity = {formula_metrics['complexity']}

Multiple interpretations:
{interpretations_text}

Create a unified, comprehensive interpretation that:
1. Combines the best insights from all interpretations
2. Provides a coherent physical explanation
3. Addresses the formula's accuracy and complexity
4. Offers practical insights for understanding and application
5. Maintains scientific rigor and clarity

Unified interpretation:
"""

class PromptManager:
    """Utility class for managing and formatting prompts"""
    
    @staticmethod
    def format_data_points(X, y, max_points: int = 100):
        """Format data points for prompt inclusion"""
        if len(X) > max_points:
            # Sample points for prompt
            indices = np.random.choice(len(X), max_points, replace=False)
            X_sample = X.iloc[indices]
            y_sample = y.iloc[indices]
        else:
            X_sample = X
            y_sample = y
        
        points_list = []
        for i in range(len(X_sample)):
            x_vals = [f"{X_sample.iloc[i][col]:.4f}" for col in X_sample.columns]
            points_list.append(f"({', '.join(x_vals)}, {y_sample.iloc[i]:.4f})")
        
        return "\n".join(points_list)
    
    @staticmethod
    def format_previous_trajectory(formulas_data: List[Dict[str, Any]], max_formulas: int = 10):
        """Format previous formulas trajectory for prompt"""
        if not formulas_data:
            return ""
        
        # Sort by error (ascending) and take best performing ones
        sorted_formulas = sorted(formulas_data, key=lambda x: x.get('error', float('inf')))[:max_formulas]
        
        trajectory_lines = []
        for i, formula_data in enumerate(sorted_formulas):
            error = formula_data.get('error', 'N/A')
            interpretability = formula_data.get('interpretability', 'N/A')
            expression = formula_data.get('expression', 'N/A')
            
            trajectory_lines.append(
                f"{i+1}. {expression} | Error: {error:.4f} | Interpretability: {interpretability:.3f}"
            )
        
        return "\n".join(trajectory_lines)
    
    @staticmethod
    def clean_llm_response(response: str) -> str:
        """Clean and format LLM response"""
        # Remove common LLM artifacts
        response = response.strip()
        response = response.replace("assistant:", "").replace("user:", "").replace("system:", "")
        
        # Remove multiple newlines
        import re
        response = re.sub(r'\n+', '\n', response)
        
        return response.strip()
    
    @staticmethod
    def extract_python_functions(response: str) -> List[str]:
        """Extract Python function definitions from LLM response"""
        import re
        
        # Pattern to match function definitions
        function_pattern = r'def\s+\w+\([^)]*\):\s*\n\s*return\s+([^\n]+)'
        
        matches = re.findall(function_pattern, response, re.MULTILINE)
        
        # Also look for direct return statements
        return_pattern = r'return\s+([^\n]+)'
        return_matches = re.findall(return_pattern, response)
        
        # Combine and clean
        all_formulas = matches + return_matches
        
        # Clean up formulas
        cleaned_formulas = []
        for formula in all_formulas:
            formula = formula.strip().rstrip(',').rstrip(';')
            if formula and formula not in cleaned_formulas:
                cleaned_formulas.append(formula)
        
        return cleaned_formulas
