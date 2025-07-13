# LLM-formula

A Python library for automated discovery of mathematical formulas from scientific data using Large Language Models (LLMs), implementing the methodology from the LLM-Feynman paper.

## Overview

LLM-Feynman combines the power of large language models with symbolic regression to automatically discover interpretable mathematical formulas from experimental data. The library consists of three main modules:

1. **Module I**: Data preprocessing and LLM-guided feature engineering
2. **Module II**: Symbolic regression with LLM-based formula generation and self-evaluation
3. **Module III**: Monte Carlo Tree Search (MCTS) based formula interpretation

## Features

- 🤖 **LLM-Powered Discovery**: Leverages large language models for intelligent formula generation
- 🔬 **Scientific Focus**: Designed specifically for scientific and materials discovery
- 📊 **Automated Feature Engineering**: LLM-guided feature recommendation and selection
- 🎯 **Self-Evaluation**: Built-in formula quality assessment using domain knowledge
- 🌳 **MCTS Interpretation**: Advanced interpretation system using Monte Carlo Tree Search
- 📈 **Pareto Optimization**: Multi-objective optimization balancing accuracy and simplicity
- 🔧 **Flexible Backend**: Support for multiple LLM providers (HuggingFace, OpenAI)

## Quick Start

```python
import pandas as pd
from llm_feynman import LLMFeynman

# Load your data
X = pd.DataFrame({
    'temperature': [300, 400, 500, 600],
    'pressure': [1.0, 1.5, 2.0, 2.5],
    'composition': [0.1, 0.2, 0.3, 0.4]
})
y = pd.Series([10.5, 15.2, 22.1, 30.8])

# Initialize LLM-Feynman
llm_feynman = LLMFeynman(
    model_type="huggingface",
    model_name="meta-llama/Llama-3.3-8b"
)

# Discover formulas with physical interpretations
formulas = llm_feynman.discover_formulas(
    X=X, 
    y=y,
    feature_meanings={
        'temperature': 'Temperature',
        'pressure': 'Pressure', 
        'composition': 'Composition ratio'
    },
    feature_dimensions={
        'temperature': 'K',
        'pressure': 'atm',
        'composition': 'dimensionless'
    },
    target_meaning='Reaction rate',
    target_dimension='mol/s',
    include_interpretation=True
)

# View results
llm_feynman.print_formula_summary()

# Get the best formula
best_formula = llm_feynman.get_best_formula(metric="r2")
print(f"Best formula: {best_formula.expression}")
print(f"R² score: {best_formula.r2:.4f}")

# Get physical interpretation
interpretation = llm_feynman.get_formula_interpretation()
print(f"Physical meaning: {interpretation}")
```

## Advanced Usage

### Custom Model Configuration

```python
# Using local HuggingFace model with custom settings
llm_feynman = LLMFeynman(
    model_type="huggingface",
    model_name="model_name",
    device="cuda",
    torch_dtype="float16"
)
```

### Detailed Configuration

```python
# Custom preprocessing configuration
preprocessing_config = {
    'remove_outliers': True,
    'outlier_method': 'iqr',
    'normalize_features': True,
    'handle_missing': 'interpolate'
}

# Custom feature engineering configuration
feature_engineering_config = {
    'max_new_features': 20,
    'feature_selection_method': 'correlation',
    'use_llm_suggestions': True,
    'polynomial_degree': 2
}

# Custom symbolic regression configuration
symbolic_regression_config = {
    'max_iterations': 50,
    'population_size': 100,
    'max_formula_length': 15,
    'use_interpretability_filter': True
}

# Run with custom configurations
formulas = llm_feynman.discover_formulas(
    X=X, y=y,
    preprocessing_config=preprocessing_config,
    feature_engineering_config=feature_engineering_config,
    symbolic_regression_config=symbolic_regression_config
)
```

### Working with Results

```python
# Compare all discovered formulas
comparison_df = llm_feynman.compare_formulas()
print(comparison_df)

# Get Pareto frontier (accuracy vs complexity)
pareto_formulas = llm_feynman.get_pareto_front()

# Validate on new data
validation_metrics = llm_feynman.validate_formula(
    formula=best_formula,
    X_val=X_test,
    y_val=y_test
)

# Export results
llm_feynman.export_formulas(
    filepath="discovered_formulas.json",
    format="json",
    include_interpretations=True
)

# Save complete results for later analysis
llm_feynman.save_results("experiment_results.pkl")
```

### Interpretation-Only Mode

```python
# If you already have formulas and want interpretations
from llm_feynman.core.symbolic_regression import Formula

existing_formulas = [
    Formula(expression="c1 * temperature + c2 * pressure", ...),
    Formula(expression="c1 * exp(c2 / temperature)", ...)
]

interpretations = llm_feynman.interpret_formulas(existing_formulas)
```

## API Reference

### Main Classes

- **`LLMFeynman`**: Main interface for the discovery pipeline
- **`DataPreprocessor`**: Handles data cleaning and preprocessing
- **`FeatureEngineer`**: LLM-guided feature engineering
- **`SymbolicRegressor`**: Formula discovery with self-evaluation
- **`FormulaInterpreter`**: MCTS-based interpretation generation

### Key Methods

- **`discover_formulas()`**: Complete discovery pipeline
- **`interpret_formulas()`**: Generate physical interpretations
- **`get_best_formula()`**: Retrieve optimal formula by metric
- **`compare_formulas()`**: Compare discovered formulas
- **`export_formulas()`**: Export results to file
