# llm_feynman/utils/plotting.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

def plot_results(results: Dict[str, Any], save_path: Optional[str] = None):
    """Plot comprehensive results from LLM-Feynman discovery"""
    
    formulas = results.get('formulas', [])
    if not formulas:
        print("No formulas to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LLM-Feynman Discovery Results', fontsize=16, fontweight='bold')
    
    # Extract metrics
    r2_scores = [f.r2 for f in formulas]
    complexities = [f.complexity for f in formulas]
    interpretability_scores = [f.interpretability for f in formulas]
    loss_values = [f.loss for f in formulas]
    
    # Plot 1: R² vs Complexity (Pareto frontier)
    axes[0, 0].scatter(complexities, r2_scores, alpha=0.7, s=60)
    axes[0, 0].set_xlabel('Complexity')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('Accuracy vs Complexity\n(Pareto Frontier)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Interpretability distribution
    axes[0, 1].hist(interpretability_scores, bins=15, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Interpretability Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Interpretability Score Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Loss vs R²
    axes[1, 0].scatter(loss_values, r2_scores, alpha=0.7, s=60)
    axes[1, 0].set_xlabel('Loss Value')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_title('Loss vs Accuracy')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Top formulas comparison
    top_formulas = sorted(formulas, key=lambda f: f.r2, reverse=True)[:5]
    formula_names = [f"Formula {i+1}" for i in range(len(top_formulas))]
    r2_top = [f.r2 for f in top_formulas]
    
    bars = axes[1, 1].bar(formula_names, r2_top, alpha=0.7)
    axes[1, 1].set_xlabel('Top Formulas')
    axes[1, 1].set_ylabel('R² Score')
    axes[1, 1].set_title('Top 5 Formulas by Accuracy')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, r2 in zip(bars, r2_top):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{r2:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results plot saved to {save_path}")
    
    plt.show()
    
    # Print top formulas
    print("\n" + "="*50)
    print("TOP 5 DISCOVERED FORMULAS")
    print("="*50)
    
    for i, formula in enumerate(top_formulas, 1):
        print(f"\n{i}. Formula: {formula.expression}")
        print(f"   R² Score: {formula.r2:.4f}")
        print(f"   MAE: {formula.mae:.4f}")
        print(f"   Complexity: {formula.complexity}")
        print(f"   Interpretability: {formula.interpretability:.3f}")
        print(f"   Loss: {formula.loss:.4f}")
