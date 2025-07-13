
import numpy as np
import pandas as pd
import random
import math
import copy
import time
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import re
from .symbolic_regression import Formula

@dataclass
class InterpretationNode:
    """Node in the MCTS tree for formula interpretation"""
    formula: Formula
    interpretation: str
    parent: Optional['InterpretationNode'] = None
    children: List['InterpretationNode'] = None
    visits: int = 0
    value: float = 0.0
    depth: int 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def is_fully_expanded(self, max_children: int = 5) -> bool:
        return len(self.children) >= max_children
    
    def add_child(self, child: 'InterpretationNode'):
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)
    
    def get_ucb_score(self, exploration_weight: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

class TheoryDistiller:
    """Monte Carlo Tree Search based formula interpretation"""
    
    def __init__(self, llm_model, max_depth: int = 5, max_iterations: int = 1000,
                 exploration_weight: float = 1.414, parallel_threads: int = 4):
        """
        Args:
            llm_model: LLM model for generating interpretations
            max_depth: Maximum depth of MCTS tree
            max_iterations: Maximum MCTS iterations
            exploration_weight: UCB exploration parameter
            parallel_threads: Number of parallel threads for evaluation
        """
        self.llm_model = llm_model
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.exploration_weight = exploration_weight
        self.parallel_threads = parallel_threads
        
    def interpret_formula(self, formula: Formula, X: pd.DataFrame, y: pd.Series,
                         metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive interpretation for a formula using MCTS
        
        Args:
            formula: Formula to interpret
            X: Feature data
            y: Target data
            metadata: Additional context
            
        Returns:
            Interpretation results
        """
        print(f"Interpreting formula: {formula.expression}")
        
        # Initialize root node
        initial_interpretation = self._generate_initial_interpretation(formula, metadata)
        root = InterpretationNode(
            formula=formula,
            interpretation=initial_interpretation,
            depth=0
        )
        
        # Run MCTS
        best_interpretations = self._run_mcts(root, X, y, metadata)
        
        # Generate final comprehensive interpretation
        final_interpretation = self._synthesize_interpretations(
            formula, best_interpretations, X, y, metadata
        )
        
        return {
            'formula': formula,
            'interpretations': best_interpretations,
            'final_interpretation': final_interpretation,
            'mcts_tree_size': self._count_nodes(root),
            'search_depth': self._get_max_depth(root)
        }
    
    def _run_mcts(self, root: InterpretationNode, X: pd.DataFrame, 
                  y: pd.Series, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run Monte Carlo Tree Search for interpretation"""
        
        for iteration in range(self.max_iterations):
            if iteration % 100 == 0:
                print(f"MCTS iteration {iteration}/{self.max_iterations}")
            
            # Selection
            node = self._select(root)
            
            # Expansion
            if not node.is_fully_expanded() and node.depth < self.max_depth:
                child = self._expand(node, metadata)
                if child:
                    node = child
            
            # Simulation
            value = self._simulate(node, X, y, metadata)
            
            # Backpropagation
            self._backpropagate(node, value)
        
        # Extract best interpretations
        return self._extract_best_interpretations(root)
    
    def _select(self, node: InterpretationNode) -> InterpretationNode:
        """Select best node using UCB"""
        current = node
        
        while not current.is_leaf() and current.is_fully_expanded():
            # Select child with highest UCB score
            current = max(current.children, key=lambda c: c.get_ucb_score(self.exploration_weight))
        
        return current
    
    def _expand(self, node: InterpretationNode, metadata: Dict[str, Any]) -> Optional[InterpretationNode]:
        """Expand node by generating new interpretation"""
        try:
            # Generate new interpretation based on parent
            new_interpretation = self._generate_refined_interpretation(
                node.formula, node.interpretation, metadata, node.depth
            )
            
            if new_interpretation and new_interpretation != node.interpretation:
                child = InterpretationNode(
                    formula=node.formula,
                    interpretation=new_interpretation
                )
                node.add_child(child)
                return child
        except Exception as e:
            print(f"Warning: Expansion failed: {e}")
        
        return None
    
    def _simulate(self, node: InterpretationNode, X: pd.DataFrame, 
                  y: pd.Series, metadata: Dict[str, Any]) -> float:
        """Simulate and evaluate interpretation quality"""
        try:
            # Evaluate interpretation quality
            quality_score = self._evaluate_interpretation_quality(
                node.interpretation, node.formula, X, y, metadata
            )
            
            # Add some randomness for exploration
            noise = random.uniform(-0.1, 0.1)
            return max(0.0, min(1.0, quality_score + noise))
            
        except Exception as e:
            print(f"Warning: Simulation failed: {e}")
            return 0.0
    
    def _backpropagate(self, node: InterpretationNode, value: float):
        """Backpropagate value up the tree"""
        current = node
        while current is not None:
            current.visits += 1
            current.value += value
            current = current.parent
    
    def _generate_initial_interpretation(self, formula: Formula, 
                                       metadata: Dict[str, Any]) -> str:
        """Generate initial interpretation for the formula"""
        from ..templates.prompt_templates import InterpretationPrompts
        
        prompt = InterpretationPrompts.initial_interpretation_prompt(
            formula=formula.expression,
            feature_meanings=metadata.get('feature_meanings', {}),
            target_meaning=metadata.get('target_meaning', ''),
            feature_dimensions=metadata.get('feature_dimensions', {}),
            target_dimension=metadata.get('target_dimension', '')
        )
        
        try:
            response = self.llm_model.generate(prompt, max_new_tokens=512)
            return self._clean_interpretation(response)
        except Exception as e:
            print(f"Warning: Initial interpretation generation failed: {e}")
            return f"Mathematical relationship: {formula.expression}"
    
    def _generate_refined_interpretation(self, formula: Formula, parent_interpretation: str,
                                       metadata: Dict[str, Any], depth: int) -> str:
        """Generate refined interpretation based on parent"""
        from ..templates.prompt_templates import InterpretationPrompts
        
        refinement_aspects = [
            "physical_mechanisms",
            "dimensional_analysis", 
            "limiting_cases",
            "parameter_sensitivity",
            "theoretical_foundations"
        ]
        
        # Choose refinement aspect based on depth
        aspect = refinement_aspects[depth % len(refinement_aspects)]
        
        prompt = InterpretationPrompts.refined_interpretation_prompt(
            formula=formula.expression,
            parent_interpretation=parent_interpretation,
            refinement_aspect=aspect,
            feature_meanings=metadata.get('feature_meanings', {}),
            target_meaning=metadata.get('target_meaning', '')
        )
        
        try:
            response = self.llm_model.generate(prompt, max_new_tokens=512)
            return self._clean_interpretation(response)
        except Exception as e:
            print(f"Warning: Refined interpretation generation failed: {e}")
            return parent_interpretation
    
    def _evaluate_interpretation_quality(self, interpretation: str, formula: Formula,
                                       X: pd.DataFrame, y: pd.Series, 
                                       metadata: Dict[str, Any]) -> float:
        """Evaluate quality of interpretation"""
        try:
            # Use LLM to score interpretation quality
            from ..templates.prompt_templates import InterpretationPrompts
            
            prompt = InterpretationPrompts.interpretation_quality_prompt(
                formula=formula.expression,
                interpretation=interpretation,
                feature_meanings=metadata.get('feature_meanings', {}),
                target_meaning=metadata.get('target_meaning', '')
            )
            
            response = self.llm_model.generate(prompt, max_new_tokens=256)
            score = self._extract_quality_score(response)
            
            return score
        except Exception as e:
            print(f"Warning: Quality evaluation failed: {e}")
            return 0.5
    
    def _extract_quality_score(self, response: str) -> float:
        """Extract numerical quality score from LLM response"""
        import re
        
        # Look for score patterns
        patterns = [
            r'score[:\s]*([0-9]*\.?[0-9]+)',
            r'quality[:\s]*([0-9]*\.?[0-9]+)',
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
                    elif score <= 10:  # 0-10 scale
                        return score / 10.0
                except ValueError:
                    continue
        
        # Default score based on interpretation length and keywords
        quality_keywords = ['physical', 'mechanism', 'relationship', 'principle', 'theory']
        keyword_count = sum(1 for keyword in quality_keywords if keyword in response.lower())
        
        base_score = min(0.8, len(response) / 500)  # Length-based score
        keyword_bonus = min(0.2, keyword_count * 0.05)  # Keyword bonus
        
        return base_score + keyword_bonus
    
    def _extract_best_interpretations(self, root: InterpretationNode) -> List[Dict[str, Any]]:
        """Extract best interpretations from MCTS tree"""
        all_nodes = []
        self._collect_all_nodes(root, all_nodes)
        
        # Sort by average value (quality)
        scored_nodes = []
        for node in all_nodes:
            if node.visits > 0:
                avg_value = node.value / node.visits
                scored_nodes.append({
                    'interpretation': node.interpretation,
                    'quality_score': avg_value,
                    'visits': node.visits,
                    'depth': node.depth
                })
        
        # Return top interpretations
        scored_nodes.sort(key=lambda x: x['quality_score'], reverse=True)
        return scored_nodes[:5]  # Top 5 interpretations
    
    def _synthesize_interpretations(self, formula: Formula, best_interpretations: List[Dict[str, Any]],
                                  X: pd.DataFrame, y: pd.Series, metadata: Dict[str, Any]) -> str:
        """Synthesize final comprehensive interpretation"""
        from ..templates.prompt_templates import InterpretationPrompts
        
        top_interpretations = [interp['interpretation'] for interp in best_interpretations[:3]]
        
        prompt = InterpretationPrompts.synthesis_prompt(
            formula=formula.expression,
            interpretations=top_interpretations,
            feature_meanings=metadata.get('feature_meanings', {}),
            target_meaning=metadata.get('target_meaning', ''),
            formula_metrics={
                'r2': formula.r2,
                'mae': formula.mae,
                'complexity': formula.complexity
            }
        )
        
        try:
            response = self.llm_model.generate(prompt, max_new_tokens=1024)
            return self._clean_interpretation(response)
        except Exception as e:
            print(f"Warning: Synthesis failed: {e}")
            # Fallback: combine top interpretations
            return "\n\n".join([f"Aspect {i+1}: {interp}" 
                               for i, interp in enumerate(top_interpretations)])
    
    def _clean_interpretation(self, text: str) -> str:
        """Clean and format interpretation text"""
        # Remove common LLM artifacts
        text = text.strip()
        text = re.sub(r'^(assistant|user|system)[:\s]*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\n+', '\n', text)
        return text
    
    def _collect_all_nodes(self, node: InterpretationNode, all_nodes: List[InterpretationNode]):
        """Collect all nodes in the tree"""
        all_nodes.append(node)
        for child in node.children:
            self._collect_all_nodes(child, all_nodes)
    
    def _count_nodes(self, root: InterpretationNode) -> int:
        """Count total nodes in tree"""
        count = 1
        for child in root.children:
            count += self._count_nodes(child)
        return count
    
    def _get_max_depth(self, root: InterpretationNode) -> int:
        """Get maximum depth of tree"""
        if not root.children:
            return root.depth
        return max(self._get_max_depth(child) for child in root.children)

class FormulaInterpreter:
    """Main interface for formula interpretation"""
    
    def __init__(self, llm_model):
        self.llm_model = llm_model
        self.theory_distiller = TheoryDistiller(llm_model)
    
    def interpret_formulas(self, formulas: List[Formula], X: pd.DataFrame, 
                          y: pd.Series, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpret multiple formulas
        
        Args:
            formulas: List of formulas to interpret
            X: Feature data
            y: Target data
            metadata: Additional context
            
        Returns:
            Interpretation results for all formulas
        """
        print(f"Interpreting {len(formulas)} formulas using MCTS...")
        
        interpretations = {}
        
        for i, formula in enumerate(formulas):
            print(f"\nInterpreting formula {i+1}/{len(formulas)}")
            
            try:
                interpretation_result = self.theory_distiller.interpret_formula(
                    formula, X, y, metadata
                )
                interpretations[f"formula_{i}"] = interpretation_result
                
            except Exception as e:
                print(f"Warning: Interpretation failed for formula {i}: {e}")
                interpretations[f"formula_{i}"] = {
                    'formula': formula,
                    'interpretations': [],
                    'final_interpretation': f"Unable to interpret: {formula.expression}",
                    'error': str(e)
                }
        
        return {
            'individual_interpretations': interpretations,
            'summary': self._create_interpretation_summary(interpretations)
        }
    
    def _create_interpretation_summary(self, interpretations: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of all interpretations"""
        total_formulas = len(interpretations)
        successful_interpretations = sum(1 for result in interpretations.values() 
                                       if 'error' not in result)
        
        # Find most interpretable formula
        best_interpretable = None
        best_score = -1
        
        for key, result in interpretations.items():
            if 'error' not in result and result['interpretations']:
                avg_quality = np.mean([interp['quality_score'] 
                                     for interp in result['interpretations']])
                if avg_quality > best_score:
                    best_score = avg_quality
                    best_interpretable = result
        
        return {
            'total_formulas': total_formulas,
            'successful_interpretations': successful_interpretations,
            'success_rate': successful_interpretations / total_formulas,
            'most_interpretable_formula': best_interpretable,
            'best_interpretation_quality': best_score
        }
