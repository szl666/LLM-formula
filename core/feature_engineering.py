# llm_feynman/core/feature_engineering.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

class FeatureEngineer:
    """Feature engineering module for LLM-Feynman"""
    
    def __init__(self, llm_model, task_type: str = "regression"):
        """
        Args:
            llm_model: LLM model for feature suggestions
            task_type: Type of task ("regression" or "classification")
        """
        self.llm_model = llm_model
        self.task_type = task_type
        self.selected_features = []
        self.engineered_features = []
        
    def engineer_features(self, X: pd.DataFrame, y: pd.Series, 
                         metadata: Dict[str, Any],
                         mutual_info_threshold: float = 0.1,
                         max_features: int = 50) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Perform comprehensive feature engineering
        
        Args:
            X: Feature matrix
            y: Target values
            metadata: Data metadata
            mutual_info_threshold: Threshold for mutual information feature selection
            max_features: Maximum number of features to keep
            
        Returns:
            Engineered feature matrix and updated metadata
        """
        X_engineered = X.copy()
        
        # Step 1: Feature selection via mutual information
        X_engineered, selected_features = self._mutual_info_selection(
            X_engineered, y, mutual_info_threshold
        )
        metadata['selected_features'] = selected_features
        
        # Step 2: LLM-guided feature matching
        suggested_features = self._llm_feature_suggestions(X_engineered, y, metadata)
        X_engineered = self._compute_suggested_features(X_engineered, suggested_features)
        metadata['suggested_features'] = suggested_features
        
        # Step 3: Limit total features
        if len(X_engineered.columns) > max_features:
            # Re-apply mutual information selection to limit features
            X_engineered, _ = self._mutual_info_selection(
                X_engineered, y, threshold=0.0, max_features=max_features
            )
        
        metadata['final_features'] = list(X_engineered.columns)
        metadata['feature_engineering_steps'] = [
            "Mutual information selection",
            "LLM-guided feature suggestions",
            "Feature computation and integration"
        ]
        
        return X_engineered, metadata
    
    def _mutual_info_selection(self, X: pd.DataFrame, y: pd.Series, 
                              threshold: float, max_features: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
        """Select features based on mutual information"""
        # Calculate mutual information
        if self.task_type == "regression":
            mi_scores = mutual_info_regression(X, y)
        else:
            # For classification, encode target if needed
            if y.dtype == 'object':
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
            else:
                y_encoded = y
            mi_scores = mutual_info_classif(X, y_encoded)
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        # Select features above threshold
        selected_features = feature_importance[
            feature_importance['mi_score'] >= threshold
        ]['feature'].tolist()
        
        # Limit number of features if specified
        if max_features and len(selected_features) > max_features:
            selected_features = selected_features[:max_features]
        
        return X[selected_features], selected_features
    
    def _llm_feature_suggestions(self, X: pd.DataFrame, y: pd.Series, 
                                metadata: Dict[str, Any]) -> List[Dict[str, str]]:
        """Get feature suggestions from LLM"""
        from ..templates.prompt_templates import FeatureEngineeringPrompts
        
        target_meaning = metadata.get('target_meaning', 'unknown property')
        
        # Use the official LLM-Feynman prompt template
        prompt = FeatureEngineeringPrompts.feature_recommendation_prompt(target_meaning)
        
        try:
            response = self.llm_model.generate(prompt)
            # Parse LLM response to extract feature suggestions
            suggestions = self._parse_feature_recommendations(response)
            return suggestions
        except Exception as e:
            print(f"Warning: LLM feature suggestion failed: {e}")
            return []
    
    def _parse_feature_recommendations(self, response: str) -> List[Dict[str, str]]:
        """Parse LLM response for feature recommendations (following S2 format)"""
        lines = response.strip().split('\n')
        suggestions = []
        
        if len(lines) >= 3:
            # Parse the structured response format
            descriptors = [desc.strip() for desc in lines[0].split(',')]
            meanings = [meaning.strip() for meaning in lines[1].split(',')]
            units = [unit.strip() for unit in lines[2].split(',')]
            
            # Ensure all lists have the same length
            min_length = min(len(descriptors), len(meanings), len(units))
            
            for i in range(min_length):
                suggestions.append({
                    'name': descriptors[i],
                    'meaning': meanings[i],
                    'dimension': units[i],
                    'formula': f"computed_{descriptors[i].lower().replace(' ', '_')}"  # Placeholder
                })
        
        return suggestions
    
    def _compute_suggested_features(self, X: pd.DataFrame, 
                                   suggestions: List[Dict[str, str]]) -> pd.DataFrame:
        """Compute suggested features and add to dataframe"""
        X_extended = X.copy()
        
        for suggestion in suggestions:
            try:
                feature_name = suggestion.get('name', '')
                formula = suggestion.get('formula', '')
                
                if feature_name and formula:
                    # Safely evaluate the formula
                    feature_values = self._safe_eval_formula(formula, X)
                    if feature_values is not None:
                        X_extended[feature_name] = feature_values
                        self.engineered_features.append(feature_name)
            except Exception as e:
                print(f"Warning: Failed to compute feature {feature_name}: {e}")
        
        return X_extended
    
    def _safe_eval_formula(self, formula: str, X: pd.DataFrame) -> Optional[pd.Series]:
        """Safely evaluate a formula string"""
        try:
            # Create a safe namespace with numpy functions and dataframe columns
            namespace = {
                'np': np,
                'log': np.log,
                'sqrt': np.sqrt,
                'exp': np.exp,
                'sin': np.sin,
                'cos': np.cos,
                'tan': np.tan,
                'abs': np.abs,
                'pow': np.power,
                **{col: X[col] for col in X.columns}
            }
            
            # Evaluate the formula
            result = eval(formula, {"__builtins__": {}}, namespace)
            
            # Convert to pandas Series if needed
            if isinstance(result, (int, float)):
                result = pd.Series([result] * len(X), index=X.index)
            elif isinstance(result, np.ndarray):
                result = pd.Series(result, index=X.index)
            elif not isinstance(result, pd.Series):
                return None
            
            # Check for invalid values
            if result.isnull().all() or np.isinf(result).any():
                return None
            
            return result
            
        except Exception:
            return None
    
    def iterative_refinement(self, X: pd.DataFrame, y: pd.Series, 
                           metadata: Dict[str, Any], 
                           stagnation_threshold: int = 10) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Perform iterative feature refinement if needed"""
        # This method can be called if formula generation stagnates
        # For now, it's a placeholder for future enhancement
        
        additional_features = self._generate_interaction_features(X)
        X_refined = pd.concat([X, additional_features], axis=1)
        
        metadata['refinement_features'] = list(additional_features.columns)
        
        return X_refined, metadata
    
    def _generate_interaction_features(self, X: pd.DataFrame, max_interactions: int = 10) -> pd.DataFrame:
        """Generate interaction features between existing features"""
        interaction_features = pd.DataFrame(index=X.index)
        
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        
        count = 0
        for i, col1 in enumerate(numeric_columns):
            for col2 in numeric_columns[i+1:]:
                if count >= max_interactions:
                    break
                
                # Multiplication interaction
                feature_name = f"{col1}_x_{col2}"
                interaction_features[feature_name] = X[col1] * X[col2]
                count += 1
                
                if count >= max_interactions:
                    break
            
            if count >= max_interactions:
                break
        
        return interaction_features
