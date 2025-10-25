
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    """Data preprocessing module for LLM-Feynman"""
    
    def __init__(self, normalization: str = "standard", handle_missing: str = "mean"):
        """
        Args:
            normalization: Type of normalization ("standard", "minmax", "none")
            handle_missing: How to handle missing values ("mean", "median", "drop")
        """
        self.normalization = normalization
        self.handle_missing = handle_missing
        self.scaler = None
        self.imputer = None
        
    def preprocess(self, X: pd.DataFrame, y: pd.Series, 
                  feature_meanings: Optional[Dict[str, str]] = None,
                  feature_dimensions: Optional[Dict[str, str]] = None,
                  target_meaning: Optional[str] = None,
                  target_dimension: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Preprocess input data
        
        Args:
            X: Feature matrix
            y: Target values
            feature_meanings: Physical meanings of features
            feature_dimensions: Dimensions of features
            target_meaning: Physical meaning of target
            target_dimension: Dimension of target
            
        Returns:
            Preprocessed X, y, and metadata
        """
        metadata = {
            'feature_meanings': feature_meanings or {},
            'feature_dimensions': feature_dimensions or {},
            'target_meaning': target_meaning or "",
            'target_dimension': target_dimension or "",
            'original_features': list(X.columns),
            'preprocessing_steps': []
        }
        
        # Handle missing values
        X_processed, y_processed = self._handle_missing_values(X.copy(), y.copy())
        metadata['preprocessing_steps'].append(f"Missing values handled: {self.handle_missing}")
        
        # Normalize features
        if self.normalization != "none":
            X_processed = self._normalize_features(X_processed)
            metadata['preprocessing_steps'].append(f"Normalization: {self.normalization}")
        
        return X_processed, y_processed, metadata
    
    def _handle_missing_values(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle missing values in the data"""
        if self.handle_missing == "drop":
            # Drop rows with any missing values
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            return X[mask], y[mask]
        else:
            # Impute missing values
            strategy = "mean" if self.handle_missing == "mean" else "median"
            self.imputer = SimpleImputer(strategy=strategy)
            
            # Impute features
            X_imputed = pd.DataFrame(
                self.imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            # Impute target if needed
            if y.isnull().any():
                y_imputed = pd.Series(
                    SimpleImputer(strategy=strategy).fit_transform(y.values.reshape(-1, 1)).flatten(),
                    index=y.index
                )
            else:
                y_imputed = y
                
            return X_imputed, y_imputed
    
    def _normalize_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Normalize features"""
        if self.normalization == "standard":
            self.scaler = StandardScaler()
        elif self.normalization == "minmax":
            self.scaler = MinMaxScaler()
        else:
            return X
        
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        return X_scaled
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform normalized features"""
        if self.scaler is not None:
            return pd.DataFrame(
                self.scaler.inverse_transform(X),
                columns=X.columns,
                index=X.index
            )
        return X
