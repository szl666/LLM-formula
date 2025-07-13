__version__ = "0.1.0"
__author__ = "Zhilong Song"


from .main import LLMFeynman
from .core import (
    DataPreprocessor,
    FeatureEngineer,
    SymbolicRegressor,
    FormulaInterpreter,
    TheoryDistiller
)

__all__ = [
    "LLMFeynman",
    "DataPreprocessor", 
    "FeatureEngineer",
    "SymbolicRegressor",
    "FormulaInterpreter",
    "TheoryDistiller"
]

