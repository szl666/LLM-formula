<<<<<<< HEAD
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

=======
__version__ = "0.1.0"
__author__ = "Zhilong Song"


from .llm_feynman.main import LLMFeynman
from .llm_feynman.core import (
    DataPreprocessor,
    FeatureEngineer,
    SymbolicRegressor,
    FormulaInterpreter
)

__all__ = [
    "LLMFeynman",
    "DataPreprocessor", 
    "FeatureEngineer",
    "SymbolicRegressor",
    "FormulaInterpreter"
]

>>>>>>> 10f4e4c (Initial import of core LLM-Feynman modules)
