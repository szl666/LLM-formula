"""
Convenience wrapper that exposes the LLMFeynman interface from the
``llm_feynman`` package. Import ``LLMFeynman`` from here if you rely on the
previous flat project layout.
"""

from llm_feynman.main import LLMFeynman  # re-export for backward compatibility

__all__ = ["LLMFeynman"]


if __name__ == "__main__":
    print("Import LLMFeynman from this module: from main import LLMFeynman")
