# LLM-formula

This repository packages the LLM-Feynman symbolic-regression toolkit together with a ready-to-run Gradio GUI.

## Repository Layout
- `llm_feynman/` – Python package containing the full LLM-Feynman pipeline (preprocessing, feature engineering, symbolic regression, interpretation, and model backends).
- `main.py` – Re-exports the `LLMFeynman` class for legacy imports (`from main import LLMFeynman`).
- `gradio_gui.py` – Rich GUI implementation.
- `launch_gui.py` / `simple_launch.py` – Convenience launchers for the GUI.
- `GUI_README.md`, `GUI_SUCCESS.md` – GUI usage notes and success stories.
- `requirements_gui.txt` – Minimal dependencies needed to run the GUI.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_gui.txt
```

## Quick Start (Python API)
```python
import pandas as pd
from llm_feynman import LLMFeynman

data = pd.read_csv("your_dataset.csv")
X = data.drop(columns=["target"])
y = data["target"]

llm = LLMFeynman(model_type="huggingface", model_name="meta-llama/Llama-3.3-8b")
formulas = llm.discover_formulas(X=X, y=y, include_interpretation=True)

llm.print_formula_summary()
best = llm.get_best_formula(metric="r2")
print(best.expression, best.r2)
```

## Launch the GUI
```bash
python launch_gui.py
```
The launcher checks dependencies, imports the `llm_feynman` package, and starts the Gradio interface on `http://127.0.0.1:7860`. Use `simple_launch.py` if you prefer a minimal CLI output.

## Notes
- The old flat module layout (`core/`, `models/`, etc.) has been consolidated under `llm_feynman/`. Update your imports accordingly (for example, `from llm_feynman.core import DataPreprocessor`).
- `main.py` remains as a thin wrapper for downstream codebases that previously imported `LLMFeynman` from the repository root.
- If you add additional entry points or scripts, keep them alongside the existing launch helpers to avoid duplicating the package source.
