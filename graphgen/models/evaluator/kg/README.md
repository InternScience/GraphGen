# KG Quality Evaluation Module

This module provides comprehensive quality evaluation for knowledge graphs built by GraphGen.

## Module Structure

The evaluation functionality has been split into modular components:

- **`accuracy_evaluator.py`**: Entity/relation/triple accuracy evaluation using LLM-as-judge
- **`consistency_evaluator.py`**: Attribute value conflict detection
- **`structure_evaluator.py`**: Graph structural robustness metrics
- **`utils.py`**: Utility functions (NetworkX conversion, text retrieval, sampling)
- **`kg_quality_evaluator.py`**: Main evaluator class that integrates all modules

## Features

### 1. Accuracy Assessment
- **Entity Recognition Accuracy**: Samples entities and validates them using LLM
- **Relation Extraction Accuracy**: Samples relations and validates them using LLM
- **Triple Validation (RLC)**: Samples triples and validates them using LLM
- Calculates Precision, Recall, and F1 scores for each metric

### 2. Consistency Assessment
- Detects attribute value conflicts (same entity, same attribute, different values)
- Calculates conflict rate: `conflict_entities_count / total_entities`
- Returns detailed conflict information

### 3. Structural Robustness Assessment
- **Noise Ratio**: Isolated nodes / total nodes (threshold: < 15%)
- **Largest Connected Component Ratio**: Largest CC nodes / total nodes (threshold: > 90%)
- **Average Node Degree**: Average degree across all nodes (threshold: 2-5)
- **Power Law Distribution R²**: Degree distribution fit (threshold: > 0.75)

## Usage

### Command Line Usage

```bash
# Run all evaluations
python -m graphgen.operators.evaluate_kg.evaluate_kg --working_dir cache

# Run specific evaluation
python -m graphgen.operators.evaluate_kg.evaluate_kg --working_dir cache --accuracy_only

# Custom configuration
python -m graphgen.operators.evaluate_kg.evaluate_kg \
    --working_dir cache \
    --sample_size 200 \
    --graph_backend networkx \
    --kv_backend json_kv
```

### Shell Script Usage

```bash
# Basic usage
bash examples/evaluate_kg/evaluate_kg.sh

# With custom options
bash examples/evaluate_kg/evaluate_kg.sh \
    --working_dir cache \
    --sample_size 200 \
    --accuracy_only
```

## Requirements

- **NetworkX**: Required for structural evaluation
- **scipy**: Required for power law distribution fitting
- **numpy**: Required for numerical calculations
- **LLM Client**: Required for accuracy evaluation (configure via `TRAINEE_*` env vars)

## Output Format

The evaluation returns a dictionary with the following structure:

```python
{
    "accuracy": {
        "entity_accuracy": {
            "precision": float,
            "recall": float,
            "f1": float,
            "true_positives": int,
            "false_positives": int,
            "sample_size": int
        },
        "relation_accuracy": { ... },
        "triple_accuracy": { ... }
    },
    "consistency": {
        "conflict_rate": float,
        "conflict_entities_count": int,
        "total_entities": int,
        "conflicts": [ ... ]
    },
    "structure": {
        "total_nodes": int,
        "total_edges": int,
        "noise_ratio": float,
        "largest_cc_ratio": float,
        "avg_degree": float,
        "powerlaw_r2": float | None,
        "thresholds": {
            "noise_ratio": { "value": float, "threshold": float, "pass": bool },
            ...
        }
    }
}
```

## Notes

- Accuracy evaluation requires LLM API access and may be slow for large sample sizes
- Structural evaluation automatically converts Kuzu storage to NetworkX for analysis
- All evaluations include error handling and will return error messages if something fails
- The evaluator automatically loads graph and chunk storage from the working directory
