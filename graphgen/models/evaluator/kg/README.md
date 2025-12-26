# KG Quality Evaluation Module

This module provides comprehensive quality evaluation for knowledge graphs built by GraphGen.

## Module Structure

The evaluation functionality is organized into modular components:

- **`accuracy_evaluator.py`**: Entity/relation extraction quality evaluation using LLM-as-a-Judge
- **`consistency_evaluator.py`**: Attribute value conflict detection
- **`structure_evaluator.py`**: Graph structural robustness metrics

The evaluation components are integrated in `graphgen/operators/evaluate/evaluate_kg.py`, which provides functions to create and use these evaluators.

## Features

### 1. Accuracy Assessment
- **Entity Extraction Quality**: Uses LLM-as-a-Judge to evaluate the quality of entity extraction from chunks
  - Evaluates accuracy (correctness of extracted entities)
  - Evaluates completeness (whether important entities are missed)
  - Evaluates precision (naming accuracy and specificity)
- **Relation Extraction Quality**: Uses LLM-as-a-Judge to evaluate the quality of relation extraction from chunks
  - Evaluates accuracy (correctness of extracted relations)
  - Evaluates completeness (whether important relations are missed)
  - Evaluates precision (relation description accuracy)
- Provides multi-dimensional quality scores (0-1 scale) with detailed reasoning for each chunk

### 2. Consistency Assessment
- **Semantic Conflict Detection**: Uses LLM-as-a-Judge to detect semantic conflicts in entity attributes
  - **Entity Type Conflicts**: Detects when the same entity is extracted with different types across chunks
  - **Entity Description Conflicts**: Detects when entity descriptions from different chunks are semantically inconsistent
  - **Relation Conflicts**: Detects when the same entity pair has conflicting relation descriptions
- Only evaluates entities with multiple source chunks (entities appearing in multiple chunks)
- Uses LLM to extract entity attributes from each chunk and compare them semantically
- Calculates conflict rate: `conflict_entities_count / total_entities`
- Returns detailed conflict information including conflict severity and reasoning

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

# Specify backends
python -m graphgen.operators.evaluate_kg.evaluate_kg \
    --working_dir cache \
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
    --accuracy_only
```

## Configuration

All evaluation thresholds use default values defined in the evaluator classes:

- **Structure thresholds**: Defined in `StructureEvaluator` with defaults:
  - `noise_ratio_threshold`: 0.15
  - `largest_cc_ratio_threshold`: 0.90
  - `avg_degree_min`: 2.0
  - `avg_degree_max`: 5.0
  - `powerlaw_r2_threshold`: 0.75

**Note**: Accuracy evaluation automatically loads chunks from the chunk storage and evaluates the quality of entity/relation extraction using LLM-as-a-Judge. No configuration file is needed.

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
            "overall_score": {
                "mean": float,
                "median": float,
                "min": float,
                "max": float,
                "std": float
            },
            "accuracy": {
                "mean": float,
                "median": float,
                "min": float,
                "max": float,
                "std": float
            },
            "completeness": {
                "mean": float,
                "median": float,
                "min": float,
                "max": float,
                "std": float
            },
            "precision": {
                "mean": float,
                "median": float,
                "min": float,
                "max": float,
                "std": float
            },
            "total_chunks": int,
            "detailed_results": [
                {
                    "chunk_id": str,
                    "chunk_content": str,
                    "extracted_entities_count": int,
                    "accuracy": float,
                    "completeness": float,
                    "precision": float,
                    "overall_score": float,
                    "accuracy_reasoning": str,
                    "completeness_reasoning": str,
                    "precision_reasoning": str,
                    "issues": [str]
                },
                ...
            ]
        },
        "relation_accuracy": {
            "overall_score": {
                "mean": float,
                "median": float,
                "min": float,
                "max": float,
                "std": float
            },
            "accuracy": {
                "mean": float,
                "median": float,
                "min": float,
                "max": float,
                "std": float
            },
            "completeness": {
                "mean": float,
                "median": float,
                "min": float,
                "max": float,
                "std": float
            },
            "precision": {
                "mean": float,
                "median": float,
                "min": float,
                "max": float,
                "std": float
            },
            "total_chunks": int,
            "detailed_results": [
                {
                    "chunk_id": str,
                    "chunk_content": str,
                    "extracted_relations_count": int,
                    "accuracy": float,
                    "completeness": float,
                    "precision": float,
                    "overall_score": float,
                    "accuracy_reasoning": str,
                    "completeness_reasoning": str,
                    "precision_reasoning": str,
                    "issues": [str]
                },
                ...
            ]
        }
    },
    "consistency": {
        "conflict_rate": float,
        "conflict_entities_count": int,
        "total_entities": int,
        "entities_checked": int,
        "conflicts": [
            {
                "entity_id": str,
                "conflict_type": str,  # "entity_type" or "description"
                "conflict_severity": float,  # 0-1, severity of the conflict
                "conflict_reasoning": str,
                "conflicting_values": [str],
                "recommended_value": str,  # for entity_type conflicts
                "conflict_details": str  # for description conflicts
            },
            ...
        ]
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

- Accuracy evaluation uses LLM-as-a-Judge to evaluate extraction quality from chunks
- Accuracy evaluation automatically loads chunks from chunk storage (no need for source_text_paths)
- The evaluator associates extracted entities/relations with their source chunks using the `source_id` field
- Structural evaluation automatically converts Kuzu storage to NetworkX for analysis
- All evaluations include error handling and will return error messages if something fails
- The evaluator automatically loads graph and chunk storage from the working directory
- LLM evaluation may take time for large numbers of chunks (controlled by `max_concurrent` parameter)
