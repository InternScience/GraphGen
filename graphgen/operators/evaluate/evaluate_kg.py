from typing import Any, Dict

from dotenv import load_dotenv

from graphgen.models import KGQualityEvaluator
from graphgen.utils import logger

# Load environment variables
load_dotenv()


def evaluate_accuracy(evaluator: KGQualityEvaluator) -> Dict[str, Any]:
    """Evaluate accuracy of entity and relation extraction.
    
    Args:
        evaluator: KGQualityEvaluator instance
        
    Returns:
        Dictionary containing entity_accuracy and relation_accuracy metrics.
    """
    logger.info("Running accuracy evaluation...")
    results = evaluator.evaluate_accuracy()
    logger.info("Accuracy evaluation completed")
    return results


def evaluate_consistency(evaluator: KGQualityEvaluator) -> Dict[str, Any]:
    """Evaluate consistency by detecting semantic conflicts.
    
    Args:
        evaluator: KGQualityEvaluator instance
        
    Returns:
        Dictionary containing consistency metrics including conflict_rate and conflicts.
    """
    logger.info("Running consistency evaluation...")
    results = evaluator.evaluate_consistency()
    logger.info("Consistency evaluation completed")
    return results


def evaluate_structure(evaluator: KGQualityEvaluator) -> Dict[str, Any]:
    """Evaluate structural robustness of the graph.
    
    Args:
        evaluator: KGQualityEvaluator instance
        
    Returns:
        Dictionary containing structural metrics including noise_ratio, largest_cc_ratio, etc.
    """
    logger.info("Running structural robustness evaluation...")
    results = evaluator.evaluate_structure()
    logger.info("Structural robustness evaluation completed")
    return results


def evaluate_all(evaluator: KGQualityEvaluator) -> Dict[str, Any]:
    """Run all evaluations (accuracy, consistency, structure).
    
    Args:
        evaluator: KGQualityEvaluator instance
        
    Returns:
        Dictionary containing all evaluation results with keys: accuracy, consistency, structure.
    """
    logger.info("Running all evaluations...")
    results = evaluator.evaluate_all()
    logger.info("All evaluations completed")
    return results


