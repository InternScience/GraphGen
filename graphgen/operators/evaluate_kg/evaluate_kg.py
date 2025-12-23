import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

from graphgen.models import KGQualityEvaluator
from graphgen.utils import CURRENT_LOGGER_VAR, logger, set_logger

# Load environment variables
load_dotenv()


def _run_evaluation(evaluator, args):
    """Run the evaluation based on arguments."""
    if args.accuracy_only:
        logger.info("Running accuracy evaluation only...")
        return {"accuracy": evaluator.evaluate_accuracy()}
    if args.consistency_only:
        logger.info("Running consistency evaluation only...")
        return {"consistency": evaluator.evaluate_consistency()}
    if args.structure_only:
        logger.info("Running structural robustness evaluation only...")
        return {"structure": evaluator.evaluate_structure()}
    logger.info("Running all evaluations...")
    return evaluator.evaluate_all()


def _print_accuracy_summary(acc):
    """Print accuracy evaluation summary."""
    if "error" not in acc:
        print("\n[Accuracy]")
        if "entity_accuracy" in acc:
            e = acc["entity_accuracy"]
            overall = e.get("overall_score", {})
            accuracy = e.get("accuracy", {})
            completeness = e.get("completeness", {})
            precision = e.get("precision", {})
            
            print(f"  Entity Extraction Quality:")
            print(f"    Overall Score: {overall.get('mean', 0):.3f} (mean), "
                  f"{overall.get('median', 0):.3f} (median)")
            print(f"    Accuracy: {accuracy.get('mean', 0):.3f} (mean), "
                  f"{accuracy.get('median', 0):.3f} (median)")
            print(f"    Completeness: {completeness.get('mean', 0):.3f} (mean), "
                  f"{completeness.get('median', 0):.3f} (median)")
            print(f"    Precision: {precision.get('mean', 0):.3f} (mean), "
                  f"{precision.get('median', 0):.3f} (median)")
            print(f"    Total Chunks Evaluated: {e.get('total_chunks', 0)}")
            
        if "relation_accuracy" in acc:
            r = acc["relation_accuracy"]
            overall = r.get("overall_score", {})
            accuracy = r.get("accuracy", {})
            completeness = r.get("completeness", {})
            precision = r.get("precision", {})
            
            print(f"  Relation Extraction Quality:")
            print(f"    Overall Score: {overall.get('mean', 0):.3f} (mean), "
                  f"{overall.get('median', 0):.3f} (median)")
            print(f"    Accuracy: {accuracy.get('mean', 0):.3f} (mean), "
                  f"{accuracy.get('median', 0):.3f} (median)")
            print(f"    Completeness: {completeness.get('mean', 0):.3f} (mean), "
                  f"{completeness.get('median', 0):.3f} (median)")
            print(f"    Precision: {precision.get('mean', 0):.3f} (mean), "
                  f"{precision.get('median', 0):.3f} (median)")
            print(f"    Total Chunks Evaluated: {r.get('total_chunks', 0)}")
    else:
        print(f"\n[Accuracy] Error: {acc['error']}")


def _print_consistency_summary(cons):
    """Print consistency evaluation summary."""
    if "error" not in cons:
        print("\n[Consistency]")
        print(f"  Conflict Rate: {cons.get('conflict_rate', 0):.3f}")
        print(f"  Conflict Entities: {cons.get('conflict_entities_count', 0)} / "
              f"{cons.get('total_entities', 0)}")
        entities_checked = cons.get('entities_checked', 0)
        if entities_checked > 0:
            print(f"  Entities Checked: {entities_checked} (entities with multiple sources)")
        conflicts = cons.get('conflicts', [])
        if conflicts:
            print(f"  Total Conflicts Found: {len(conflicts)}")
            # Show sample conflicts
            sample_conflicts = conflicts[:3]
            for conflict in sample_conflicts:
                print(f"    - {conflict.get('entity_id', 'N/A')}: {conflict.get('conflict_type', 'N/A')} "
                      f"(severity: {conflict.get('conflict_severity', 0):.2f})")
    else:
        print(f"\n[Consistency] Error: {cons['error']}")


def _print_structure_summary(struct):
    """Print structural robustness evaluation summary."""
    if "error" not in struct:
        print("\n[Structural Robustness]")
        print(f"  Total Nodes: {struct.get('total_nodes', 0)}")
        print(f"  Total Edges: {struct.get('total_edges', 0)}")
        
        thresholds = struct.get("thresholds", {})
        
        # Noise Ratio
        noise_check = thresholds.get("noise_ratio", {})
        noise_threshold = noise_check.get("threshold", "N/A")
        noise_pass = noise_check.get("pass", False)
        print(f"  Noise Ratio: {struct.get('noise_ratio', 0):.3f} "
              f"({'✓' if noise_pass else '✗'} < {noise_threshold})")
        
        # Largest CC Ratio
        lcc_check = thresholds.get("largest_cc_ratio", {})
        lcc_threshold = lcc_check.get("threshold", "N/A")
        lcc_pass = lcc_check.get("pass", False)
        print(f"  Largest CC Ratio: {struct.get('largest_cc_ratio', 0):.3f} "
              f"({'✓' if lcc_pass else '✗'} > {lcc_threshold})")
        
        # Avg Degree
        avg_degree_check = thresholds.get("avg_degree", {})
        avg_degree_threshold = avg_degree_check.get("threshold", "N/A")
        avg_degree_pass = avg_degree_check.get("pass", False)
        # Format threshold for display (handle tuple case)
        if isinstance(avg_degree_threshold, tuple):
            threshold_str = f"{avg_degree_threshold[0]}-{avg_degree_threshold[1]}"
        else:
            threshold_str = str(avg_degree_threshold)
        print(f"  Avg Degree: {struct.get('avg_degree', 0):.2f} "
              f"({'✓' if avg_degree_pass else '✗'} {threshold_str})")
        
        # Power Law R²
        if struct.get('powerlaw_r2') is not None:
            powerlaw_check = thresholds.get("powerlaw_r2", {})
            powerlaw_threshold = powerlaw_check.get("threshold", "N/A")
            powerlaw_pass = powerlaw_check.get("pass", False)
            print(f"  Power Law R²: {struct.get('powerlaw_r2', 0):.3f} "
                  f"({'✓' if powerlaw_pass else '✗'} > {powerlaw_threshold})")
    else:
        print(f"\n[Structural Robustness] Error: {struct['error']}")


def _print_summary(results):
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print("KG Quality Evaluation Summary")
    print("=" * 60)

    if "accuracy" in results:
        _print_accuracy_summary(results["accuracy"])
    if "consistency" in results:
        _print_consistency_summary(results["consistency"])
    if "structure" in results:
        _print_structure_summary(results["structure"])

    print("\n" + "=" * 60)


def main():
    """Main function to run KG quality evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate knowledge graph quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python -m graphgen.operators.evaluate_kg.evaluate_kg --working_dir cache

  # Custom output
  python -m graphgen.operators.evaluate_kg.evaluate_kg \\
    --working_dir cache \\
    --output cache/kg_evaluation.json

  # Specify backends
  python -m graphgen.operators.evaluate_kg.evaluate_kg \\
    --working_dir cache \\
    --graph_backend networkx \\
    --kv_backend json_kv
        """,
    )

    parser.add_argument(
        "--working_dir",
        type=str,
        default="cache",
        help="Working directory containing graph and chunk storage (default: cache)",
    )
    parser.add_argument(
        "--graph_backend",
        type=str,
        default="kuzu",
        choices=["kuzu", "networkx"],
        help="Graph storage backend (default: kuzu)",
    )
    parser.add_argument(
        "--kv_backend",
        type=str,
        default="rocksdb",
        choices=["rocksdb", "json_kv"],
        help="KV storage backend (default: rocksdb)",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=10,
        help="Maximum concurrent LLM requests (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for evaluation results (default: working_dir/kg_evaluation.json)",
    )
    parser.add_argument(
        "--accuracy_only",
        action="store_true",
        help="Only run accuracy evaluation",
    )
    parser.add_argument(
        "--consistency_only",
        action="store_true",
        help="Only run consistency evaluation",
    )
    parser.add_argument(
        "--structure_only",
        action="store_true",
        help="Only run structural robustness evaluation",
    )

    args = parser.parse_args()

    # Set up logging
    log_dir = Path(args.working_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    default_logger = set_logger(str(log_dir / "evaluate_kg.log"), name="evaluate_kg")
    CURRENT_LOGGER_VAR.set(default_logger)

    # Determine output path
    if args.output is None:
        output_path = Path(args.working_dir) / "kg_evaluation.json"
    else:
        output_path = Path(args.output)

    logger.info("Starting KG quality evaluation...")
    logger.info(f"Working directory: {args.working_dir}")
    logger.info(f"Graph backend: {args.graph_backend}")
    logger.info(f"KV backend: {args.kv_backend}")

    try:
        evaluator = KGQualityEvaluator(
            working_dir=args.working_dir,
            graph_backend=args.graph_backend,
            kv_backend=args.kv_backend,
            max_concurrent=args.max_concurrent,
        )
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        raise

    # Run evaluation
    try:
        results = _run_evaluation(evaluator, args)

        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation completed. Results saved to: {output_path}")

        # Print summary
        _print_summary(results)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
