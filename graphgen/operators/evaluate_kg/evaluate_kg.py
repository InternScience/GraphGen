import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from graphgen.models import KGQualityEvaluator
from graphgen.utils import CURRENT_LOGGER_VAR, logger, set_logger

# Load environment variables
load_dotenv()


def main():
    """Main function to run KG quality evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate knowledge graph quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python -m graphgen.operators.evaluate_kg.evaluate_kg --working_dir cache

  # Custom sample size and output
  python -m graphgen.operators.evaluate_kg.evaluate_kg \\
    --working_dir cache \\
    --sample_size 200 \\
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
        "--sample_size",
        type=int,
        default=100,
        help="Sample size for accuracy evaluation (default: 100)",
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
    logger.info(f"Sample size: {args.sample_size}")

    # Initialize evaluator
    try:
        evaluator = KGQualityEvaluator(
            working_dir=args.working_dir,
            graph_backend=args.graph_backend,
            kv_backend=args.kv_backend,
            sample_size=args.sample_size,
            max_concurrent=args.max_concurrent,
        )
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        raise

    # Run evaluation
    try:
        if args.accuracy_only:
            logger.info("Running accuracy evaluation only...")
            results = {"accuracy": evaluator.evaluate_accuracy()}
        elif args.consistency_only:
            logger.info("Running consistency evaluation only...")
            results = {"consistency": evaluator.evaluate_consistency()}
        elif args.structure_only:
            logger.info("Running structural robustness evaluation only...")
            results = {"structure": evaluator.evaluate_structure()}
        else:
            logger.info("Running all evaluations...")
            results = evaluator.evaluate_all()

        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation completed. Results saved to: {output_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("KG Quality Evaluation Summary")
        print("=" * 60)

        if "accuracy" in results:
            acc = results["accuracy"]
            if "error" not in acc:
                print("\n[Accuracy]")
                if "entity_accuracy" in acc:
                    e = acc["entity_accuracy"]
                    print(f"  Entity - Precision: {e.get('precision', 0):.3f}, "
                          f"Recall: {e.get('recall', 0):.3f}, F1: {e.get('f1', 0):.3f}")
                if "relation_accuracy" in acc:
                    r = acc["relation_accuracy"]
                    print(f"  Relation - Precision: {r.get('precision', 0):.3f}, "
                          f"Recall: {r.get('recall', 0):.3f}, F1: {r.get('f1', 0):.3f}")
                if "triple_accuracy" in acc:
                    t = acc["triple_accuracy"]
                    print(f"  Triple (RLC) - Precision: {t.get('precision', 0):.3f}, "
                          f"Recall: {t.get('recall', 0):.3f}, F1: {t.get('f1', 0):.3f}")
            else:
                print(f"\n[Accuracy] Error: {acc['error']}")

        if "consistency" in results:
            cons = results["consistency"]
            if "error" not in cons:
                print("\n[Consistency]")
                print(f"  Conflict Rate: {cons.get('conflict_rate', 0):.3f}")
                print(f"  Conflict Entities: {cons.get('conflict_entities_count', 0)} / "
                      f"{cons.get('total_entities', 0)}")
            else:
                print(f"\n[Consistency] Error: {cons['error']}")

        if "structure" in results:
            struct = results["structure"]
            if "error" not in struct:
                print("\n[Structural Robustness]")
                print(f"  Total Nodes: {struct.get('total_nodes', 0)}")
                print(f"  Total Edges: {struct.get('total_edges', 0)}")
                print(f"  Noise Ratio: {struct.get('noise_ratio', 0):.3f} "
                      f"({'✓' if struct.get('noise_ratio', 1) < 0.15 else '✗'} < 15%)")
                print(f"  Largest CC Ratio: {struct.get('largest_cc_ratio', 0):.3f} "
                      f"({'✓' if struct.get('largest_cc_ratio', 0) > 0.90 else '✗'} > 90%)")
                print(f"  Avg Degree: {struct.get('avg_degree', 0):.2f} "
                      f"({'✓' if 2 <= struct.get('avg_degree', 0) <= 5 else '✗'} 2-5)")
                if struct.get('powerlaw_r2') is not None:
                    print(f"  Power Law R²: {struct.get('powerlaw_r2', 0):.3f} "
                          f"({'✓' if struct.get('powerlaw_r2', 0) > 0.75 else '✗'} > 0.75)")
            else:
                print(f"\n[Structural Robustness] Error: {struct['error']}")

        print("\n" + "=" * 60)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
