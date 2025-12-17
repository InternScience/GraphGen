import argparse
import logging
import os
import sys
import time
from importlib import resources
from typing import Any, Dict

import ray
import yaml
from dotenv import load_dotenv
from ray.data.block import Block
from ray.data.datasource.filename_provider import FilenameProvider

from graphgen.engine import Engine
from graphgen.operators import operators
from graphgen.utils import CURRENT_LOGGER_VAR, logger, set_logger

sys_path = os.path.abspath(os.path.dirname(__file__))

load_dotenv()

# Suppress non-error output temporarily
# Save original streams for restoration
_original_stdout = sys.stdout
_original_stderr = sys.stderr
_devnull = None


def set_working_dir(folder):
    os.makedirs(folder, exist_ok=True)


def save_config(config_path, global_config):
    if not os.path.exists(os.path.dirname(config_path)):
        os.makedirs(os.path.dirname(config_path))
    with open(config_path, "w", encoding="utf-8") as config_file:
        yaml.dump(
            global_config, config_file, default_flow_style=False, allow_unicode=True
        )


class NodeFilenameProvider(FilenameProvider):
    def __init__(self, node_id: str):
        self.node_id = node_id

    def get_filename_for_block(
        self, block: Block, write_uuid: str, task_index: int, block_index: int
    ) -> str:
        # format: {node_id}_{write_uuid}_{task_index:06}_{block_index:06}.json
        return f"{self.node_id}_{write_uuid}_{task_index:06d}_{block_index:06d}.jsonl"

    def get_filename_for_row(
        self,
        row: Dict[str, Any],
        write_uuid: str,
        task_index: int,
        block_index: int,
        row_index: int,
    ) -> str:
        raise NotImplementedError(
            f"Row-based filenames are not supported by write_json. "
            f"Node: {self.node_id}, write_uuid: {write_uuid}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        help="Config parameters for GraphGen.",
        default=resources.files("graphgen")
        .joinpath("configs")
        .joinpath("aggregated_config.yaml"),
        type=str,
    )

    args = parser.parse_args()

    with open(args.config_file, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    working_dir = config.get("global_params", {}).get("working_dir", "cache")
    unique_id = int(time.time())
    output_path = os.path.join(working_dir, "output", f"{unique_id}")
    set_working_dir(output_path)
    log_path = os.path.join(working_dir, "logs", "Driver.log")
    driver_logger = set_logger(
        log_path,
        name="GraphGen",
        console_level=logging.ERROR,
        if_stream=True,
    )
    CURRENT_LOGGER_VAR.set(driver_logger)
    logger.info(
        "GraphGen with unique ID %s logging to %s",
        unique_id,
        log_path,
    )

    # Temporarily suppress non-error output (print statements, third-party libraries, Ray Data progress)
    # Only redirect stdout to preserve stderr for logger error output
    global _devnull
    _devnull = open(os.devnull, 'w')
    sys.stdout = _devnull

    try:
        engine = Engine(config, operators)
        ds = ray.data.from_items([])
        results = engine.execute(ds)

        for node_id, dataset in results.items():
            node_output_path = os.path.join(output_path, f"{node_id}")
            os.makedirs(node_output_path, exist_ok=True)
            dataset.write_json(
                node_output_path,
                filename_provider=NodeFilenameProvider(node_id),
                pandas_json_args_fn=lambda: {
                    "force_ascii": False,
                    "orient": "records",
                    "lines": True,
                },
            )
            logger.info("Node %s results saved to %s", node_id, node_output_path)

        save_config(os.path.join(output_path, "config.yaml"), config)
        logger.info("GraphGen completed successfully. Data saved to %s", output_path)
    finally:
        # Restore original stdout before printing results
        sys.stdout = _original_stdout
        if _devnull:
            _devnull.close()
            _devnull = None
        
        # Print save information to console
        if 'results' in locals() and results:
            print("\n" + "="*60)
            print("GraphGen execution completed successfully!")
            print("="*60)
            for node_id, dataset in results.items():
                node_output_path = os.path.join(output_path, f"{node_id}")
                print(f"✓ Node '{node_id}' results saved to: {node_output_path}")
            print(f"✓ Config saved to: {os.path.join(output_path, 'config.yaml')}")
            print(f"✓ Logs saved to: {log_path}")
            print("="*60 + "\n")
        else:
            print("\n⚠️  Warning: No results were generated.\n")


if __name__ == "__main__":
    main()
