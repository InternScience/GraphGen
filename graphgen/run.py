import argparse
import os
import time
from importlib import resources

import ray
import yaml
from dotenv import load_dotenv

from graphgen.engine import Engine
from graphgen.operators import operators
from graphgen.utils import CURRENT_LOGGER_VAR, logger, set_logger

load_dotenv()

sys_path = os.path.abspath(os.path.dirname(__file__))


def set_working_dir(folder):
    os.makedirs(folder, exist_ok=True)


def save_config(config_path, global_config):
    if not os.path.exists(os.path.dirname(config_path)):
        os.makedirs(os.path.dirname(config_path))
    with open(config_path, "w", encoding="utf-8") as config_file:
        yaml.dump(
            global_config, config_file, default_flow_style=False, allow_unicode=True
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
        if_stream=True,
    )
    CURRENT_LOGGER_VAR.set(driver_logger)
    logger.info(
        "GraphGen with unique ID %s logging to %s",
        unique_id,
        log_path,
    )

    engine = Engine(config, operators)
    ds = ray.data.from_items([])
    engine.execute(ds, output_dir=output_path)

    save_config(os.path.join(output_path, "config.yaml"), config)
    logger.info("GraphGen completed successfully. Data saved to %s", output_path)


if __name__ == "__main__":
    main()
