import argparse
import os
import time
from importlib.resources import files

import yaml
from dotenv import load_dotenv

from .graphgen import GraphGen
from .models import OpenAIModel, Tokenizer, TraverseStrategy
from .utils import read_file, set_logger

sys_path = os.path.abspath(os.path.dirname(__file__))

load_dotenv()


def set_working_dir(folder):
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, "data", "graphgen"), exist_ok=True)
    os.makedirs(os.path.join(folder, "logs"), exist_ok=True)


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
        default=files("graphgen").joinpath("configs", "graphgen_config.yaml"),
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory for GraphGen.",
        default=sys_path,
        required=True,
        type=str,
    )

    args = parser.parse_args()

    working_dir = args.output_dir
    set_working_dir(working_dir)
    unique_id = int(time.time())
    set_logger(
        os.path.join(working_dir, "logs", f"graphgen_{unique_id}.log"), if_stream=False
    )
    print(
        "GraphGen with unique ID",
        unique_id,
        "logging to",
        os.path.join(working_dir, "logs", f"graphgen_{unique_id}.log"),
    )

    with open(args.config_file, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    input_file = config["input_file"]
    data = read_file(input_file)

    synthesizer_llm_client = OpenAIModel(
        model_name=os.getenv("SYNTHESIZER_MODEL"),
        api_key=os.getenv("SYNTHESIZER_API_KEY"),
        base_url=os.getenv("SYNTHESIZER_BASE_URL"),
    )
    trainee_llm_client = OpenAIModel(
        model_name=os.getenv("TRAINEE_MODEL"),
        api_key=os.getenv("TRAINEE_API_KEY"),
        base_url=os.getenv("TRAINEE_BASE_URL"),
    )

    traverse_strategy = TraverseStrategy(**config["traverse_strategy"])

    graph_gen = GraphGen(
        working_dir=working_dir,
        unique_id=unique_id,
        synthesizer_llm_client=synthesizer_llm_client,
        trainee_llm_client=trainee_llm_client,
        search_config=config["search"],
        tokenizer_instance=Tokenizer(model_name=config["tokenizer"]),
        traverse_strategy=traverse_strategy,
    )

    graph_gen.insert(data, config["data_type"])

    if config["search"]["enabled"]:
        graph_gen.search()

    graph_gen.quiz(max_samples=config["quiz_samples"])

    graph_gen.judge(re_judge=config["re_judge"])

    graph_gen.traverse()

    path = os.path.join(
        working_dir, "data", "graphgen", str(unique_id), f"config-{unique_id}.yaml"
    )
    save_config(path, config)


if __name__ == "__main__":
    main()
