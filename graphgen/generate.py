import argparse
import os
import time
from importlib.resources import files

import yaml
from dotenv import load_dotenv

from .graphgen import GraphGen
from .models import OpenAIModel, Tokenizer, TraverseStrategy
from .utils import logger, read_file, set_logger

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
        default=files("graphgen").joinpath("configs", "aggregated_config.yaml"),
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

    with open(args.config_file, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    input_file = config["input_file"]
    data = read_file(input_file)
    output_data_type = config["output_data_type"]

    unique_id = int(time.time())
    set_logger(
        os.path.join(
            working_dir, "logs", f"graphgen_{output_data_type}_{unique_id}.log"
        ),
        if_stream=True,
    )
    logger.info(
        "GraphGen with unique ID %s logging to %s",
        unique_id,
        os.path.join(
            working_dir, "logs", f"graphgen_{output_data_type}_{unique_id}.log"
        ),
    )

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

    graph_gen = GraphGen(
        working_dir=working_dir,
        unique_id=unique_id,
        synthesizer_llm_client=synthesizer_llm_client,
        trainee_llm_client=trainee_llm_client,
        search_config=config["search"],
        tokenizer_instance=Tokenizer(model_name=config["tokenizer"]),
    )

    graph_gen.insert(data, config["input_data_type"])

    if config["search"]["enabled"]:
        graph_gen.search()

    # Use pipeline according to the output data type
    if output_data_type in ["atomic", "aggregated", "multi_hop"]:
        graph_gen.quiz(max_samples=config["quiz_samples"])
        graph_gen.judge(re_judge=config["re_judge"])
        traverse_strategy = TraverseStrategy(**config["traverse_strategy"])
        traverse_strategy.qa_form = output_data_type
        graph_gen.traverse(traverse_strategy=traverse_strategy)
    elif output_data_type == "cot":
        graph_gen.generate_reasoning(method_params=config["method_params"])
    else:
        raise ValueError(f"Unsupported output data type: {output_data_type}")

    output_path = os.path.join(working_dir, "data", "graphgen", str(unique_id))
    save_config(os.path.join(output_path, f"config-{unique_id}.yaml"), config)
    logger.info("GraphGen completed successfully. Data saved to %s", output_path)


if __name__ == "__main__":
    main()
