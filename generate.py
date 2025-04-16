import os
import json
import time
import argparse
import yaml
from dotenv import load_dotenv

from graphgen.graphgen import GraphGen
from graphgen.models import OpenAIModel, Tokenizer, TraverseStrategy
from graphgen.utils import set_logger

sys_path = os.path.abspath(os.path.dirname(__file__))
unique_id = int(time.time())
set_logger(os.path.join(sys_path, "cache", "logs", f"graphgen_{unique_id}.log"), if_stream=False)
config_path = os.path.join(sys_path, "cache", "data", "graphgen", str(unique_id), f"config-{unique_id}.yaml")

load_dotenv()

def save_config(global_config):
    if not os.path.exists(os.path.dirname(config_path)):
        os.makedirs(os.path.dirname(config_path))
    with open(config_path, "w", encoding='utf-8') as config_file:
        yaml.dump(global_config, config_file, default_flow_style=False, allow_unicode=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',
                        help='Config parameters for GraphGen.',
                        default='graphgen_config.yaml',
                        type=str)
    args = parser.parse_args()
    with open(args.config_file, "r", encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    input_file = config['input_file']

    if config['data_type'] == 'raw':
        with open(input_file, "r", encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    elif config['data_type'] == 'chunked':
        with open(input_file, "r", encoding='utf-8') as f:
            data = json.load(f)
    else:
        raise ValueError(f"Invalid data type: {config['data_type']}")

    synthesizer_llm_client = OpenAIModel(
        model_name=os.getenv("SYNTHESIZER_MODEL"),
        api_key=os.getenv("SYNTHESIZER_API_KEY"),
        base_url=os.getenv("SYNTHESIZER_BASE_URL")
    )
    trainee_llm_client = OpenAIModel(
        model_name=os.getenv("TRAINEE_MODEL"),
        api_key=os.getenv("TRAINEE_API_KEY"),
        base_url=os.getenv("TRAINEE_BASE_URL")
    )

    traverse_strategy = TraverseStrategy(
        **config['traverse_strategy']
    )

    graph_gen = GraphGen(
        unique_id=unique_id,
        synthesizer_llm_client=synthesizer_llm_client,
        trainee_llm_client=trainee_llm_client,
        if_web_search=config['web_search'],
        tokenizer_instance=Tokenizer(
            model_name=config['tokenizer']
        ),
        traverse_strategy=traverse_strategy
    )

    graph_gen.insert(data, config['data_type'])

    graph_gen.quiz(max_samples=config['quiz_samples'])

    graph_gen.judge(re_judge=False)

    graph_gen.traverse()

    save_config(config)
