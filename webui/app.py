import json
import os
import gradio as gr
import yaml

import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from gradio_i18n import Translate, gettext as _

from graphgen.graphgen import GraphGen
from models import OpenAIModel, Tokenizer, TraverseStrategy
from test_api import test_api_connection

def load_config() -> dict:
    with open("config.yaml", "r", encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_config(config: dict):
    with open("config.yaml", "w", encoding='utf-8') as f:
        yaml.dump(config, f)

def load_env() -> dict:
    env = {}
    if os.path.exists(".env"):
        with open(".env", "r", encoding='utf-8') as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    env[key] = value
    return env

def save_env(env: dict):
    with open(".env", "w", encoding='utf-8') as f:
        for key, value in env.items():
            f.write(f"{key}={value}\n")

def init_graph_gen(config: dict, env: dict) -> GraphGen:
    graph_gen = GraphGen()

    # Set up LLM clients
    graph_gen.synthesizer_llm_client = OpenAIModel(
        model_name=env.get("SYNTHESIZER_MODEL", ""),
        base_url=env.get("SYNTHESIZER_BASE_URL", ""),
        api_key=env.get("SYNTHESIZER_API_KEY", "")
    )

    graph_gen.trainee_llm_client = OpenAIModel(
        model_name=env.get("TRAINEE_MODEL", ""),
        base_url=env.get("TRAINEE_BASE_URL", ""),
        api_key=env.get("TRAINEE_API_KEY", "")
    )

    # Set up tokenizer
    graph_gen.tokenizer_instance = Tokenizer(config.get("tokenizer", "cl100k_base"))

    # Set up traverse strategy
    strategy_config = config.get("traverse_strategy", {})
    graph_gen.traverse_strategy = TraverseStrategy(
        qa_form=config.get("qa_form", "multi_hop"),
        expand_method=strategy_config.get("expand_method", "max_tokens"),
        bidirectional=strategy_config.get("bidirectional", True),
        max_extra_edges=strategy_config.get("max_extra_edges", 5),
        max_tokens=strategy_config.get("max_tokens", 256),
        max_depth=strategy_config.get("max_depth", 2),
        edge_sampling=strategy_config.get("edge_sampling", "max_loss"),
        isolated_node_strategy=strategy_config.get("isolated_node_strategy", "add"),
        difficulty_order=strategy_config.get("difficulty_order", ["medium", "medium", "medium"])
    )

    graph_gen.if_web_search = config.get("web_search", False)

    return graph_gen

# pylint: disable=redefined-outer-name
def run_graphgen(
        input_file: str,
        data_type: str,
        qa_form: str,
        tokenizer: str,
        web_search: bool,
        expand_method: str,
        bidirectional: bool,
        max_extra_edges: int,
        max_tokens: int,
        max_depth: int,
        edge_sampling: str,
        isolated_node_strategy: str,
        difficulty_level: str,
        synthesizer_model: str,
        synthesizer_base_url: str,
        synthesizer_api_key: str,
        trainee_model: str,
        trainee_base_url: str,
        trainee_api_key: str,
        quiz_samples: int,
        progress=gr.Progress()
) -> str:
    # Save configurations
    config = {
        "data_type": data_type,
        "input_file": input_file,
        "qa_form": qa_form,
        "tokenizer": tokenizer,
        "web_search": web_search,
        "quiz_samples": quiz_samples,
        "traverse_strategy": {
            "expand_method": expand_method,
            "bidirectional": bidirectional,
            "max_extra_edges": max_extra_edges,
            "max_tokens": max_tokens,
            "max_depth": max_depth,
            "edge_sampling": edge_sampling,
            "isolated_node_strategy": isolated_node_strategy,
            "difficulty_order": [difficulty_level] * 3
        }
    }
    save_config(config)

    env = {
        "SYNTHESIZER_MODEL": synthesizer_model,
        "SYNTHESIZER_BASE_URL": synthesizer_base_url,
        "SYNTHESIZER_API_KEY": synthesizer_api_key,
        "TRAINEE_MODEL": trainee_model,
        "TRAINEE_BASE_URL": trainee_base_url,
        "TRAINEE_API_KEY": trainee_api_key
    }
    save_env(env)

    # Initialize GraphGen
    graph_gen = init_graph_gen(config, env)

    try:
        # Load input data
        if data_type == "raw":
            with open(input_file, "r", encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
        else:  # chunked
            with open(input_file, "r", encoding='utf-8') as f:
                data = json.load(f)

        progress(0.2, "Data loaded")

        # Process the data
        graph_gen.insert(data, data_type)
        progress(0.4, "Data inserted")

        # Generate quiz
        graph_gen.quiz(max_samples=quiz_samples)
        progress(0.6, "Quiz generated")

        # Judge statements
        graph_gen.judge(re_judge=False)
        progress(0.8, "Statements judged")

        # Traverse graph
        graph_gen.traverse()
        progress(1.0, "Graph traversed")

        return "Task completed successfully!"

    except Exception as e: # pylint: disable=broad-except
        return f"Error occurred: {str(e)}"

config = load_env()

# Create Gradio interface
with gr.Blocks(title="GraphGen Demo") as demo:
    lang = gr.Radio(
        choices=[
            ("English", "en"),
            ("简体中文", "zh"),
        ],
        value="en",
        label=_("Language"),
        render=False,
    )
    with Translate(
        "translation.yaml",
        lang,
        placeholder_langs=["en", "zh"],
        persistant=False,  # True to save the language setting in the browser. Requires gradio >= 5.6.0
    ):
        lang.render()
        # Header
        gr.Image(
            value=f"{root_dir}/resources/images/logo.png",
            label="GraphGen Banner",
            elem_id="banner",
            show_label=False,
            interactive=False,
        )
        gr.Markdown(
            """
            This is a demo for the [GraphGen](https://github.com/open-sciencelab/GraphGen) project. 
            GraphGen is a framework for synthetic data generation guided by knowledge graphs. 
            """
        )
        with gr.Row():
            # Model Configuration Column
            with gr.Column(scale=1):
                gr.Markdown("### Model Configuration")
                synthesizer_model = gr.Textbox(label="Synthesizer Model", value=config.get("SYNTHESIZER_MODEL", ""))
                synthesizer_base_url = gr.Textbox(label="Synthesizer Base URL", value=config.get("SYNTHESIZER_BASE_URL", ""))
                synthesizer_api_key = gr.Textbox(label="Synthesizer API Key", type="password", value=config.get("SYNTHESIZER_API_KEY", ""))
                trainee_model = gr.Textbox(label="Trainee Model", value=config.get("TRAINEE_MODEL", ""))
                trainee_base_url = gr.Textbox(label="Trainee Base URL", value=config.get("TRAINEE_BASE_URL", ""))
                trainee_api_key = gr.Textbox(label="Trainee API Key", type="password", value=config.get("TRAINEE_API_KEY", ""))
                test_connection_btn = gr.Button("Test Connection", variant="primary")
                gr.Button("Save Config", variant="primary")

            # Input Configuration Column
            with gr.Column(scale=1):
                gr.Markdown("### Input Configuration")
                input_file = gr.Textbox(label="Input File Path", value="resources/examples/raw_demo.jsonl")
                data_type = gr.Radio(choices=["raw", "chunked"], label="Data Type", value="raw")
                qa_form = gr.Radio(choices=["atomic", "multi_hop", "open"], label="QA Form", value="multi_hop")
                tokenizer = gr.Textbox(label="Tokenizer", value="cl100k_base")
                web_search = gr.Checkbox(label="Enable Web Search", value=False)
                quiz_samples = gr.Number(label="Quiz Samples", value=2, minimum=1)

            # Traverse Strategy Column
            with gr.Column(scale=1):
                gr.Markdown("### Traverse Strategy")
                expand_method = gr.Radio(choices=["max_width", "max_tokens"], label="Expand Method", value="max_tokens")
                bidirectional = gr.Checkbox(label="Bidirectional", value=True)
                max_extra_edges = gr.Slider(minimum=1, maximum=10, value=5, label="Max Extra Edges", step=1)
                max_tokens = gr.Slider(minimum=64, maximum=1024, value=256, label="Max Tokens", step=64)
                max_depth = gr.Slider(minimum=1, maximum=5, value=2, label="Max Depth", step=1)
                edge_sampling = gr.Radio(choices=["max_loss", "min_loss", "random"], label="Edge Sampling",
                                         value="max_loss")
                isolated_node_strategy = gr.Radio(choices=["add", "ignore"], label="Isolated Node Strategy", value="ignore")
                difficulty_level = gr.Radio(choices=["easy", "medium", "hard"], label="Difficulty Level", value="medium")

        # Submission and Output Rows
        with gr.Row():
            submit_btn = gr.Button("Run GraphGen")
        with gr.Row():
            output = gr.Textbox(label="Output")

        # Test Connection
        test_connection_btn.click(
            test_api_connection,
            inputs=[synthesizer_base_url, synthesizer_api_key, synthesizer_model],
            outputs=output
        ).then(
            test_api_connection,
            inputs=[trainee_base_url, trainee_api_key, trainee_model],
            outputs=output
        )

        # Event Handling
        submit_btn.click(
            run_graphgen,
            inputs=[
                input_file, data_type, qa_form, tokenizer, web_search,
                expand_method, bidirectional, max_extra_edges, max_tokens,
                max_depth, edge_sampling, isolated_node_strategy, difficulty_level,
                synthesizer_model, synthesizer_base_url, synthesizer_api_key,
                trainee_model, trainee_base_url, trainee_api_key,
                quiz_samples
            ],
            outputs=output
        )

if __name__ == "__main__":
    demo.launch()
