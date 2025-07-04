<p align="center">
  <img src="resources/images/logo.png"/>
</p>

<!-- icon -->

[![stars](https://img.shields.io/github/stars/open-sciencelab/GraphGen.svg)](https://github.com/open-sciencelab/GraphGen)
[![forks](https://img.shields.io/github/forks/open-sciencelab/GraphGen.svg)](https://github.com/open-sciencelab/GraphGen)
[![open issues](https://img.shields.io/github/issues-raw/open-sciencelab/GraphGen)](https://github.com/open-sciencelab/GraphGen/issues)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/open-sciencelab/GraphGen)](https://github.com/open-sciencelab/GraphGen/issues)
[![documentation](https://img.shields.io/badge/docs-latest-blue)](https://graphgen-cookbook.readthedocs.io/en/latest/)
[![wechat](https://img.shields.io/badge/wechat-brightgreen?logo=wechat&logoColor=white)](https://cdn.vansin.top/internlm/dou.jpg)
[![arXiv](https://img.shields.io/badge/Paper-arXiv-white)](https://arxiv.org/abs/2505.20416)
[![Hugging Face](https://img.shields.io/badge/Paper-on%20HF-white?logo=huggingface&logoColor=yellow)](https://huggingface.co/papers/2505.20416)

[![Hugging Face](https://img.shields.io/badge/Demo-on%20HF-blue?logo=huggingface&logoColor=yellow)](https://huggingface.co/spaces/chenzihong/GraphGen)
[![OpenXLab](https://img.shields.io/badge/Demo-on%20OpenXLab-blue?logo=openxlab&logoColor=yellow)](https://g-app-center-000704-6802-aerppvq.openxlab.space)


GraphGen: Enhancing Supervised Fine-Tuning for LLMs with Knowledge-Driven Synthetic Data Generation

<details close>
<summary><b>📚 Table of Contents</b></summary>

- 📝 [What is GraphGen?](#-what-is-graphgen)
- 🚀 [Quick Start](#-quick-start)
- 📌 [Latest Updates](#-latest-updates)
- 🏗️ [System Architecture](#-system-architecture)
- 🍀 [Acknowledgements](#-acknowledgements)
- 📚 [Citation](#-citation)
- 📜 [License](#-license)

[//]: # (- 🌟 [Key Features]&#40;#-key-features&#41;)
[//]: # (- 📅 [Roadmap]&#40;#-roadmap&#41;)
[//]: # (- 💰 [Cost Analysis]&#40;#-cost-analysis&#41;)
[//]: # (- ⚙️ [Configurations]&#40;#-configurations&#41;)

</details>

## 📝 What is GraphGen?

GraphGen is a framework for synthetic data generation guided by knowledge graphs. Please check the [**paper**](https://arxiv.org/abs/2505.20416) and [best practice](https://github.com/open-sciencelab/GraphGen/issues/17).

Here is post-training result which **over 50% SFT data** comes from GraphGen and our data clean pipeline.

| Domain | Dataset | Ours | Qwen2.5-7B-Instruct (baseline)	|
| :-: | :-: | :-: | :-: |
| Plant| [SeedBench](https://github.com/open-sciencelab/SeedBench) | **65.9** | 51.5 |
| Common | CMMLU | 73.6 | **75.8** |
| Logic | GPQA-Diamond | **40.0** | 33.3 |
| Math | AIME24 | **20.6** | 16.7 |
| | AIME25 | **22.7** | 7.2 |

It begins by constructing a fine-grained knowledge graph from the source text，then identifies knowledge gaps in LLMs using the expected calibration error metric, prioritizing the generation of QA pairs that target high-value, long-tail knowledge.
Furthermore, GraphGen incorporates multi-hop neighborhood sampling to capture complex relational information and employs style-controlled generation to diversify the resulting QA data.

## 🚀 Quick Start

Experience GraphGen through [Web](https://g-app-center-000704-6802-aerppvq.openxlab.space) or [Backup Web Entrance](https://openxlab.org.cn/apps/detail/tpoisonooo/GraphGen)

For any questions, please check [FAQ](https://github.com/open-sciencelab/GraphGen/issues/10), open new [issue](https://github.com/open-sciencelab/GraphGen/issues) or join our [wechat group](https://cdn.vansin.top/internlm/dou.jpg) and ask.

### Preparation

1. Install [uv](https://docs.astral.sh/uv/reference/installer/)

    ```bash
    # You could try pipx or pip to install uv when meet network issues, refer the uv doc for more details
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2. Clone the repository

    ```bash
    git clone https://github.com/open-sciencelab/GraphGen
    cd GraphGen
    ```

3. Create a new uv environment

    ```bash
     uv venv --python 3.10
    ```
   
4. Configure the dependencies

    ```bash
    uv pip install -r requirements.txt
    ```

### Run Gradio Demo

   ```bash
   uv run webui/app.py
   ```

![ui](https://github.com/user-attachments/assets/3024e9bc-5d45-45f8-a4e6-b57bd2350d84)

### Run from PyPI

1. Install GraphGen
   ```bash
   uv pip install graphg
   ```

2. Run in CLI
   ```bash
   SYNTHESIZER_MODEL=your_synthesizer_model_name \
   SYNTHESIZER_BASE_URL=your_base_url_for_synthesizer_model \
   SYNTHESIZER_API_KEY=your_api_key_for_synthesizer_model \
   TRAINEE_MODEL=your_trainee_model_name \
   TRAINEE_BASE_URL=your_base_url_for_trainee_model \
   TRAINEE_API_KEY=your_api_key_for_trainee_model \
   graphg --output_dir cache
   ```

### Run from Source

1. Configure the environment
   - Create an `.env` file in the root directory
     ```bash
     cp .env.example .env
     ```
   - Set the following environment variables:
     ```bash
     # Synthesizer is the model used to construct KG and generate data
     SYNTHESIZER_MODEL=your_synthesizer_model_name
     SYNTHESIZER_BASE_URL=your_base_url_for_synthesizer_model
     SYNTHESIZER_API_KEY=your_api_key_for_synthesizer_model
     # Trainee is the model used to train with the generated data
     TRAINEE_MODEL=your_trainee_model_name
     TRAINEE_BASE_URL=your_base_url_for_trainee_model
     TRAINEE_API_KEY=your_api_key_for_trainee_model
     ```
2. (Optional) If you want to modify the default generated configuration, you can edit the content of the configs/graphgen_config.yaml file.
    ```yaml
    # configs/graphgen_config.yaml
    # Example configuration
    data_type: "raw"
    input_file: "resources/examples/raw_demo.jsonl"
    # more configurations...
    ```
3. Run the generation script
   ```bash
   bash scripts/generate.sh
   ```
4. Get the generated data
   ```bash
   ls cache/data/graphgen
   ```

### Run with Docker
1. Build the Docker image
   ```bash
   docker build -t graphgen .
   ```
2. Run the Docker container
   ```bash
    docker run -p 7860:7860 graphgen
    ```


## 📌 Latest Updates

- **2025.04.21**: We have released the initial version of GraphGen.

## 🏗️ System Architecture

See [analysis](https://deepwiki.com/open-sciencelab/GraphGen) by deepwiki for a technical overview of the GraphGen system, its architecture, and core functionalities. 


### Workflow
![workflow](resources/images/flow.png)


## 🍀 Acknowledgements
- [SiliconFlow](https://siliconflow.cn) Abundant LLM API, some models are free
- [LightRAG](https://github.com/HKUDS/LightRAG) Simple and efficient graph retrieval solution
- [ROGRAG](https://github.com/tpoisonooo/ROGRAG) ROGRAG: A Robustly Optimized GraphRAG Framework


## 📚 Citation
If you find this repository useful, please consider citing our work:
```bibtex
@misc{chen2025graphgenenhancingsupervisedfinetuning,
      title={GraphGen: Enhancing Supervised Fine-Tuning for LLMs with Knowledge-Driven Synthetic Data Generation}, 
      author={Zihong Chen and Wanli Jiang and Jinzhe Li and Zhonghang Yuan and Huanjun Kong and Wanli Ouyang and Nanqing Dong},
      year={2025},
      eprint={2505.20416},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.20416}, 
}
```

## 📜 License
This project is licensed under the [Apache License 2.0](LICENSE).
