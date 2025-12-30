<p align="center">
  <img src="resources/images/logo.png"/>
</p>


GraphGen: Enhancing Supervised Fine-Tuning for LLMs with Knowledge-Driven Synthetic Data Generation

<details close>
<summary><b>📚 Table of Contents</b></summary>

- 📝 [What is GraphGen?](#-what-is-graphgen)
- ⚙️ [Support List](#-support-list)
- 🚀 [Quick Start](#-quick-start)

</details>

## 📝 What is GraphGen?

GraphGen is a framework for synthetic data generation guided by knowledge graphs.

It begins by constructing a fine-grained knowledge graph from the source text，then identifies knowledge gaps in LLMs using the expected calibration error metric, prioritizing the generation of QA pairs that target high-value, long-tail knowledge.
Furthermore, GraphGen incorporates multi-hop neighborhood sampling to capture complex relational information and employs style-controlled generation to diversify the resulting QA data.

After data generation, you can use the generated data to finetune your LLMs.


## ⚙️ Support List

We support various LLM inference servers, API servers, inference clients, input file formats, data modalities, output data formats, and output data types.
Users can flexibly configure according to the needs of synthetic data.


| Inference Server                                                         | Api Server                                                                     | Inference Client                                           | Data Source                                                                                                                                                                                                                                                                           | Data Modal    | Data Type                                       |
|--------------------------------------------------------------------------|--------------------------------------------------------------------------------|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|-------------------------------------------------|
| [![hf-icon]HF][hf]<br>[![sg-icon]SGLang][sg]<br>[![vllm-icon]vllm][vllm] | [![sif-icon]Silicon][sif]<br>[![oai-icon]OpenAI][oai]<br>[![az-icon]Azure][az] | HTTP<br>[![ol-icon]Ollama][ol]<br>[![oai-icon]OpenAI][oai] | Files(CSV, JSON, PDF, TXT, etc.)<br>Databases([![uniprot-icon]UniProt][uniprot], [![ncbi-icon]NCBI][ncbi], [![rnacentral-icon]RNAcentral][rnacentral])<br>Search Engines([![bing-icon]Bing][bing], [![google-icon]Google][google])<br>Knowledge Graphs([![wiki-icon]Wikipedia][wiki]) | TEXT<br>IMAGE | Aggregated<br>Atomic<br>CoT<br>Multi-hop<br>VQA |

<!-- links -->
[hf]: https://huggingface.co/docs/transformers/index
[sg]: https://docs.sglang.ai
[vllm]: https://github.com/vllm-project/vllm
[sif]: https://siliconflow.cn
[oai]: https://openai.com
[az]: https://azure.microsoft.com/en-us/services/cognitive-services/openai-service/
[ol]: https://ollama.com
[uniprot]: https://www.uniprot.org/
[ncbi]: https://www.ncbi.nlm.nih.gov/
[rnacentral]: https://rnacentral.org/
[wiki]: https://www.wikipedia.org/
[bing]: https://www.bing.com/
[google]: https://www.google.com


<!-- icons -->
[hf-icon]: https://www.google.com/s2/favicons?domain=https://huggingface.co
[sg-icon]: https://www.google.com/s2/favicons?domain=https://docs.sglang.ai
[vllm-icon]: https://www.google.com/s2/favicons?domain=https://docs.vllm.ai
[sif-icon]: https://www.google.com/s2/favicons?domain=siliconflow.com
[oai-icon]: https://www.google.com/s2/favicons?domain=https://openai.com
[az-icon]: https://www.google.com/s2/favicons?domain=https://azure.microsoft.com
[ol-icon]: https://www.google.com/s2/favicons?domain=https://ollama.com

[uniprot-icon]: https://www.google.com/s2/favicons?domain=https://www.uniprot.org
[ncbi-icon]: https://www.google.com/s2/favicons?domain=https://www.ncbi.nlm.nih.gov/
[rnacentral-icon]: https://www.google.com/s2/favicons?domain=https://rnacentral.org/
[wiki-icon]: https://www.google.com/s2/favicons?domain=https://www.wikipedia.org/
[bing-icon]: https://www.google.com/s2/favicons?domain=https://www.bing.com/
[google-icon]: https://www.google.com/s2/favicons?domain=https://www.google.com


## 🚀 Quick Start

### Preparation

1. Install [uv](https://docs.astral.sh/uv/reference/installer/)

    ```bash
    # You could try pipx or pip to install uv when meet network issues, refer the uv doc for more details
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2. Clone the repository

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
   python -m webui.app
   ```
   
   For hot-reload during development, run
   ```bash
   PYTHONPATH=. gradio webui/app.py
   ```

### Run from Source

1. Configure the environment
   - Create an `.env` file in the root directory
     ```bash
     cp .env.example .env
     ```
   - Set the following environment variables:
     ```bash
      # Tokenizer
      TOKENIZER_MODEL=
      
      # LLM
      # Support different backends: http_api, openai_api, ollama_api, ollama, huggingface, tgi, sglang, tensorrt
      # Synthesizer is the model used to construct KG and generate data
      # Trainee is the model used to train with the generated data

      # http_api / openai_api
      SYNTHESIZER_BACKEND=openai_api
      SYNTHESIZER_MODEL=gpt-4o-mini
      SYNTHESIZER_BASE_URL=
      SYNTHESIZER_API_KEY=
      TRAINEE_BACKEND=openai_api
      TRAINEE_MODEL=gpt-4o-mini
      TRAINEE_BASE_URL=
      TRAINEE_API_KEY=
      
      # azure_openai_api
      # SYNTHESIZER_BACKEND=azure_openai_api
      # The following is the same as your "Deployment name" in Azure
      # SYNTHESIZER_MODEL=<your-deployment-name>
      # SYNTHESIZER_BASE_URL=https://<your-resource-name>.openai.azure.com/openai/deployments/<your-deployment-name>/chat/completions
      # SYNTHESIZER_API_KEY=
      # SYNTHESIZER_API_VERSION=<api-version>
      
      # # ollama_api
      # SYNTHESIZER_BACKEND=ollama_api
      # SYNTHESIZER_MODEL=gemma3
      # SYNTHESIZER_BASE_URL=http://localhost:11434
      #
      # Note: TRAINEE with ollama_api backend is not supported yet as ollama_api does not support logprobs.
      
      # # huggingface
      # SYNTHESIZER_BACKEND=huggingface
      # SYNTHESIZER_MODEL=Qwen/Qwen2.5-0.5B-Instruct
      #
      # TRAINEE_BACKEND=huggingface
      # TRAINEE_MODEL=Qwen/Qwen2.5-0.5B-Instruct
      
      # # sglang
      # SYNTHESIZER_BACKEND=sglang
      # SYNTHESIZER_MODEL=Qwen/Qwen2.5-0.5B-Instruct
      # SYNTHESIZER_TP_SIZE=1
      # SYNTHESIZER_NUM_GPUS=1
      
      # TRAINEE_BACKEND=sglang
      # TRAINEE_MODEL=Qwen/Qwen2.5-0.5B-Instruct
      # SYNTHESIZER_TP_SIZE=1
      # SYNTHESIZER_NUM_GPUS=1
      
      # # vllm
      # SYNTHESIZER_BACKEND=vllm
      # SYNTHESIZER_MODEL=Qwen/Qwen2.5-0.5B-Instruct
      # SYNTHESIZER_NUM_GPUS=1
      
      # TRAINEE_BACKEND=vllm
      # TRAINEE_MODEL=Qwen/Qwen2.5-0.5B-Instruct
      # TRAINEE_NUM_GPUS=1
     ```
2. (Optional) Customize generation parameters in `config.yaml` .

   Edit the corresponding YAML file, e.g.:

    ```yaml
      # examples/generate/generate_aggregated_qa/aggregated_config.yaml
      global_params:
        working_dir: cache
        graph_backend: kuzu # graph database backend, support: kuzu, networkx
        kv_backend: rocksdb # key-value store backend, support: rocksdb, json_kv
   
      nodes:
        - id: read_files # id is unique in the pipeline, and can be referenced by other steps
          op_name: read
          type: source
          dependencies: []
          params:
            input_path:
              - examples/input_examples/jsonl_demo.jsonl # input file path, support json, jsonl, txt, pdf. See examples/input_examples for examples

      # additional settings...
    ```

3. Generate data

   Pick the desired format and run the matching script:
      
   | Format       | Script to run                                                          | Notes                                                                      |
   | ------------ | ---------------------------------------------------------------------- | -------------------------------------------------------------------------- |
   | `atomic`     | `bash examples/generate/generate_atomic_qa/generate_atomic.sh`         | Atomic Q\&A pairs covering basic knowledge                                 |
   | `aggregated` | `bash examples/generate/generate_aggregated_qa/generate_aggregated.sh` | Aggregated Q\&A pairs incorporating complex, integrated knowledge          |
   | `multi-hop`  | `examples/generate/generate_multi_hop_qa/generate_multi_hop.sh`        | Multi-hop reasoning Q\&A pairs                                             |


4. Get the generated data
   ```bash
   ls cache/output
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


### Workflow
![workflow](resources/images/flow.png)
