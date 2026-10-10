<p align="center">
  <img src="assets/logo.png"/>
</p>

<!-- icon -->

[![stars](https://img.shields.io/github/stars/open-sciencelab/GraphGen.svg)](https://github.com/open-sciencelab/GraphGen)
[![forks](https://img.shields.io/github/forks/open-sciencelab/GraphGen.svg)](https://github.com/open-sciencelab/GraphGen)
[![open issues](https://img.shields.io/github/issues-raw/open-sciencelab/GraphGen)](https://github.com/open-sciencelab/GraphGen/issues)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/open-sciencelab/GraphGen)](https://github.com/open-sciencelab/GraphGen/issues)
[![documentation](https://img.shields.io/badge/docs-latest-blue)](https://graphgen-cookbook.readthedocs.io/en/latest/)
[![pypi](https://img.shields.io/pypi/v/graphg.svg?style=flat&logo=pypi&logoColor=white)](https://pypi.org/project/graphg/)
[![wechat](https://img.shields.io/badge/wechat-brightgreen?logo=wechat&logoColor=white)](https://cdn.vansin.top/internlm/dou.jpg)
[![arXiv](https://img.shields.io/badge/Paper-arXiv-white)](https://arxiv.org/abs/2505.20416)
[![Hugging Face](https://img.shields.io/badge/Paper-on%20HF-white?logo=huggingface&logoColor=yellow)](https://huggingface.co/papers/2505.20416)

[![Hugging Face](https://img.shields.io/badge/Demo-on%20HF-blue?logo=huggingface&logoColor=yellow)](https://huggingface.co/spaces/chenzihong/GraphGen)
[![Model Scope](https://img.shields.io/badge/%F0%9F%A4%96%20Demo-on%20MS-green)](https://modelscope.cn/studios/chenzihong/GraphGen)

GraphGen: Enhancing Supervised Fine-Tuning for LLMs with Knowledge-Driven Synthetic Data Generation

[English](README.md) | [中文](README_zh.md)

<details close>
<summary><b>📚 目录</b></summary>

- 📝 [什么是 GraphGen？](#-什么是-graphgen)
- 📌 [最新更新](#-最新更新)
- ⚙️ [支持列表](#-支持列表)
- 🚀 [快速开始](#-快速开始)
- 🏗️ [系统架构](#-系统架构)
- 🍀 [致谢](#-致谢)
- 📚 [引用](#-引用)
- 📜 [许可证](#-许可证)
- 📅 [星标历史](#-星标历史)


[//]: # (- 🌟 [主要特性](#主要特性))
[//]: # (- 💰 [成本分析](#成本分析))
[//]: # (- ⚙️ [配置说明](#配置说明))

</details>


## 📝 什么是 GraphGen？

GraphGen 是一个基于知识图谱的数据合成框架。请查看[**论文**](https://arxiv.org/abs/2505.20416)和[最佳实践](https://github.com/open-sciencelab/GraphGen/issues/17)。


GraphGen 首先根据源文本构建细粒度的知识图谱，然后利用期望校准误差指标识别大语言模型中的知识缺口，优先生成针对高价值长尾知识的问答对。  
此外，GraphGen 采用多跳邻域采样捕获复杂关系信息，并使用风格控制生成来丰富问答数据的多样性。

在数据生成后，您可以使用[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 和 [xtuner](https://github.com/InternLM/xtuner)对大语言模型进行微调。

## 📌 最新功能
- 🎉 **2026.04.13**：基于 GraphGen 的论文 *Knowledge-to-Verification: Exploring RLVR for LLMs in Knowledge-Intensive Domains* 已被 **ACL 2026** 主会录取！恭喜！[代码](https://github.com/superfarther/K2V)
- **2026.02.04**：支持使用直接读入 HuggingFace 数据集进行数据生成
- **2026.01.15**：合成垂域评测数据（单选题、多选题、填空题和判断题型）🌟🌟


<details>
<summary>历史更新记录</summary>

- **2025.12.26**：引入知识图谱评估指标，包括准确度评估（实体/关系抽取质量）、一致性评估（冲突检测）和结构鲁棒性评估（噪声比、连通性、度分布）
- **2025.12.16**：支持 [rocksdb](https://github.com/facebook/rocksdb) 作为键值存储后端, [kuzudb](https://github.com/kuzudb/kuzu) 作为图数据库后端。
- **2025.12.16**：支持 [vllm](https://github.com/vllm-project/vllm) 作为本地推理后端。
- **2025.12.16**：使用 [ray](https://github.com/ray-project/ray) 重构了数据生成 pipeline，提升了分布式执行和资源管理的效率。
- **2025.12.1**：新增对 [NCBI](https://www.ncbi.nlm.nih.gov/) 和 [RNAcentral](https://rnacentral.org/) 数据库的检索支持，现在可以从这些生物信息学数据库中提取DNA和RNA数据。
- **2025.10.30**：我们支持多种新的 LLM 客户端和推理后端，包括 [Ollama_client]([Ollama_client](https://github.com/open-sciencelab/GraphGen/blob/main/graphgen/models/llm/api/ollama_client.py), [http_client](https://github.com/open-sciencelab/GraphGen/blob/main/graphgen/models/llm/api/http_client.py), [HuggingFace Transformers](https://github.com/open-sciencelab/GraphGen/blob/main/graphgen/models/llm/local/hf_wrapper.py) 和 [SGLang](https://github.com/open-sciencelab/GraphGen/blob/main/graphgen/models/llm/local/sglang_wrapper.py)。
- **2025.10.23**：我们现在支持视觉问答（VQA）数据生成。运行脚本：`bash scripts/generate/generate_vqa.sh`。
- **2025.10.21**：我们现在通过 [MinerU](https://github.com/opendatalab/MinerU) 支持 PDF 作为数据生成的输入格式。
- **2025.09.29**：我们在 [Hugging Face](https://huggingface.co/spaces/chenzihong/GraphGen) 和 [ModelScope](https://modelscope.cn/studios/chenzihong/GraphGen) 上自动更新 Gradio 应用。
- **2025.08.14**：支持利用 Leiden 社区发现算法对知识图谱进行社区划分，合成 CoT 数据。
- **2025.07.31**：新增 Google、Bing、Wikipedia 和 UniProt 作为搜索后端，帮助填补数据缺口。  
- **2025.04.21**：发布 GraphGen 初始版本。

</details>

## GraphGen的效果
### Pretrain

受 Kimi-K2 的 技术报告 (https://arxiv.org/pdf/2507.20534) (Improving Token Utility with Rephrasing) 和 ByteDance Seed 的 [Reformulation for Pretraining Data Augmentation](https://arxiv.org/abs/2502.04235)（MGA 框架）启发，GraphGen 引入了一套**重述流水线（rephrase pipeline）**——利用大语言模型对语料进行改写，生成同一知识内容的多种表达变体，替代传统的简单重复训练。

**实验设置：** 使用 Qwen3-0.6B 模型，基于 [SlimPajama-6B](https://huggingface.co/datasets/DKYoon/SlimPajama-6B) 数据集从头训练。

| 方法 | [ARC-E](https://allenai.org/data/arc) | [ARC-C](https://allenai.org/data/arc) | [HellaSwag](https://rowanzellers.com/hellaswag/) | [GSM8K](https://github.com/openai/grade-school-math) | [TruthfulQA-MC1](https://github.com/sylinrl/TruthfulQA) | [TruthfulQA-MC2](https://github.com/sylinrl/TruthfulQA) | **平均值** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| SlimPajama-6B 训练 2 epoch | 25.55 | 21.08 | 24.48 | 0.08 | 24.36 | 49.90 | 24.24 |
| SlimPajama-6B + Executive-Summary Rephrase 训练 1 epoch | 26.43 | **22.70** | **24.75** | **1.36** | **26.19** | 51.90 | **25.56**(↑1.32) |
| SlimPajama-6B + Cross-Domain Rephrase 训练 1 epoch | **28.79** | 20.22 | 24.46 | 0.00 | 24.97 | **52.41** | 25.14(↑0.9) |

两种重述方法均在**零额外数据**的情况下，将平均性能较基线提升约 1 个百分点——所有增益均来自于对相同知识的不同表达方式。


### SFT
以下是在超过 50 % 的 SFT 数据来自 GraphGen 及我们的数据清洗流程时的训练后结果：

| 领域 |                            数据集                            |  我们的方案   | Qwen2.5-7B-Instruct（基线） |
|:--:|:---------------------------------------------------------:|:--------:|:-----------------------:|
| 植物 | [SeedBench](https://github.com/open-sciencelab/SeedBench) | **65.9** |          51.5           |
| 常识 |        [CMMLU](https://github.com/haonan-li/CMMLU)        |   73.6   |        **75.8**         |
| 知识 |      [GPQA-Diamond](https://github.com/idavidrein/gpqa)   | **40.0** |          33.3           |
| 数学 | [AIME24](https://artofproblemsolving.com/wiki/index.php/2024_AIME_I) | **20.6** |          16.7           |
|    | [AIME25](https://artofproblemsolving.com/wiki/index.php/2025_AIME_I) | **22.7** |           7.2           |

### RLVR
带有可验证奖励的强化学习（RLVR）在提升大语言模型（LLM）的推理能力方面展现了巨大的潜力。然而，由于缺乏高质量的可验证数据，其在知识密集型领域的应用尚未得到有效探索。通过利用 **GraphGen** 进行自动化的可验证数据合成，我们将 RLVR 扩展到了这些更广泛的领域。我们使用合成的数据，直接在 Qwen2.5-7B base 模型上进行强化学习，且不使用任何前置 SFT ([代码](https://github.com/superfarther/K2V))。结果如下：
| 领域 |                            数据集                            |  我们的方案   | Qwen2.5-7B-Instruct（基线） |
|:--:|:---------------------------------------------------------:|:--------:|:-----------------------:|
| 植物 | [SeedBench](https://github.com/open-sciencelab/SeedBench) | **66.8** |          51.5           |
| 法律 |   [LawBench](https://github.com/open-compass/LawBench)    | **55.2** |         54.76           |
| 医学 |       [MedQA](https://github.com/jind11/MedQA)            | **87.1** |          80.7           |
| 通用 | [BBH](https://github.com/suzgunmirac/BIG-Bench-Hard)      | **55.3** |          49.6           |

更多细节见 [`examples/generate/generate_masked_fill_in_blank_qa`](./examples/generate/generate_masked_fill_in_blank_qa)。

## ⚙️ 支持列表

我们支持多种 LLM 推理服务器、API 服务器、推理客户端、输入文件格式、数据模态、输出数据格式和输出数据类型。
可以根据合成数据的需求进行灵活配置。

| 推理服务器                                                                    | API 服务器                                                                        | 推理客户端                                                      | 输入文件格式                                                                                                                                                                                                                                                   | 数据模态          | 输出数据类型                                          |
|--------------------------------------------------------------------------|--------------------------------------------------------------------------------|------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|-------------------------------------------------|
| [![hf-icon]HF][hf]<br>[![sg-icon]SGLang][sg]<br>[![vllm-icon]vllm][vllm] | [![sif-icon]Silicon][sif]<br>[![oai-icon]OpenAI][oai]<br>[![az-icon]Azure][az] | HTTP<br>[![ol-icon]Ollama][ol]<br>[![oai-icon]OpenAI][oai] | 文件(CSV, JSON, JSONL, PDF, TXT等)<br>数据库([![uniprot-icon]UniProt][uniprot], [![ncbi-icon]NCBI][ncbi], [![rnacentral-icon]RNAcentral][rnacentral])<br>搜索引擎([![bing-icon]Bing][bing], [![google-icon]Google][google])<br>知识图谱([![wiki-icon]Wikipedia][wiki]) | TEXT<br>IMAGE | Aggregated<br>Atomic<br>CoT<br>Multi-hop<br>VQA |

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


## 🚀 快速开始

通过 [Huggingface](https://huggingface.co/spaces/chenzihong/GraphGen) 或 [Modelscope](https://modelscope.cn/studios/chenzihong/GraphGen) 体验 GraphGen。

如有任何问题，请查看 [FAQ](https://github.com/open-sciencelab/GraphGen/issues/10)、提交新的 [issue](https://github.com/open-sciencelab/GraphGen/issues) 或加入我们的[微信群](https://cdn.vansin.top/internlm/dou.jpg)咨询。

### 准备工作

1. 安装 [uv](https://docs.astral.sh/uv/reference/installer/)

    ```bash
    # 若遇到网络问题，可尝试使用 pipx 或 pip 安装 uv，详见 uv 文档
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2. 克隆仓库

    ```bash
    git clone --depth=1 https://github.com/open-sciencelab/GraphGen
    cd GraphGen
    ```
3. 创建新的 uv 环境

    ```bash
    uv venv --python 3.10
    ```
4. 安装依赖

    ```bash
    uv pip install -r requirements.txt
    ```

### 运行 Gradio 演示

   ```bash
   python -m webui.app
   ```


![ui](https://github.com/user-attachments/assets/3024e9bc-5d45-45f8-a4e6-b57bd2350d84)

### 从 PyPI 运行

1. 安装 GraphGen
   ```bash
   uv pip install graphg
   ```

2. CLI 运行
    ```bash
    SYNTHESIZER_MODEL=your_synthesizer_model_name \
    SYNTHESIZER_BASE_URL=your_base_url_for_synthesizer_model \
    SYNTHESIZER_API_KEY=your_api_key_for_synthesizer_model \
    TRAINEE_MODEL=your_trainee_model_name \
    TRAINEE_BASE_URL=your_base_url_for_trainee_model \
    TRAINEE_API_KEY=your_api_key_for_trainee_model \
    graphg --output_dir cache
    ```

### 源码运行

1. 配置环境
   - 在项目根目录创建 `.env` 文件
     ```bash
     cp .env.example .env
     ```
   - 设置以下环境变量：
     ```bash
      # Tokenizer
      TOKENIZER_MODEL=
      
      # LLM
      # 支持不同的后端：http_api、openai_api、ollama_api、ollama、huggingface、tgi、sglang、tensorrt
      # Synthesizer 用于构建知识图谱并生成数据
      # Trainee 用于使用生成数据进行训练

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
2. （可选）如需修改默认生成配置，可编辑 `config.yaml` 文件。

   例如：

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

      # 其他设置...
    ```

3. 生成数据

   选择所需格式并运行对应脚本：
   
   | 格式              | 运行脚本                                                                         | 说明              |
   |-----------------|------------------------------------------------------------------------------|-----------------|
   | `cot`           | `bash examples/generate/generate_cot_qa/generate_cot.sh`                     | 思维链问答对          |
   | `atomic`        | `bash examples/generate/generate_atomic_qa/generate_atomic.sh`               | 覆盖基础知识的原子问答对    |
   | `aggregated`    | `bash examples/generate/generate_aggregated_qa/generate_aggregated.sh`       | 整合复杂知识的聚合问答对    |
   | `multi_hop`     | `bash examples/generate/generate_multi_hop_qa/generate_multi_hop.sh`         | 多跳推理问答对         |
   | `vqa`           | `bash examples/generate/generate_vqa/generate_vqa.sh`                        | 视觉问答对，结合视觉和文本理解 |
   | `multi_choice`  | `bash examples/generate/generate_multi_choice_qa/generate_multi_choice.sh`   | 单项选择题问答对        |
   | `multi_answer`  | `bash examples/generate/generate_multi_answer_qa/generate_multi_answer.sh`   | 多项选择题问答对        |
   | `fill_in_blank` | `bash examples/generate/generate_fill_in_blank_qa/generate_fill_in_blank.sh` | 填空题问答对          |
   | `true_false`    | `bash examples/generate/generate_true_false_qa/generate_true_false.sh`       | 判断题问答对          |


4. 查看生成结果
   ```bash
   ls cache/output
   ```

### 使用 Docker 运行
1. 构建镜像
   ```bash
   docker build -t graphgen .
   ```
2. 启动容器
   ```bash
    docker run -p 7860:7860 graphgen
    ```


## 🏗️ 系统架构
参阅 deepwiki 的[分析](https://deepwiki.com/open-sciencelab/GraphGen)了解 GraphGen 系统、架构与核心功能的技术概览。


### 工作流程
![workflow](assets/flow.png)


## 🍀 致谢
- [SiliconFlow](https://siliconflow.cn) 提供丰富的 LLM API，部分模型免费
- [LightRAG](https://github.com/HKUDS/LightRAG) 简单高效的图检索方案
- [ROGRAG](https://github.com/tpoisonooo/ROGRAG) 鲁棒优化版 GraphRAG 框架
- [DB-GPT](https://github.com/eosphoros-ai/DB-GPT) AI 原生数据应用开发框架


## 📚 引用
如果本项目对你有帮助，请考虑引用我们的工作：
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

## 📜 许可证
本项目采用 [Apache License 2.0](LICENSE) 许可证。

## 📅 星标历史

[![Star History Chart](https://star-history.dera.page/svg?repos=InternScience/GraphGen&type=Date)](https://star-history.dera.page/#InternScience/GraphGen&Date)


## 自维护 Fork：多项目 SDK 与事件 Profile

本仓库同时维护 `Leon-Algo/GraphGen` fork。跨项目可复用能力、项目自有事件抽取 profile、安装与兼容边界见[多项目 SDK 指南](docs/MULTI_PROJECT_SDK.md)；不可变版本/tag 与发布验证规则见[发布规范](docs/RELEASING.md)；fork 补丁及 SAG 来源归属见[补丁台账](docs/PMS_PATCHES.md)。

Fork 能力均显式 opt-in：默认 LightRAG + ECE 管线保持不变，PMS 专用提示词不会自动成为其他项目的策略。
