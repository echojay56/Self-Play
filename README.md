

---

# 短视频 AI 专家问答数据集生成器 (基于 LLM 自我对弈)

这是一个 Python 脚本，利用大型语言模型 (LLM) 的自我对弈 (Self-Play) 机制，模拟短视频领域的“学习者”、“专家”和“评判者”三个角色，迭代生成高质量、有深度、实操性强的短视频创作与运营相关的问答数据集。

该数据集旨在捕捉真实创作者可能遇到的复杂问题，并通过多轮专家回答和严格评判的循环，产出接近世界级专家水平的解决方案，可用于 LLM 微调、知识库构建或 RAG (Retrieval Augmented Generation) 系统。

## 核心机制：自我对弈 (Self-Play)

脚本的核心是一个迭代循环：

1.  **学习者 (Learner):** 扮演不同经验水平和场景的短视频创作者，提出一个具体、有挑战性、反映实际痛点的问题。
2.  **专家 (Expert):** 针对学习者提出的问题，提供一个系统、全面、具有前瞻性和可执行性的初始回答。
3.  **评判者 (Judge):** 以极高标准严格评估专家的回答，指出具体缺陷、逻辑漏洞或改进空间，并提供建设性意见。
4.  **专家 (Expert - Refinement):** 根据评判者的意见，对上一轮的回答进行深度优化和重构，力求达到更高水平。

这个循环会重复设定的次数 (`--refinement_iterations`)，最终产出经过多轮“专家”与“评判者”博弈和优化的最终回答。

## 功能特性

*   **角色扮演:** 通过精心设计的系统提示词，使 LLM 扮演学习者、专家和评判者，模拟真实的问答和评审过程。
*   **迭代优化:** 支持多轮回答-评判-优化的循环，逐步提升回答质量。
*   **参数配置:** 支持通过命令行参数配置模型、生成条目数、优化轮数、温度等。
*   **结构化输出:** 将生成的问答数据及其优化过程保存为结构化的 JSON 文件。
*   **日志记录:** 详细记录生成过程，便于追踪和调试。
*   **错误处理与重试:** 集成基本的 API 调用错误处理和重试机制。
*   **中文内容:** 提示词和生成内容专注于中文短视频领域的专业知识。

## 环境要求

*   Python 3.6+
*   安装 `openai` 库 (`pip install openai`)
*   一个兼容 OpenAI API 的 LLM 服务端点 (例如：OpenAI 官方 API, 或本地部署的兼容服务如 vLLM, text-generation-webui 等)
*   有效的 API Key (取决于你使用的服务)

## 安装与配置

1. **克隆仓库 (如果适用) 或下载脚本:**

   ```bash
   # 如果是仓库
   git clone <仓库地址>
   cd <项目目录>
   # 如果是单个文件，直接下载
   ```

2. **安装依赖:**

   ```bash
   pip install openai
   ```

3. **配置 LLM API:**
   打开脚本文件 (`your_script_name.py`)，找到以下代码行：

   ```python
   client = openai.OpenAI(api_key="11223434", base_url="http://192.168.1.6:8085/v1")
   ```

   *   将 `base_url` 修改为你实际使用的 LLM 服务端点地址。
   *   将 `api_key` 修改为你实际的 API 密钥。**注意：将 API Key 直接写在代码中不安全，推荐使用环境变量。** 如果你的服务不需要 API Key，可以留空或使用一个占位符。如果使用环境变量，可以修改为 `api_key=os.getenv("OPENAI_API_KEY")` 并在运行脚本前设置环境变量。

## 使用方法

通过命令行运行脚本，并使用参数进行配置：

```bash
python your_script_name.py [OPTIONS]
```

**常用参数说明:**

*   `--model <模型名称>`: 指定用于生成内容的 LLM 模型名称。例如：`gpt-4`, `qwen2-7b-instruct` 等。默认为 `/data/wangfeng/model_cache/qwen/Qwen2___5-32B-Instruct`。
*   `--num_entries <数量>`: 指定要生成的数据集条目数量。默认为 20。
*   `--refinement_iterations <轮数>`: 指定每条数据进行优化的迭代次数。默认为 3。
*   `--temperature <温度>`: 控制 LLM 生成的随机性 (0.0-1.0)。较高的值会产生更具创造性的回答，较低的值则更稳定。默认为 0.7。
*   `--max_tokens_answer <数量>`: 专家回答的最大 token 数。默认为 16000 (适用于长回答)。
*   `--max_tokens_other <数量>`: 问题、评判等其他内容的最大 token 数。默认为 512。
*   `--output_file <文件名>`: 指定输出的 JSON 文件名。默认为 `short_video_ai_expert_dataset_zh_v5.json`。
*   `--max_attempts_multiplier <乘数>`: 每条数据允许的最大尝试次数是 `num_entries * multiplier`。用于控制因 API 错误等原因导致的重试上限。默认为 2。

**示例:**

生成 50 条数据，每条优化 5 轮，使用 `gpt-4` 模型，输出到 `my_dataset.json`:

```bash
python your_script_name.py --num_entries 50 --refinement_iterations 5 --model gpt-4 --output_file my_dataset.json
```

生成 10 条数据，不进行优化 (仅生成初始回答)，使用默认模型：

```bash
python your_script_name.py --num_entries 10 --refinement_iterations 0
```

## 输出格式

脚本将生成一个 JSON 文件，其结构如下：

```json
[
    {
        "question": "学习者提出的原始问题。",
        "initial_answer": "专家针对原始问题生成的第一个回答。",
        "final_answer": "经过多轮优化后，专家最终生成的回答。",
        "refinement_process": [
            {
                "answer": "初始回答内容。",
                "critique": "初始回答的标记，或第一轮评判者对初始回答的意见。"
            },
            {
                "answer": "根据第一轮评判优化后的回答内容。",
                "critique": "第二轮评判者对优化后回答的意见。"
            },
            // ... 更多优化轮次 ...
            {
                "answer": "最终优化后的回答内容。",
                "critique": "最终优化版本标记，或最后一轮评判者对倒数第二轮回答的意见。"
            }
        ]
    },
    // ... 更多问答条目 ...
]
```

`refinement_process` 列表记录了每一轮的回答 (`answer`) 和紧随其后的评判意见 (`critique`)。最后一个条目的 `critique` 可能标记为“最终优化版本”或包含最后一轮的评判意见（取决于循环逻辑）。

## 日志

脚本使用 Python 的 `logging` 模块记录运行过程中的信息、警告和错误。默认日志级别为 `INFO`，会打印每一步的主要进展。你可以修改脚本开头的 `logging.basicConfig(level=logging.INFO, ...)` 来调整日志级别（例如改为 `logging.DEBUG` 获取更详细信息）。

## 潜在应用

*   **LLM 微调:** 使用 `question` 和 `final_answer` 对 LLM 进行微调，使其在短视频领域具备更强的专业问答能力。
*   **知识库/FAQ:** 构建一个高质量的短视频领域知识库或常见问题解答集。
*   **RAG 数据源:** 作为 RAG 系统的数据源，为用户查询提供权威、深入的答案。
*   **AI 助手训练:** 用于训练专注于短视频创作和运营的 AI 助手。
*   **研究:** 研究 LLM 在特定领域通过自我对弈提升知识和推理能力。

## 贡献

欢迎对本项目提出改进意见或贡献代码。

## 许可证

本项目采用 MIT 许可证。详情请见 LICENSE 文件 (如果存在)。

---
