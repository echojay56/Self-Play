import openai
import json
import os
import random
import time
import argparse # 引入 argparse
import logging # 引入 logging
from typing import List, Dict, Tuple

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 配置信息 --- (部分移至 argparse)
# 从环境变量加载 API 密钥

# --- 常量定义 (将提示词设为常量，便于管理) ---

# 系统消息
# 系统消息 (更加具体和定向)
LEARNER_SYSTEM_PROMPT_CN = "你是一位富有创造力的短视频内容创作者，熟悉各大平台特性和内容趋势。你思维活跃，擅长提出有深度、具体且挑战常规思维的问题。你的问题应当反映真实创作者在实践中遇到的痛点和思考。"

EXPERT_SYSTEM_PROMPT_CN = "你是一位世界级的短视频领域专家，拥有丰富的创作、运营和商业化经验。你的回答应当系统全面、前瞻创新、高度可执行，并包含具体案例与数据支持。你的专业知识覆盖内容创作、流量增长、变现策略和行业发展趋势等各个方面。"

JUDGE_SYSTEM_PROMPT_CN = "你是一位极其严苛的短视频专业知识评估专家，以超高标准审视每一个回答。你必须敏锐地识别出回答中的薄弱环节、逻辑漏洞、过时观点或缺乏深度的部分。你的评判应当具体、直接且富有建设性，避免任何形式的空洞赞美。你的职责是确保专业建议的实用性和卓越性。"

# --- 提示词生成函数 (增强版) ---
# --- 初始化 OpenAI ---
try:
    # 从环境变量加载 API 密钥
    # 初始化 OpenAI 客户端 (推荐使用新版 SDK)
    client = openai.OpenAI(api_key="11223434", base_url="http://192.168.1.8:8080/v1")  # 会自动使用环境变量中的 key
except ValueError as e:
    print(f"错误：{e}")
    exit(1)
except Exception as e:
    print(f"初始化 OpenAI 时发生意外错误: {e}")
    exit(1)

def get_learner_prompt_cn() -> str:
    """生成学习者角色的提问提示词 (增强多样性引导)"""
    levels = ["初入行的", "有1-2年经验的", "拥有3-5年经验的", "资深且寻求突破的"]
    domains = [
        "短视频内容创意与差异化定位",
        "视频剪辑技巧与视觉表现力提升",
        "算法机制理解与流量增长策略",
        "私域流量构建与商业化变现",
        "直播带货与互动技巧优化",
        "数据分析与内容迭代方法论",
        "多平台内容矩阵运营与效率提升",
        "行业趋势把握与长期发展规划"
    ]
    scenarios = [
        "水果短视频",
        "水果电商卖货",
    ]
    level = random.choice(levels)
    domain = random.choice(domains)
    scenario = random.choice(scenarios)
    question_styles = [
        f"请针对{domain}领域，提出一个我目前遇到的最具挑战性的实操问题，这个问题应该具体到特定场景并需要专业深入的解答。",
        f"在{domain}领域，许多创作者普遍存在一个错误认知或关键盲点。请提出一个能揭示这种认知差距的深度问题。",
        f"作为一名{scenario}，我在实践{domain}时面临两难选择：是坚持内容质量还是迎合算法偏好？请构建一个具体的决策困境并寻求高水平指导。",
        f"在竞争激烈的短视频市场中，关于{domain}，我想了解如何突破同质化内容的困境，实现真正的差异化增长。请提出一个直指核心的战略性问题。",
    ]


    style_template = random.choice(question_styles)
    question_prompt_base = style_template.format(domain=domain, scenario=scenario)

    return (
        f"假设你是一位专注于短视频创作的{level}{scenario}，"
        f"{question_prompt_base} "
        f"请确保这个问题：\n"
        f"1. 反映出你在实际创作和运营中遇到的具体障碍或深层次思考\n"
        f"2. 足够具体，能引发专业讨论而非泛泛而谈\n"
        f"3. 有一定深度和挑战性，能推动行业认知进步\n"
        f"直接输出问题本身，不需要任何前言或说明。问题应当表达清晰、结构完整，如同一位真实创作者在行业社区提出的高质量问题。"
    )

def get_expert_initial_prompt_cn(question: str) -> str:
    """生成专家首次回答的提示词"""
    return (
        f"一位短视频创作者提出了以下问题：\n"
        f"--- 问题开始 ---\n{question}\n--- 问题结束 ---\n\n"
        f"请以世界级短视频领域专家的身份，提供一个极具价值的专业回答。你的回答必须：\n"
        f"1. 展现系统化思维：建立清晰框架，涵盖问题各个层面，从根本原理到实操细节\n"
        f"2. 给出前瞻性洞见：不仅解决当下问题，还要预见行业发展趋势和潜在机会\n"
        f"3. 提供具体可执行步骤：详细到可立即实施的行动计划，而非空泛建议\n"
        f"4. 融入实战案例：引用真实成功案例或数据支持你的观点\n"
        f"5. 避免陈词滥调：不要重复创作者可能已经听过无数次的基础建议\n\n"
        f"你的目标是提供一个让创作者眼前一亮的专业回答，内容深度和价值远超一般网络搜索结果。请保持语言精炼、结构清晰，确保每句话都承载实质性价值。"
    )


def get_judge_prompt_cn(question: str, answer: str) -> str:
    """生成评判者评估回答的提示词 (增强要求)"""
    return (
        f"请以最高行业标准，严格评估以下专家回答的质量。\n\n"
        f"--- 原始问题 ---\n{question}\n--- 问题结束 ---\n\n"
        f"--- 专家回答 ---\n{answer}\n--- 回答结束 ---\n\n"
        f"请对回答进行多维度深入评估：\n\n"
        f"1. **专业深度与洞察力**\n"
        f"   • 回答是否展现了远超普通从业者的专业认知？\n"
        f"   • 是否提供了非显而易见的深度洞察或突破性思路？\n"
        f"   • 是否揭示了问题背后的根本原理和核心逻辑？\n\n"

        f"2. **实操价值与可执行性**\n"
        f"   • 建议是否具体到可直接执行的步骤和方法？\n"
        f"   • 是否考虑了不同资源条件和能力水平的实施路径？\n"
        f"   • 是否提供了预期效果和衡量成功的具体指标？\n\n"

        f"3. **创新性与前瞻性**\n"
        f"   • 观点是否超越了行业共识，提供了差异化思路？\n"
        f"   • 是否考虑了短视频领域的未来发展趋势？\n"
        f"   • 建议是否能帮助创作者实现长期竞争优势？\n\n"

        f"4. **针对性与完整性**\n"
        f"   • 回答是否精准理解并全面解决了问题的各个层面？\n"
        f"   • 是否考虑了问题背后可能隐含的需求？\n"
        f"   • 回答结构是否完整，逻辑是否严密连贯？\n\n"

        f"**评判要求：**\n"
        f"1. 必须指出2-3个最关键的具体缺陷或改进空间，并详细说明原因\n"
        f"2. 每个缺陷必须配有明确的、可立即实施的改进建议或追问方向\n"
        f"3. 评判必须具体精准，避免模糊评价（如'不够深入'、'可以更好'）\n"
        f"4. 如发现专业性错误或重大逻辑漏洞，必须明确指出\n\n"

    f"你的评判目标是推动专家产出真正卓越、无可挑剔的高水平回答，而非简单肯定或否定。每一条批评必须切中要害，每一条建议必须有实质性提升价值。"
    )


def get_expert_refinement_prompt_cn(question: str, previous_answer: str, critique: str) -> str:
    """生成专家根据评判进行优化的提示词 (增强要求)"""
    return (
        f"请审视以下原始问题、你的上一版回答以及评估意见：\n"
        f"--- 问题 ---\n{question}\n--- 问题结束 ---\n\n"
        f"--- 你的上一版回答 ---\n{previous_answer}\n--- 回答结束 ---\n\n"
        f"--- 专业评估意见 ---\n{critique}\n--- 评估意见结束 ---\n\n"

        f"**改进要求：**\n"
        f"1. 针对评估中指出的每一个具体问题，进行深入思考和根本性改进\n"
        f"2. 不要简单修补或表面调整，而应考虑结构性优化或重大内容增补\n"
        f"3. 提升回答的深度、实用性和创新性，使其真正达到世界顶级专家水准\n"
        f"4. 确保每个建议都具体、可执行，避免抽象表述\n"
        f"5. 优化论述结构和表达方式，使内容更加清晰有力\n\n"

        f"**改进过程：**\n"
        f"- 先分析评估意见中的核心批评点\n"
        f"- 对每个批评点进行深度思考，找出根本问题\n"
        f"- 针对性提出更高质量、更具深度的内容\n"
        f"- 重构回答，确保逻辑连贯、结构优化\n\n"

        f"你的目标是创造一个质量飞跃的新版回答，使其在专业深度、实用价值和创新洞见上都达到无懈可击的水平。请直接输出完整的优化后回答，不需要解释你的修改过程。"
    )

# --- OpenAI API 交互封装 (增加日志) ---

def call_llm(system_message: str, user_prompt: str, max_tokens: int, model_name: str, temperature: float) -> str:
    """调用 OpenAI API 并返回响应内容，增加日志记录"""
    logging.debug(f"向模型 {model_name} 发送请求。 System Prompt: {system_message[:100]}... User Prompt: {user_prompt[:100]}...")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
            #max_tokens=max_tokens,
            temperature=temperature,
            n=1,
            stop=None,
        )
        if response and response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content.strip()
            if content:
                logging.debug(f"模型 {model_name} 成功返回内容。")
                return content
            else:
                logging.warning(f"模型 {model_name} 返回了空内容。 User Prompt: {user_prompt[:200]}...")
                return "错误：LLM 返回空响应。"
        else:
            logging.warning(f"来自模型 {model_name} 的响应结构无效。 User Prompt: {user_prompt[:200]}...")
            return "错误：无效的响应结构。"

    except openai.APIError as e:
        logging.error(f"OpenAI API 错误: {e}. Model: {model_name}, User Prompt: {user_prompt[:200]}...")
        # 简单的重试逻辑
        time.sleep(5)
        try:
             logging.info("正在重试 API 调用...")
             response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt},
                ],
                #max_tokens=max_tokens,
                temperature=temperature,
                n=1,
                stop=None,
            )
             if response and response.choices and len(response.choices) > 0:
                 content = response.choices[0].message.content.strip()
                 if content:
                    logging.info("API 调用重试成功。")
                    return content
                 else:
                    logging.warning(f"重试后模型 {model_name} 仍返回空内容。")
                    return "错误：重试后LLM返回空响应。"
             else:
                 logging.warning(f"重试后来自模型 {model_name} 的响应结构仍然无效。")
                 return "错误：重试时响应结构无效。"
        except Exception as retry_e:
            logging.error(f"重试时 OpenAI API 错误: {retry_e}. Model: {model_name}, User Prompt: {user_prompt[:200]}...")
            return f"错误：API 错误 - {retry_e}"
    except Exception as e:
        logging.error(f"发生意外错误: {e}. Model: {model_name}, User Prompt: {user_prompt[:200]}...")
        return f"错误：意外错误 - {e}"

# --- Self-Play 核心逻辑 ---

def run_self_play_iteration(model_name: str, temperature: float, max_tokens_answer: int, max_tokens_other: int, num_refinement_iterations: int) -> Dict:
    """运行一轮完整的提问、回答和优化循环"""
    logging.info("-" * 20)
    logging.info("开始新一轮自我对弈迭代...")

    # 1. 生成问题 (学习者角色)
    logging.info("1. 学习者：正在生成问题...")
    learner_prompt = get_learner_prompt_cn()
    question = call_llm(LEARNER_SYSTEM_PROMPT_CN, learner_prompt, max_tokens_other, model_name, temperature)
    if question.startswith("错误：") or not question:
        logging.error("   生成问题失败。跳过此轮迭代。")
        return None # 表示失败
    logging.info(f"   生成的问题: {question}")

    # 2. 初始回答生成 (专家角色)
    logging.info("2. 专家：正在生成初始回答...")
    expert_initial_prompt = get_expert_initial_prompt_cn(question)
    current_answer = call_llm(EXPERT_SYSTEM_PROMPT_CN, expert_initial_prompt, max_tokens_answer, model_name, temperature)
    if current_answer.startswith("错误：") or not current_answer:
        logging.error(f"   为问题 '{question[:100]}...' 生成初始回答失败。跳过此轮迭代。")
        return None # 表示失败
    logging.info(f"   生成的初始回答 (预览): {current_answer[:500]}...")

    # 存储优化历史
    refinement_history = [{"answer": current_answer, "critique": "初始回答"}] # 标记初始回答

    # 3. 自我评判与优化循环
    for i in range(num_refinement_iterations):
        logging.info(f"3. 开始第 {i+1}/{num_refinement_iterations} 轮优化...")

        # 3a. 自我评估 (评判者角色)
        logging.info("   3a. 评判者：正在生成评判意见...")
        judge_prompt = get_judge_prompt_cn(question, current_answer)
        critique = call_llm(JUDGE_SYSTEM_PROMPT_CN, judge_prompt, max_tokens_other, model_name, temperature)
        if critique.startswith("错误：") or not critique:
            logging.warning("      生成评判意见失败。停止此条目的优化。")
            # 即使评判失败，也记录下尝试和失败结果
            refinement_history[-1]["critique"] = f"第{i+1}轮评判失败: {critique}"
            break # 如果评判失败则停止优化

        logging.info(f"      生成的评判意见 (预览): {critique[:500]}...")
        # 更新上一轮的评判意见
        refinement_history[-1]["critique"] = critique

        # 3b. 优化回答 (专家角色)
        logging.info("   3b. 专家：正在根据评判生成优化后的回答...")
        expert_refinement_prompt = get_expert_refinement_prompt_cn(question, current_answer, critique)
        refined_answer = call_llm(EXPERT_SYSTEM_PROMPT_CN, expert_refinement_prompt, max_tokens_answer, model_name, temperature)
        if refined_answer.startswith("错误：") or not refined_answer:
             logging.warning("      生成优化后回答失败。停止此条目的优化。")
             # 记录优化失败
             refinement_history.append({"answer": f"第{i+1}轮优化失败: {refined_answer}", "critique": None})
             break # 如果优化失败则停止

        logging.info(f"      生成的优化后回答 (预览): {refined_answer[:500]}...")
        current_answer = refined_answer
        # 添加新的优化步骤，critique 暂时为 None，将在下一轮或结束时填充
        refinement_history.append({"answer": current_answer, "critique": f"第 {i+1} 轮优化后的回答"})

    # 4. 准备结果
    # 移除最后一个answer条目多余的critique标记
    if len(refinement_history) > 1 and refinement_history[-1]["critique"] is not None and refinement_history[-1]["critique"].endswith("优化后的回答"):
         refinement_history[-1]["critique"] = "最终优化版本"


    final_data_entry = {
        "question": question,
        "initial_answer": refinement_history[0]['answer'], # 单独提出初始回答
        "final_answer": current_answer,
        "refinement_process": refinement_history # 包含初始回答和每次的评判与优化结果
    }
    logging.info("本轮迭代完成。")
    logging.info("-" * 20)
    return final_data_entry

# --- 主程序执行 (使用 argparse) ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 Self-Play 方法生成短视频 AI 专家数据集")
    parser.add_argument("--model", type=str, default="/data/wangfeng/model_cache/qwen/Qwen2___5-32B-Instruct", help="要使用的 OpenAI 模型名称 (例如 gpt-3.5-turbo, gpt-4)")
    parser.add_argument("--num_entries", type=int, default=20, help="要生成的数据集条目数量")
    parser.add_argument("--refinement_iterations", type=int, default=3, help="每条数据的优化迭代次数")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM 的采样温度 (0.0-1.0)")
    parser.add_argument("--max_tokens_answer", type=int, default=16000, help="专家回答的最大 token 数")
    parser.add_argument("--max_tokens_other", type=int, default=512, help="问题、评判等其他内容的最大 token 数")
    parser.add_argument("--output_file", type=str, default="short_video_ai_expert_dataset_zh_v5.json", help="输出 JSON 数据集的文件名")
    parser.add_argument("--max_attempts_multiplier", type=int, default=2, help="每条数据允许的最大尝试次数乘数 (乘以 num_entries)")

    args = parser.parse_args()

    logging.info(f"开始生成数据集...")
    logging.info(f"配置: 模型={args.model}, 条目数={args.num_entries}, 优化轮数={args.refinement_iterations}, 温度={args.temperature}")
    logging.info(f"输出文件: {args.output_file}")

    dataset = []
    attempts = 0
    max_total_attempts = args.num_entries * args.max_attempts_multiplier

    while len(dataset) < args.num_entries and attempts < max_total_attempts:
        attempts += 1
        logging.info(f"\n--- 正在生成第 {len(dataset) + 1}/{args.num_entries} 条数据 (总尝试次数 {attempts}/{max_total_attempts}) ---")
        entry = run_self_play_iteration(
            model_name=args.model,
            temperature=args.temperature,
            max_tokens_answer=args.max_tokens_answer,
            max_tokens_other=args.max_tokens_other,
            num_refinement_iterations=args.refinement_iterations
        )
        if entry: # 仅当迭代成功时添加
            dataset.append(entry)
            logging.info(f"成功生成第 {len(dataset)} 条数据。")
        else:
            logging.warning("本轮迭代失败，将尝试重新生成。")
        # 可选延迟
        time.sleep(1) # 稍微减少延迟，如果API不限速

    if len(dataset) < args.num_entries:
        logging.warning(f"\n警告：已达到最大尝试次数 ({max_total_attempts})，但只成功生成了 {len(dataset)} 条数据。")
    else:
        logging.info(f"\n成功生成了 {len(dataset)} 条数据。")

    # 将数据集保存到 JSON 文件
    logging.info(f"数据生成结束。正在将数据集保存到 {args.output_file}...")
    try:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
        logging.info(f"数据集已成功保存到 {args.output_file}。")
    except Exception as e:
        logging.error(f"将数据集保存到 JSON 时出错: {e}")

    logging.info("脚本执行完毕。")