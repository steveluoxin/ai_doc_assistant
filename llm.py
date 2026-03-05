from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI
from prompts import BASE_PROMPT, TASK_PROMPTS
import json
import re
from schemas import TASK_SCHEMAS
import signal

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com"
)

def get_embedding(text: str):
    """调用 DeepSeek API 获取 embedding"""
    response = client.embeddings.create(
        model="deepseek-embedding",
        input=text
    )
    return response.data[0].embedding

SYSTEM_PROMPT = """
    你是一个严格的 AI 接口实现助手。
    你必须：
    - 只输出 JSON
    - 不使用 Markdown
    - 不添加任何解释性文字
    - 确保 JSON 可被程序直接解析
    """

# 超时控制部分（关键）
class LLMTimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise LLMTimeoutError("LLM 调用超时")

def parse_json_safe(raw_text: str):
    # 尝试直接解析
    try:
        return json.loads(raw_text)
    except Exception:
        pass

     # 尝试提取第一对大括号 {...}
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {}

def call_llm(prompt: str, task: str, timeout_sec: int = 10): 
    # 带超时保护的 LLM 调用
    # - 超时直接返回 schema 兜底
    # - 永远不抛异常给上层
    schema_fallback = TASK_SCHEMAS.get(task, {})
    # rewrite 任务不使用 JSON schema，不使用严格 SYSTEM_PROMPT
    if task == "rewrite":
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800
        )
        return resp.choices[0].message.content.strip()
    # 注册超时信号
    # signal.signal(signal.SIGALRM, _timeout_handler)
    # signal.alarm(timeout_sec)

    try:
        resp = client.chat.completions.create(
            model = "deepseek-chat",
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature = 0.5,
            max_tokens = 800
        )
        raw_output = resp.choices[0].message.content.strip()
        result = parse_json_safe(raw_output)   
        # schema兜底
        for key, default in schema_fallback.items():
            if key not in result or not isinstance(result[key], type(default)):
                result[key] = default
        return result
    except Exception as e:
        raise e
    finally:
        pass
    #     # 取消超时信号
    #     signal.alarm(0)

def call_llm_with_retry(prompt: str, task: str, retries: int = 1):
    for attempt in range(retries + 1):
        try:
            return call_llm(prompt, task)
        except Exception as e:
            if attempt == retries:
                return TASK_SCHEMAS.get(task, {})
