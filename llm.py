import os
import json
import re
import requests  # 使用 requests 调用 Deepseek API
from prompts import BASE_PROMPT, TASK_PROMPTS
from schemas import TASK_SCHEMAS
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

def deepseek_request(endpoint: str, payload: dict):
    """统一 Deepseek API 请求"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    url = f"{DEEPSEEK_BASE_URL}/{endpoint}"
    resp = requests.post(url, headers=headers, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()

def get_embedding(text: str):
    """调用 Deepseek embedding（本地/线上都用这个函数）"""
    payload = {"model": "deepseek-embedding", "input": text}
    response = deepseek_request("v1/embeddings", payload)
    return response["data"][0]["embedding"]

SYSTEM_PROMPT = """
你是一个严格的 AI 接口实现助手。
你必须：
- 只输出 JSON
- 不使用 Markdown
- 不添加任何解释性文字
- 确保 JSON 可被程序直接解析
"""

def call_llm(prompt: str, task: str, timeout_sec: int = 10):
    schema_fallback = TASK_SCHEMAS.get(task, {})

    if task == 'rewrite':
        # Special handling for rewrite: return string
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,
            "max_tokens": 800
        }
        try:
            response = deepseek_request("v1/chat/completions", payload)
            return response["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""  # return empty string on failure

    # Original behavior for other tasks
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 800
    }

    try:
        response = deepseek_request("v1/chat/completions", payload)
        raw_output = response["choices"][0]["message"]["content"].strip()
        result = parse_json_safe(raw_output)
        # schema兜底
        for key, default in schema_fallback.items():
            if key not in result or not isinstance(result[key], type(default)):
                result[key] = default
        return result
    except Exception:
        return schema_fallback  # 失败返回兜底

def parse_json_safe(raw_text: str):
    try:
        return json.loads(raw_text)
    except Exception:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return {}

def call_llm_with_retry(prompt: str, task: str, retries: int = 1):
    for attempt in range(retries + 1):
        try:
            return call_llm(prompt, task)
        except Exception as e:
            if attempt == retries:
                return TASK_SCHEMAS.get(task, {})
