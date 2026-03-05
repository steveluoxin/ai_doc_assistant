import os
from openai import OpenAI

def load_document(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
)

BASE_PROMPT = """
    【角色】
    你是一名资深 AI 产品分析助手，擅长从产品文档中提炼关键信息。

    【产品文档】
    ---
    {document}
    ---

    【用户问题】
    {question}
    """

TASK_PROMPTS = {
    "summary": {
        "task_instruction": "请总结该产品的核心功能。",
        "output_requirement": """
            - 只输出 3 条要点
            - 每条不超过 2 行
            - 使用编号 1 / 2 / 3
            """,
    },
    "risk": {
        "task_instruction": "请分析该产品可能存在的风险或问题。",
        "output_requirement": """
        - 从产品或使用角度分析
        - 输出 3 条主要风险
        - 使用编号 1 / 2 / 3
        """,
    },
    "advice": {
        "task_instruction": "请给出针对该产品的改进建议。",
        "output_requirement": """
        - 建议要具体、可执行
        - 输出 3 条
        - 使用编号 1 / 2 / 3
        """,
    },
}

def detect_task(question: str) -> str:
    """
    根据用户问题，粗略判断当前任务类型
    """
    q = question.lower()

    if "风险" in question or "risk" in q or "问题" in question:
        return "risk"
    if "建议" in question or "advice" in q or "改进" in question:
        return "advice"

    # 默认任务
    return "summary"

def build_prompt(task: str, question: str, document: str) -> str:
    task_conf = TASK_PROMPTS[task]
    return f"""
    {BASE_PROMPT.format(document=document, question=question)}

    【当前任务】
    {task_conf["task_instruction"]}

    【输出要求】
    {task_conf["output_requirement"]}
    """


def ask_llm(question: str, document: str) -> str:
    task = detect_task(question)
    prompt = build_prompt(task, question, document)

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个严格遵守指令的 AI 产品分析助手"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=800
    )

    return response.choices[0].message.content.strip()

def main():
    doc_path = input("请输入产品文档路径: \n").strip()
    question = input("请输入您的问题: \n").strip()

    document = load_document(doc_path)

    answer = ask_llm(question, document)
    print("\n【AI回答】：")
    print(answer)


if __name__ == "__main__":
    main()
