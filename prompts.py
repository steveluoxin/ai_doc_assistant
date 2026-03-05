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
        "instruction": "请总结产品的核心功能。",
        "schema": {
            "task": "summary",
            "points": [
                {"title": "string", "description": "string"}
            ]
        }
    },
    "risk": {
        "instruction": "请分析产品风险。",
        "schema": {
            "task": "risk",
            "risks": [
                {"risk": "string", "impact": "string"}
            ]
        }
    },
    "advice": {
        "instruction": "请给出针对该产品的改进建议。",
        "schema": {
            "task": "advice",
            "advices": [
                { "advice": "string", "reason": "string"}
            ]
        }
    }
}

def build_prompt(task, document, question):
    if task == "summary":
        schema_desc = """
            请严格返回 JSON，格式如下：
            {
                "points": ["...", "...", "..."]
            }
            """
    elif task == "risk":
        schema_desc = """
            请严格返回 JSON，格式如下：
            {
                "risks": [
                    {"title": "...", "desc": "..."}
                ]
            }
            """
    elif task == "advice":
        schema_desc = """
            请严格返回 JSON，格式如下：
            {
                "advices": ["...", "..."]
            }
            """
    else:
        schema_desc = "返回空 JSON {}"

    return f"""
        你是 AI 产品文档助手。

        【产品文档】
        {document}

        【问题】
        {question}

        【输出要求】
        {schema_desc}
        """


def build_merge_prompts(task: str, partial_results: list):
    if task == "summary":
        return f"""
            以下是多个文档片段的总结结果（JSON）：
            {partial_results}
            
            请基于这些结果：
            - 去重
            - 合并相似要点
            - 保留 3～5 条最核心功能点

            严格返回 JSON：
            {{
                "points": ["..."]
            }}

            只输出 JSON，不要解释说明。
        """
    elif task == "risk":
        return f"""
            请基于这些风险结果：
            - 去重（相同或高度相似的风险只保留一条）
            - 合并描述相近的风险
            - 用清晰、具体的语言表述风险点
            - 不要编造文档中不存在的风险

            严格返回 JSON：
            {{
            "risks": [
                {{
                "title": "风险标题",
                "desc": "风险描述"
                }}
            ]
            }}

            只输出 JSON，不要解释说明。
        """
    elif task == "advice":
        return f"""
            以下是多个文档片段中给出的建议列表（JSON）：

            {partial_results}

            请基于这些建议：
            - 去除重复或高度相似的建议
            - 合并表述接近的建议
            - 使用简洁、可执行的语言
            - 如有必要，可按重要性排序（重要的在前）

            严格返回 JSON：
            {{
                "advices": ["建议1", "建议2", "建议3"]
            }}
            只输出 JSON，不要解释说明。
        """
    else:
        return f"""
            以下是模型输出结果（JSON）：

            {partial_results}

            请整理为一个 JSON 对象并返回。
            只输出 JSON。
        """
