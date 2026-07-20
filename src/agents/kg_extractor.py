"""从证据文本中抽取实体关系三元组。
"""

from __future__ import annotations
# 延迟类型注解，支持类内前向引用

from typing import Any
# 通用任意类型注解

from src.llm.siliconflow import SiliconFlowClient
# 导入自定义大模型客户端
from src.models.schemas import EntityRelation
# 导入Pydantic数据模型，存储三元组结构化结果

# 常量定义
MAX_EVIDENCE_NUM = 8
CONTENT_TRUNCATE_LENGTH = 300
DEFAULT_CONFIDENCE = 0.5

# 全局常量：抽取任务固定Prompt模板
EXTRACT_PROMPT = """从以下研究证据中抽取实体关系三元组。

## 子任务
{question}

## 证据
{evidence}

请严格输出 JSON：
{{
  "relations": [
    {{
      "subject": "实体A",
      "predicate": "关系类型",
      "object": "实体B",
      "evidence_url": "来源URL或空",
      "confidence": 0.0
    }}
  ]
}}

关系类型限定为：竞争、供应、归属、影响、依赖、合作、位于、发布、替代。
实体类型侧重：公司、人物、技术、政策、事件、地区、产品。
最多输出 8 条，不要编造证据中不存在的关系。
""".format(max_num=MAX_EVIDENCE_NUM)

# 实体关系抽取业务类（类名拼写错误 Knowledeg → Knowledge）
class KnowledegExtractor:
    """LLM实体关系抽取器"""
    # 构造方法：注入大模型客户端（依赖注入设计）
    def __init__(self, llm:SiliconFlowClient) -> None:
        self._llm = llm  # 私有实例属性，下划线标识内部变量  仅类内部使用

    # 核心抽取方法
    def extract(
        self,          # 第一个参数必须是self，代表当前类实例
        question:str,
        evidence: list[dict[str, Any]],   # 列表，列表内元素是键为字符串、值任意的字典
    ) -> list[EntityRelation]:             # 函数返回值为自定义模型对象列表
        # 边界判断：无证据直接返回空列表
        if not evidence:
            return []

        # 截取前8条证据，拼接成文本，每条内容截断300字符防止超长
        evidence_text = "\n\n".join(          # "分隔符".join(可迭代对象)：把多条文本用两个换行拼接成一整段证据文本
            f"-{e.get('title', '')}\n URL: {e.get('url', '')}\n"
            f"内容:{str(e.get('content', ''))[:CONTENT_TRUNCATE_LENGTH ]}"
            for e in evidence[:MAX_EVIDENCE_NUM]
        )
        # 填充Prompt模板占位符
        prompt = EXTRACT_PROMPT.format(question=question, evidence=evidence_text)
        
        # LLM调用 + JSON解析异常捕获
        try:
            # 调用封装好的chat_json接口，自动返回字典
            data = self._llm.chat_json([{"role":"user", "content":prompt}])
            # 取不到relations键则赋值空列表
            raw = data.get("relations",[])
        except ValueError:
            # JSON解析失败直接返回空
            return []

        relations: list[EntityRelation] = []
        # 遍历大模型返回的原始三元组
        for item in raw:
            # str.strip()去除字符串首尾空格、换行、制表符，清洗 LLM 输出多余空白字符，避免空实体。
            # 取值+转字符串+去除首尾空白
            subject = str(item.get("subject","")).strip()
            predicate = str(item.get("predicate","")).strip()
            obj = str(item.get("object","")).strip()

            # 实体/关系任一为空则丢弃这条数据
            if not subject or not predicate or not obj:
                continue
            
            # 置信度容错转换，非数字默认0.5
            try:
                confidence = float(item.get("confidence",DEFAULT_CONFIDENCE))
            except (TypeError, ValueError):
                confidence = DEFAULT_CONFIDENCE
            
            # 实例化结构化模型，置信度限制在 [0,1] 区间
            relations.append(
                EntityRelation(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    evidence_url=str(item.get("evidence_url","")),
                    confidence=max(0.0, min(1.0, confidence)),
                )
            )
        # 最多返回8条结果，截断多余数据
        return relations[:MAX_EVIDENCE_NUM]