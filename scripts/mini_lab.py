from __future__ import annotations

from typing import Any

CACHE:dict[str,list[str]] = {
    "什么是 RAG？":["RAG = 检索增强生成", "先检索再让模型回答"],
}


def fake_search(query:str) -> dict[str,Any]:
    """假装“真的去联网搜索”（只有未命中缓存时才会走到这里）。"""
    print(f"  [联网] 真实调用 fake_search({query!r})")
    return{
        "success":True,
        "from_cache":False,
        "query":query,
        "results":[f"关于「{query}」的假证据 A", f"关于「{query}」的假证据 B"],
    }

def fake_calc(expr:str) -> dict:
    """假装计算成功，返回固定结构（类似 ToolExecutionResult.data）。"""
    try:
        result = eval(expr)
        return {
            "success":True,
            "expr":expr,
            "result":result,
        }
    except Exception as e:
        return {"success":False,"error":str(e)}

def search_with_cache(query:str) -> dict[str,Any]:
    """B2 核心：先查缓存，再决定是否真搜。"""
    if query in CACHE:
        print(f"  [缓存] 命中，跳过 fake_search({query!r})")
        return {
            "success":True,
            "from_cache":True,
            "query":query,
            "results":CACHE[query],
            "message":"命中本地缓存，已跳过联网搜索",
        }
    return fake_search(query)

def fake_llm_chat(messages:list[dict[str,Any]],round_id:int) -> dict[str,Any]:
    """
    第 1 轮：要求调工具 search
    第 2 轮：看过工具结果后，给出最终回答
    """

    if round_id == 1:
        user_text = messages[-1]["content"]
        return {
            "content": None,
            "tool_calls":[
                {
                    "id":"call_1",
                    "name":"search",
                    "arguments":{"query":user_text},
                }
            ],
        }
    
    tool_msg = next(m for m in messages if m["role"] == "tool")
    return {
        "content":f"最终回答（基于工具结果：{tool_msg['content']}",
        "tool_calls":None,
    }

def mini_research(question:str,max_rounds:int = 3) -> str:
    messages:list[dict[str,Any]] = [
        {"role":"system","content":"你是研究助手，需要时先调用 search。"},
        {"role":"user","content":question},
    ]

    for round_id in range(1,max_rounds + 1):
        response = fake_llm_chat(messages,round_id)

        if not response["tool_calls"]:
            return str(response["content"] or "（空回答）")

        messages.append(
            {
                "role":"assistant",
                "content":response["content"],
                "tool_calls":response["tool_calls"],
            }
        )

        for tc in response["tool_calls"]:
            name = tc["name"]
            args = tc["arguments"]
            
            # B2：search 不再直接 fake_search，而是 search_with_cache
            if name == "search":
                result = search_with_cache(args["query"])
            elif name == "calc":
                result = fake_calc(args.get("expr", ""))
            else:
                result = {"success":False,"error":f"未知工具: {name}"}
            
            messages.append(
                {
                    "role":"tool",
                    "tool_call_id":tc["id"],
                    "content":str(result),
                }
            )

    return "超过最大轮次，仍未得到最终回答"

if __name__ == "__main__":
    print("=== 用例1：问题在 CACHE 里（应打印 [缓存]，不打印 [联网]）===")
    print(mini_research("什么是 RAG？"))
    print()

    print("=== 用例2：问题不在 CACHE 里（应打印 [联网]）===")
    print(mini_research("什么是 Agent？"))