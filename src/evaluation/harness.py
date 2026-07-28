"""对比实验评估框架。

统一封装不同 Agent 架构（单智能体 single / 多智能体编排 multi）、
不同开关（知识图谱 KG、校验轮次 verify）的批量运行、
异常捕获、指标聚合、结果格式化输出、命令行启动入口
"""

from __future__ import annotations

from dataclasses import dataclass  # 数据类装饰器，简化纯配置 / 数据实体类，无需手写__init__、__repr__、__eq__、__hash__、__str__
from typing import Any, Callable  # 类型提示，增强代码可读性、可维护性

from src.evaluation.metrics import (
    compare_to_markdown,
    evaluate_report,
    scores_to_radar_text,
)
from src.models.schemas import EvaluationScores, ResearchReport


@dataclass
class ExperimentConfig:
    """单次实验配置。"""
    name: str
    mode: str = "multi"  # single | multi
    max_verify_rounds: int = 2
    enable_kg: bool = True
    chunk_strategy: str = "semantic"


class EvaluationHarness:
    """对同一问题跑多组配置并输出对比。"""
    def __init__(
        self,
        runner: Callable[[ExperimentConfig, str], tuple[ResearchReport, list]],
    ) -> None:
        """
        runner(config, question) -> (report, tasks)
        """
        self._runner = runner

    def run(
        self,
        question: str,
        configs: list[ExperimentConfig] | None = None,
    ) -> dict[str, Any]:
        # 1. 默认消融实验四组配置
        configs = configs or [
            ExperimentConfig(name="multi+verify+kg", mode="multi", max_verify_rounds=2, enable_kg=True),
            ExperimentConfig(name="multi+no_verify", mode="multi", max_verify_rounds=0, enable_kg=True),
            ExperimentConfig(name="multi+no_kg", mode="multi", max_verify_rounds=2, enable_kg=False),
            ExperimentConfig(name="single", mode="single", max_verify_rounds=0, enable_kg=False),
        ]

        results: list[dict[str, Any]] = []
        radars: dict[str, str] = {}

        # 2. 遍历所有配置批量执行
        for cfg in configs:
            print(f"[eval] 运行配置: {cfg.name}")
            try:
                report, tasks = self._runner(cfg, question)
                scores = evaluate_report(report, tasks=tasks)
            except Exception as exc:
                print(f"[eval] 配置 {cfg.name} 失败: {exc}")
                scores = EvaluationScores(details={"error": str(exc)})
                report = ResearchReport(question=question, summary=f"失败: {exc}")

            # 3. 抽取关键指标存入结果字典
            row = {
                "name": cfg.name,
                "completeness": scores.completeness,
                "accuracy": scores.accuracy,
                "citation_density": scores.citation_density,
                "reflection_rounds": scores.reflection_rounds,
                "overall": scores.overall,
                "summary": report.summary[:120],
            }
            results.append(row)
            radars[cfg.name] = scores_to_radar_text(scores)

        # 4. 生成对比表格，统一返回结构化结果
        markdown = compare_to_markdown(results)
        return {
            "question": question,
            "results": results,
            "markdown_table": markdown,
            "radars": radars,
        }


def main() -> None:
    """命令行入口：python -m src.evaluation.harness \"问题\""""
    import argparse
    import json

    from src.factory import create_orchestrator, create_research_agent

    # 1. 命令行参数解析
    parser = argparse.ArgumentParser(description="研究系统评估对比实验")
    parser.add_argument("question", help="研究问题")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="仅跑 multi 与 single 两组，节省时间",
    )
    args = parser.parse_args()

    # 2. 定义runner回调：桥接评估框架与Agent工厂
    def runner(cfg: ExperimentConfig, question: str):
        # 2.1 单智能体直接调用研究代理
        if cfg.mode == "single":
            agent = create_research_agent()
            report = agent.research(question, session_id=f"eval_{cfg.name}")
            return report, []
        # 2.2 多智能体编排调用 Orchestrator
        orch = create_orchestrator(
            max_verify_rounds=cfg.max_verify_rounds,
            enable_kg=cfg.enable_kg,
        )
        report = orch.research(question, session_id=f"eval_{cfg.name}")
        return report, orch.last_tasks

    # 3. quick模式精简运行配置
    configs = None
    if args.quick:
        configs = [
            ExperimentConfig(name="multi+verify", mode="multi", max_verify_rounds=1),
            ExperimentConfig(name="single", mode="single", max_verify_rounds=0),
        ]

    # 4. 实例化调度器、执行评估、打印全部输出
    harness = EvaluationHarness(runner)
    output = harness.run(args.question, configs=configs)
    print("\n" + output["markdown_table"] + "\n")
    for name, radar in output["radars"].items():
        print(f"### {name}\n{radar}\n")
    
    # ensure_ascii=False：中文不转 unicode 编码，控制台直接显示中文分数、配置名
    # indent=2：格式化缩进，方便复制、保存实验日志
    print(json.dumps(output["results"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
