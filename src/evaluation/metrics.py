"""研究报告自动评估指标。"""

from __future__ import annotations

import re
from typing import Any

from src.models.schemas import EvaluationScores, ResearchReport, ResearchTask


def evaluate_report(
    report: ResearchReport,
    tasks: list[ResearchTask] | None = None,
    evidence_urls: set[str] | None = None,
) -> EvaluationScores:
    """
    计算评估分数（0-100）。

    - completeness: 子任务覆盖 / 发现与章节充实度
    - accuracy: 引用 URL 是否出现在证据来源中
    - citation_density: 每 100 字引用密度（归一化）
    - reflection_rounds: 自反思轮次得分
    """
    tasks = tasks or []
    evidence_urls = evidence_urls or set()

    # 从任务来源补全证据 URL
    for task in tasks:
        for src in task.sources:
            url = src.get("url")
            if url:
                evidence_urls.add(url)

    completeness = _score_completeness(report, tasks)
    accuracy = _score_accuracy(report, evidence_urls)
    citation_density = _score_citation_density(report)
    reflection = _score_reflection(report.verify_rounds)

    overall = (
        completeness * 0.35
        + accuracy * 0.30
        + citation_density * 0.20
        + reflection * 0.15
    )

    return EvaluationScores(
        completeness=round(completeness, 2),
        accuracy=round(accuracy, 2),
        citation_density=round(citation_density, 2),
        reflection_rounds=round(reflection, 2),
        overall=round(overall, 2),
        details={
            "task_count": len(tasks),
            "citation_count": len(report.citations),
            "finding_count": len(report.key_findings),
            "verify_rounds": report.verify_rounds,
            "mode": report.mode,
        },
    )


def _score_completeness(report: ResearchReport, tasks: list[ResearchTask]) -> float:
    if tasks:
        done = sum(1 for t in tasks if t.status.value == "done")
        coverage = done / max(len(tasks), 1)
        finding_bonus = min(len(report.key_findings) / max(len(tasks), 1), 1.0)
        unresolved_penalty = min(len(report.unresolved_questions) * 0.08, 0.3)
        return max(0.0, min(100.0, (coverage * 70 + finding_bonus * 30) - unresolved_penalty * 100))

    # 单 Agent 无任务时基于报告本身
    base = 40.0
    base += min(len(report.key_findings) * 10, 40)
    base += 10 if report.summary else 0
    base -= min(len(report.unresolved_questions) * 5, 20)
    return max(0.0, min(100.0, base))


def _score_accuracy(report: ResearchReport, evidence_urls: set[str]) -> float:
    if not report.citations:
        return 30.0 if evidence_urls else 0.0

    valid = 0
    for cite in report.citations:
        if not cite.url:
            continue
        if cite.url in evidence_urls or any(
            cite.url in u or u in cite.url for u in evidence_urls
        ):
            valid += 1
    ratio = valid / max(len(report.citations), 1)
    return ratio * 100.0


def _score_citation_density(report: ResearchReport) -> float:
    text = " ".join(
        [report.summary, *report.key_findings, *report.sections]
    )
    text = text.strip()
    if not text:
        return 0.0
    # 统计 [1][2] 类引用标记与 citations 数
    markers = len(re.findall(r"\[\d+\]", text))
    chars = max(len(text), 1)
    density_per_100 = (markers / chars) * 100
    # 0.5+/100字视为较充足；映射到 0-100
    score = min(density_per_100 / 0.8, 1.0) * 70
    score += min(len(report.citations) * 5, 30)
    return max(0.0, min(100.0, score))


def _score_reflection(rounds: int) -> float:
    if rounds <= 0:
        return 40.0
    if rounds == 1:
        return 70.0
    return 100.0


def scores_to_radar_text(scores: EvaluationScores, width: int = 20) -> str:
    """生成简易文本雷达/条形图。"""
    items = [
        ("完整性", scores.completeness),
        ("准确性", scores.accuracy),
        ("引用密度", scores.citation_density),
        ("自反思", scores.reflection_rounds),
        ("综合", scores.overall),
    ]
    lines = ["评估雷达（文本条形）:"]
    for name, value in items:
        filled = int((value / 100) * width)
        bar = "█" * filled + "░" * (width - filled)
        lines.append(f"  {name:<6} |{bar}| {value:5.1f}")
    return "\n".join(lines)


def compare_to_markdown(rows: list[dict[str, Any]]) -> str:
    """对比实验结果转 Markdown 表。"""
    if not rows:
        return "_无实验结果_"
    headers = [
        "配置",
        "完整性",
        "准确性",
        "引用密度",
        "自反思",
        "综合",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("name", "")),
                    f"{row.get('completeness', 0):.1f}",
                    f"{row.get('accuracy', 0):.1f}",
                    f"{row.get('citation_density', 0):.1f}",
                    f"{row.get('reflection_rounds', 0):.1f}",
                    f"{row.get('overall', 0):.1f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)
