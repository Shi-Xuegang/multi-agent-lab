"""研究过程 Mermaid 图生成。"""

from __future__ import annotations

from src.models.schemas import ResearchTask


def _safe_id(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in raw)[:40]


def build_process_mermaid(
    question: str,
    tasks: list[ResearchTask],
    sections: list[str] | None = None,
) -> str:
    """
    生成研究过程图：
    子任务 DAG → 搜索来源 → 报告章节
    """
    lines = ["flowchart TD", '  Q["研究问题"]']
    q_label = question.replace('"', "'")[:60]
    lines.append(f'  Q === QLabel["{q_label}"]')

    for task in tasks:
        tid = _safe_id(task.id)
        t_label = task.question.replace('"', "'")[:50]
        status = task.status.value
        lines.append(f'  {tid}["{task.id}: {t_label}<br/>{status}"]')
        lines.append(f"  Q --> {tid}")
        for dep in task.depends_on:
            lines.append(f"  {_safe_id(dep)} --> {tid}")

        for i, src in enumerate(task.sources[:3]):
            sid = f"{tid}_s{i}"
            title = str(src.get("title") or src.get("url") or "source").replace(
                '"', "'"
            )[:40]
            lines.append(f'  {sid}(["{title}"])')
            lines.append(f"  {tid} --> {sid}")

    if sections:
        lines.append('  R["最终报告"]')
        for i, section in enumerate(sections[:5]):
            sec_id = f"sec{i}"
            label = section.replace('"', "'").replace("\n", " ")[:40]
            lines.append(f'  {sec_id}["章节{i + 1}: {label}"]')
            lines.append(f"  R --> {sec_id}")
            for task in tasks:
                if task.sources:
                    lines.append(f"  {_safe_id(task.id)} --> R")
                    break

    return "\n".join(lines)
