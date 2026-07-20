"""知识图谱导出为 Mermaid / HTML。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.models.schemas import EntityRelation


def _safe_node(name: str) -> str:
    node = "".join(ch if ch.isalnum() else "_" for ch in name)
    return (node or "entity")[:48]


def relations_to_mermaid(relations: Iterable[EntityRelation]) -> str:
    """将三元组转为 Mermaid flowchart。"""
    lines = ["flowchart LR"]
    seen_nodes: set[str] = set()
    count = 0
    for rel in relations:
        if count >= 40:
            break
        s_id = _safe_node(rel.subject)
        o_id = _safe_node(rel.object)
        if s_id not in seen_nodes:
            lines.append(f'  {s_id}["{rel.subject[:40]}"]')
            seen_nodes.add(s_id)
        if o_id not in seen_nodes:
            lines.append(f'  {o_id}["{rel.object[:40]}"]')
            seen_nodes.add(o_id)
        pred = rel.predicate.replace('"', "'")[:20]
        lines.append(f"  {s_id} -- {pred} --> {o_id}")
        count += 1

    if len(lines) == 1:
        lines.append('  empty["暂无实体关系"]')
    return "\n".join(lines)


def export_kg_html(
    relations: list[EntityRelation],
    output_path: str | Path,
    title: str = "Knowledge Graph Snapshot",
) -> Path:
    """导出简易可浏览 HTML（内嵌 Mermaid）。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mermaid = relations_to_mermaid(relations)
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <title>{title}</title>
  <script type="module">
    import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
    mermaid.initialize({{ startOnLoad: true, theme: "neutral" }});
  </script>
  <style>
    body {{ font-family: "Segoe UI", sans-serif; margin: 2rem; background: #f7f7f5; color: #222; }}
    h1 {{ font-size: 1.4rem; }}
    .diagram {{ background: #fff; padding: 1.5rem; border: 1px solid #ddd; }}
    pre {{ white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>共 {len(relations)} 条关系三元组</p>
  <div class="diagram">
    <pre class="mermaid">
{mermaid}
    </pre>
  </div>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    return path
