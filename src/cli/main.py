"""命令行入口。
包括：
- 单 Agent 研究模式：使用一个 Agent 完成研究任务
- 多 Agent 编排模式：使用多个 Agent 协同完成研究任务
- 评估实验模式：对不同研究模式进行对比评估
- 交互模式：与 Agent 进行交互，输入问题，获取研究结果
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Path(__file__) 获取当前 cli/main.py 文件绝对路径
# resolve() 将路径解析为绝对路径
# parents[2] 获取当前文件的父目录的父目录，即项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  

# 如果项目根目录不在 sys.path 中，则添加到 sys.path 中
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# rich 系列：终端美化 UI 组件（控制台、面板、Markdown、表格、加载状态）
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

# 评估模块：实验配置、评估指标、雷达图
from src.evaluation.harness import EvaluationHarness, ExperimentConfig
from src.evaluation.metrics import evaluate_report, scores_to_radar_text
from src.factory import create_orchestrator, create_research_agent


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(
        description="多 Agent 深度研究系统",
    )
    parser.add_argument(
        "question",
        nargs="?",  # 可选位置参数，不传时进入交互模式，传值直接单次执行研究
        help="研究问题（交互模式下可省略）",
    )
    parser.add_argument(
        "-i",  # 交互模式选项
        "--interactive",
        action="store_true",
        help="交互模式，可连续提问",
    )
    parser.add_argument(
        "--mode",
        choices=["multi", "single"],
        default="multi",
        help="研究模式：multi=完整多 Agent（默认），single=单 Agent",
    )
    parser.add_argument(
        "--session",
        default="default",
        help="会话 ID，用于 Qdrant / 图存储分组",
    )
    parser.add_argument(
        "--max-verify-rounds",
        type=int,  # 自动将入参转为整数
        default=None,
        help="自反思最大轮次（默认读取 .env / 配置）",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="对问题跑对比评估实验",
    )
    parser.add_argument(
        "--eval-quick",
        action="store_true",
        help="快速评估（仅 multi vs single）",
    )
    parser.add_argument(
        "--no-kg",
        action="store_true",
        help="多 Agent 模式下禁用知识图谱抽取",
    )
    return parser


def render_report(console: Console, report) -> None:
    """用 Rich 渲染研究报告。"""
    console.print()
    console.print(Panel(f"[bold]{report.question}[/bold]", title="研究问题", border_style="blue"))
    console.print(Panel(report.summary, title="核心摘要", border_style="green"))

    if report.sections:
        sections_md = "\n\n".join(
            f"### 章节 {i + 1}\n{s}" for i, s in enumerate(report.sections)
        )
        console.print(
            Panel(Markdown(sections_md), title="分节论述", border_style="magenta")
        )

    if report.key_findings:
        findings_md = "\n".join(f"- {f}" for f in report.key_findings)
        console.print(
            Panel(Markdown(findings_md), title="关键发现", border_style="yellow")
        )

    if report.unresolved_questions:
        unresolved_md = "\n".join(f"- {q}" for q in report.unresolved_questions)
        console.print(
            Panel(Markdown(unresolved_md), title="未解决问题", border_style="red")
        )

    if report.citations:
        table = Table(title="引用来源", show_header=True)
        table.add_column("编号", style="cyan", width=6)
        table.add_column("标题", style="white")
        table.add_column("URL", style="dim")
        for cite in report.citations:
            table.add_row(str(cite.index), cite.title, cite.url)
        console.print(table)

    if report.process_diagram:
        console.print(
            Panel(
                report.process_diagram,
                title="研究过程图 (Mermaid)",
                border_style="cyan",
            )
        )
    if report.kg_diagram:
        console.print(
            Panel(
                report.kg_diagram,
                title="知识图谱快照 (Mermaid)",
                border_style="cyan",
            )
        )


def render_tool_history(console: Console, agent) -> None:
    """渲染工具调用历史（单 Agent）。"""
    history = getattr(agent, "tool_call_history", None) # 安全获取对象属性，属性不存在返回默认值
    if not history:
        return

    table = Table(title="工具调用记录", show_header=True)
    table.add_column("工具", style="cyan")
    table.add_column("参数", style="white")
    table.add_column("状态", style="green")

    for record in history:
        status = (
            "[green]成功[/green]"
            if record.result.success
            else f"[red]失败: {record.result.error}[/red]"
        )
        table.add_row(record.tool_name, str(record.arguments), status)
    console.print(table)


def run_single(console: Console, question: str, session_id: str, agent=None) -> None:
    """单 Agent 研究。"""
    console.print(f"\n[bold blue]正在研究 (single):[/bold blue] {question}\n")
    if agent is None:
        agent = create_research_agent()

    # 自动执行进入 / 退出逻辑：进入显示加载动画，代码块结束自动清除，无需手动管理。
    with console.status("[bold green]搜索与分析中..."):
        report = agent.research(question, session_id=session_id)

    render_report(console, report)
    render_tool_history(console, agent)
    scores = evaluate_report(report)
    console.print(Panel(scores_to_radar_text(scores), title="即时评估", border_style="white"))
    console.print(
        f"\n[dim]模式=single，证据 {len(agent.collected_evidence)} 条，"
        f"元数据={report.metadata}[/dim]"
    )


def run_multi(
    console: Console,
    question: str,
    session_id: str,
    max_verify_rounds: int | None,
    enable_kg: bool,
    orchestrator=None,
) -> None:
    """多 Agent 研究。"""
    console.print(f"\n[bold blue]正在研究 (multi):[/bold blue] {question}\n")
    if orchestrator is None:
        orchestrator = create_orchestrator(
            max_verify_rounds=max_verify_rounds, # 自反思最大轮次
            enable_kg=enable_kg, # 是否启用知识图谱抽取
        )
    # 内部调度 Planner/Researcher/Synthesizer/Verifier 多智能体协作
    with console.status("[bold green]多 Agent 编排执行中..."):
        report = orchestrator.research(question, session_id=session_id)

    render_report(console, report)
    scores = evaluate_report(report, tasks=orchestrator.last_tasks) # 评估时额外传入编排任务记录orchestrator.last_tasks，多维度打分
    console.print(Panel(scores_to_radar_text(scores), title="即时评估", border_style="white"))

    if orchestrator.events:
        console.print(
            Panel(
                "\n".join(f"- {e}" for e in orchestrator.events),
                title="编排事件",  # 打印多 Agent 分工、调度、反思全流程事件日志
                border_style="dim",
            )
        )
    console.print(
        f"\n[dim]模式=multi，验证轮次={report.verify_rounds}，"  # 打印验证轮次、元数据
        f"元数据={report.metadata}[/dim]"
    )

def run_eval(console: Console, question: str, quick: bool = False) -> None:
    """对比评估。"""
    console.print(f"\n[bold blue]评估实验:[/bold blue] {question}\n")

    def runner(cfg: ExperimentConfig, q: str):
        if cfg.mode == "single":
            agent = create_research_agent()
            report = agent.research(q, session_id=f"eval_{cfg.name}")
            return report, []
        orch = create_orchestrator(
            max_verify_rounds=cfg.max_verify_rounds,
            enable_kg=cfg.enable_kg,
        )
        report = orch.research(q, session_id=f"eval_{cfg.name}")
        return report, orch.last_tasks

    configs = None
    if quick:
        configs = [
            ExperimentConfig(
                name="multi+verify", mode="multi", max_verify_rounds=1, enable_kg=True # 快速评估：多 Agent 模式，验证一轮，启用知识图谱抽取
            ),
            ExperimentConfig(name="single", mode="single"), # 快速评估：单 Agent 模式
        ]

    harness = EvaluationHarness(runner) # 创建评估器，传入 runner 函数
    with console.status("[bold green]对比实验运行中..."):
        output = harness.run(question, configs=configs) # 运行评估，传入问题和实验配置

    console.print(Markdown(output["markdown_table"])) # 打印评估结果表格
    for name, radar in output["radars"].items(): # 循环打印每个评估指标的雷达图
        console.print(Panel(radar, title=name, border_style="white")) # 打印评估结果雷达图


def main() -> None:
    """CLI 主入口。"""
    parser = build_parser()
    args = parser.parse_args()
    console = Console()

    console.print(
        Panel(
            "[bold]多 Agent 深度研究系统[/bold]\n"
            "Orchestrator · Planner · Researcher · Synthesizer · Verifier\n"
            f"当前模式: {args.mode}",
            border_style="bright_blue",
        )
    )

    try:
        if args.eval or args.eval_quick:
            if not args.question:
                parser.error("--eval 需要提供研究问题")
            run_eval(console, args.question, quick=args.eval_quick)
            return

        if args.interactive:
            console.print("[dim]交互模式，输入 quit 退出。[/dim]\n")
            agent = None
            orchestrator = None
            if args.mode == "single":
                agent = create_research_agent()
            else:
                orchestrator = create_orchestrator(
                    max_verify_rounds=args.max_verify_rounds,
                    enable_kg=not args.no_kg,
                )

            while True:
                try:
                    question = console.input("[bold cyan]研究问题> [/bold cyan]").strip()
                except (EOFError, KeyboardInterrupt):
                    console.print("\n再见！")
                    break

                if not question:
                    continue
                if question.lower() in ("quit", "exit", "q"):
                    console.print("再见！")
                    break

                if args.mode == "single":
                    run_single(console, question, args.session, agent=agent)
                else:
                    run_multi(
                        console,
                        question,
                        f"{args.session}_{abs(hash(question)) % 100000}",
                        args.max_verify_rounds,
                        enable_kg=not args.no_kg,
                        orchestrator=orchestrator,
                    )
                console.print()

        elif args.question:
            if args.mode == "single":
                run_single(console, args.question, args.session)
            else:
                run_multi(
                    console,
                    args.question,
                    args.session,
                    args.max_verify_rounds,
                    enable_kg=not args.no_kg,
                )
        else:
            parser.print_help()
            sys.exit(1)

    except EnvironmentError as exc:
        console.print(f"[red]配置错误:[/red] {exc}")
        sys.exit(1)
    except Exception as exc:
        console.print(f"[red]运行失败:[/red] {exc}")
        raise


if __name__ == "__main__":
    main()
