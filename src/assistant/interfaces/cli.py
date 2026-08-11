"""Text chat CLI - `assistant` on the command line.

This is the primary interface to build and debug against before layering
voice on top. Run `ollama serve` and pull a model first (see README).
"""
from __future__ import annotations

import sys

from rich.console import Console
from rich.markdown import Markdown

from assistant.config import get_settings
from assistant.core.orchestrator import Orchestrator
from assistant.llm.engine import get_engine
from assistant.logging_config import get_logger, setup_logging

log = get_logger(__name__)
console = Console()


def main() -> None:
    setup_logging()
    settings = get_settings()
    engine = get_engine()

    if not engine.health_check():
        console.print(
            f"[bold red]Cannot reach local model server at {engine.base_url}.[/bold red]\n"
            f"Start it with: [bold]ollama serve[/bold]  (and make sure "
            f"`ollama pull {settings.llm.model}` has been run once)."
        )
        sys.exit(1)

    orchestrator = Orchestrator(engine=engine)
    console.print(
        f"[bold cyan]{settings.assistant.name}[/bold cyan] is ready "
        f"(fully offline, model: {settings.llm.model}). Type 'exit' to quit.\n"
    )

    while True:
        try:
            user_text = console.input("[bold green]you[/bold green] > ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]goodbye[/dim]")
            break

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            console.print("[dim]goodbye[/dim]")
            break

        with console.status("[dim]thinking...[/dim]", spinner="dots"):
            result = orchestrator.handle_turn(user_text)

        if result.tool_calls_made:
            console.print(f"[dim]used tools: {', '.join(result.tool_calls_made)}[/dim]")
        console.print(f"[bold cyan]{settings.assistant.name}[/bold cyan] > ", end="")
        console.print(Markdown(result.text))


if __name__ == "__main__":
    main()
