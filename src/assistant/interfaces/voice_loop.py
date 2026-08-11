"""Voice interface - `assistant-voice` on the command line.

Loop: listen for an utterance -> transcribe -> send to orchestrator ->
speak the reply. Push-to-talk-free: silence-detection via VAD decides
when you've finished speaking (see audio/stt.py).

Requires the [voice] extra: pip install -e ".[voice]"
"""
from __future__ import annotations

import sys

from rich.console import Console

from assistant.config import get_settings
from assistant.core.orchestrator import Orchestrator
from assistant.llm.engine import get_engine
from assistant.logging_config import get_logger, setup_logging

log = get_logger(__name__)
console = Console()


def main() -> None:
    setup_logging()
    settings = get_settings()

    if not settings.audio.enabled:
        console.print(
            "[bold yellow]audio.enabled is false in config.[/bold yellow] "
            "Set it to true in config/config.yaml once whisper + piper are installed."
        )
        sys.exit(1)

    try:
        from assistant.audio.stt import MicListener, Transcriber
        from assistant.audio.tts import Speaker
    except ImportError as e:
        console.print(
            f"[bold red]Voice dependencies missing:[/bold red] {e}\n"
            f"Install with: pip install -e \".[voice]\""
        )
        sys.exit(1)

    engine = get_engine()
    if not engine.health_check():
        console.print(f"[bold red]Cannot reach local model server at {engine.base_url}.[/bold red]")
        sys.exit(1)

    orchestrator = Orchestrator(engine=engine)
    mic = MicListener()
    transcriber = Transcriber()
    try:
        speaker = Speaker()
    except FileNotFoundError as e:
        console.print(f"[bold red]{e}[/bold red]")
        sys.exit(1)

    console.print(f"[bold cyan]{settings.assistant.name}[/bold cyan] voice mode ready. Ctrl+C to quit.\n")

    while True:
        try:
            console.print("[dim]listening...[/dim]")
            audio = mic.listen()
            user_text = transcriber.transcribe(audio)
            if not user_text:
                continue
            console.print(f"[bold green]you[/bold green] > {user_text}")

            result = orchestrator.handle_turn(user_text)
            console.print(f"[bold cyan]{settings.assistant.name}[/bold cyan] > {result.text}")
            speaker.say(result.text)
        except KeyboardInterrupt:
            console.print("\n[dim]goodbye[/dim]")
            break
        except Exception:  # noqa: BLE001 - never let one bad turn kill the loop
            log.exception("Error in voice loop turn")
            console.print("[bold red]Something went wrong that turn - still listening.[/bold red]")


if __name__ == "__main__":
    main()
