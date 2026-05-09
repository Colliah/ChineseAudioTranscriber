from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from core.transcriber import (
    AudioValidationError,
    FasterWhisperTranscriber,
    MicrophoneError,
    TranscriptionResult,
    TranscriptChunk,
    format_timestamp,
)
from utils.file_manager import save_transcript


app = typer.Typer(
    help="Local terminal speech-to-text powered by faster-whisper large-v3 on CPU INT8.",
    no_args_is_help=True,
)
console = Console()


def build_transcript_table(chunks: list[TranscriptChunk]) -> Table:
    table = Table(title="Live Transcription", expand=True)
    table.add_column("Time", style="cyan", no_wrap=True)
    table.add_column("Transcription", style="white")
    for chunk in chunks[-12:]:
        time_range = f"{format_timestamp(chunk.start)} -> {format_timestamp(chunk.end)}"
        table.add_row(time_range, chunk.text)
    return table


def build_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )


@app.command()
def transcribe(
    file_path: Path = typer.Argument(..., help="Path to an mp3, wav, or m4a file under 15 minutes."),
) -> None:
    """Transcribe an audio file and save the result under outputs/."""
    transcriber = FasterWhisperTranscriber()
    chunks: list[TranscriptChunk] = []
    progress = build_progress()

    def on_progress(current_seconds: float) -> None:
        progress.update(task_id, completed=current_seconds)

    try:
        audio_seconds = transcriber.validate_audio_file(file_path)[1]
        task_id = progress.add_task("Transcribing", total=audio_seconds)
        with Live(
            Group(progress, build_transcript_table(chunks)),
            console=console,
            refresh_per_second=4,
        ) as live:
            result = transcriber.transcribe_file(
                file_path=file_path,
                on_progress=on_progress,
                on_chunk=lambda chunk: _append_and_refresh(chunk, chunks, progress, live),
            )

        output_path = save_transcript(result.text, file_path)
        _print_summary(result, output_path)
    except FileNotFoundError as exc:
        console.print(f"[bold red]Missing file:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except AudioValidationError as exc:
        console.print(f"[bold red]Audio error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        console.print("\n[yellow]Transcription cancelled.[/yellow]")
        raise typer.Exit(code=130)


@app.command()
def listen(
    silence_seconds: float = typer.Option(1.0, help="Seconds of silence before transcription starts."),
    max_record_seconds: float = typer.Option(60.0, help="Maximum recording length."),
    vad_threshold: int = typer.Option(500, help="Microphone VAD RMS threshold."),
) -> None:
    """Capture microphone speech, transcribe after silence, and print the result."""
    transcriber = FasterWhisperTranscriber()
    console.print("[bold cyan]Listening...[/bold cyan] Speak now. Pause to transcribe.")

    try:
        result = transcriber.listen_once(
            silence_seconds=silence_seconds,
            max_record_seconds=max_record_seconds,
            vad_threshold=vad_threshold,
        )
        console.print(
            Panel.fit(
                result.text or "No transcription text was produced.",
                title="Transcription",
                border_style="bright_green",
            )
        )
        console.print(f"[dim]Processed in {result.duration_seconds:.2f}s[/dim]")
    except MicrophoneError as exc:
        console.print(f"[bold red]Microphone error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except AudioValidationError as exc:
        console.print(f"[bold red]Audio error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        console.print("\n[yellow]Listening cancelled.[/yellow]")
        raise typer.Exit(code=130)


def _append_and_refresh(
    chunk: TranscriptChunk,
    chunks: list[TranscriptChunk],
    progress: Progress,
    live: Live,
) -> None:
    chunks.append(chunk)
    live.update(Group(progress, build_transcript_table(chunks)))


def _print_summary(result: TranscriptionResult, output_path: Path) -> None:
    console.print(
        Panel.fit(
            f"Saved: [bold]{output_path}[/bold]\n"
            f"Audio: {result.audio_seconds:.2f}s\n"
            f"Processing: {result.duration_seconds:.2f}s",
            title="Done",
            border_style="green",
        )
    )


if __name__ == "__main__":
    app()
