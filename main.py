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

from core.nlp_analyzer import ContextAnalysis, OllamaAnalyzer, OllamaUnavailable
from core.sound_classifier import (
    SoundAnalysisError,
    SoundAnalysisResult,
    SoundClassifierUnavailable,
    SoundEvent,
    YAMNetSoundClassifier,
    filter_non_speech_events,
    format_sound_events,
    format_non_speech_notes,
)
from core.transcriber import (
    AudioValidationError,
    FasterWhisperTranscriber,
    LanguageCode,
    MicrophoneError,
    TranscriptionResult,
    TranscriptChunk,
    format_timestamp,
)
from utils.file_manager import save_text_report, save_transcript


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


def build_sound_table(events: list[SoundEvent]) -> Table:
    table = Table(title="Non-Speech Sound Notes", expand=True)
    table.add_column("Time", style="cyan", no_wrap=True)
    table.add_column("Label", style="white")
    table.add_column("Category", style="magenta")
    table.add_column("Score", style="green", justify="right")
    for event in events:
        time_range = f"{format_timestamp(event.start)} -> {format_timestamp(event.end)}"
        table.add_row(time_range, event.label, event.category, f"{event.score:.2f}")
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
    language: LanguageCode = typer.Option(
        "auto",
        help="Transcription language: auto, en, vi, or zh.",
    ),
    with_sounds: bool = typer.Option(False, help="Run YAMNet sound event analysis after transcription."),
    sound_threshold: float = typer.Option(0.12, help="Minimum YAMNet confidence score."),
    sound_top_k: int = typer.Option(10, help="Maximum YAMNet classes to keep per frame."),
    context_analysis: bool = typer.Option(
        False,
        help="Run hybrid pitch/energy and Ollama Llama-3 contextual analysis every block.",
    ),
    context_window_seconds: float = typer.Option(60.0, help="Seconds of transcript context per Llama-3 analysis block."),
    ollama_model: str = typer.Option("llama3", help="Local Ollama model used for context analysis."),
) -> None:
    """Transcribe English, Vietnamese, or Chinese audio and save the result."""
    transcriber = FasterWhisperTranscriber()
    chunks: list[TranscriptChunk] = []
    progress = build_progress()

    def on_progress(current_seconds: float) -> None:
        progress.update(task_id, completed=current_seconds)

    try:
        audio_seconds = transcriber.validate_audio_file(file_path)[1]
        task_id = progress.add_task("Transcribing", total=audio_seconds)
        ollama_analyzer = OllamaAnalyzer(model=ollama_model) if context_analysis else None
        sound_result: SoundAnalysisResult | None = None
        non_speech_events: list[SoundEvent] = []
        if with_sounds or context_analysis:
            sound_result = _analyze_sounds_for_transcription(
                file_path=file_path,
                threshold=sound_threshold,
                top_k=sound_top_k,
                required=with_sounds,
            )
            if sound_result is not None:
                non_speech_events = filter_non_speech_events(sound_result.events)

        with Live(
            Group(progress, build_transcript_table(chunks)),
            console=console,
            refresh_per_second=4,
        ) as live:
            result = transcriber.transcribe_file(
                file_path=file_path,
                language=language,
                on_progress=on_progress,
                on_chunk=lambda chunk: _append_and_refresh(chunk, chunks, progress, live),
                sound_events=non_speech_events,
                context_analysis_enabled=context_analysis,
                context_window_seconds=context_window_seconds,
                context_analysis_runner=(
                    lambda buffer: _run_context_analysis(buffer, ollama_analyzer, live, chunks, progress)
                    if ollama_analyzer is not None
                    else None
                ),
            )

        transcript_text = _build_transcript_output(result)
        if sound_result is not None:
            sound_text = format_non_speech_notes(non_speech_events)
            transcript_text = f"{transcript_text}\n\n[Non-speech sound notes]\n{sound_text}".strip()

        output_path = save_transcript(transcript_text, file_path)
        _print_summary(result, output_path)
        if sound_result is not None:
            console.print(build_sound_table(non_speech_events))
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
    language: LanguageCode = typer.Option(
        "auto",
        help="Transcription language: auto, en, vi, or zh.",
    ),
    silence_seconds: float = typer.Option(1.0, help="Seconds of silence before transcription starts."),
    max_record_seconds: float = typer.Option(60.0, help="Maximum recording length."),
    vad_threshold: int = typer.Option(500, help="Microphone VAD RMS threshold."),
) -> None:
    """Capture microphone speech, transcribe after silence, and print the result."""
    transcriber = FasterWhisperTranscriber()
    console.print("[bold cyan]Listening...[/bold cyan] Speak now. Pause to transcribe.")

    try:
        result = transcriber.listen_once(
            language=language,
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
        _print_language_hint(result)
    except MicrophoneError as exc:
        console.print(f"[bold red]Microphone error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except AudioValidationError as exc:
        console.print(f"[bold red]Audio error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        console.print("\n[yellow]Listening cancelled.[/yellow]")
        raise typer.Exit(code=130)


@app.command("analyze-sound")
def analyze_sound(
    file_path: Path = typer.Argument(..., help="Path to an audio file."),
    threshold: float = typer.Option(0.12, help="Minimum YAMNet confidence score."),
    top_k: int = typer.Option(10, help="Maximum classes to keep per YAMNet frame."),
    include_voice: bool = typer.Option(False, help="Include speech/conversation events in the report."),
) -> None:
    """Detect non-speech sound events with YAMNet and save a text report."""
    try:
        classifier = YAMNetSoundClassifier(threshold=threshold, top_k=top_k)
        with console.status("[bold blue]Running YAMNet sound analysis...[/bold blue]"):
            result = classifier.analyze_file(file_path)
        report_events = result.events if include_voice else filter_non_speech_events(result.events)
        report = format_sound_events(report_events)
        output_path = save_text_report(report, file_path, suffix="_sounds")
        console.print(build_sound_table(report_events))
        console.print(
            Panel.fit(
                f"Saved: [bold]{output_path}[/bold]\n"
                f"Audio: {result.duration_seconds:.2f}s\n"
                f"Processing: {result.processing_seconds:.2f}s",
                title="Sound Analysis Done",
                border_style="green",
            )
        )
    except FileNotFoundError as exc:
        console.print(f"[bold red]Missing file:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except SoundClassifierUnavailable as exc:
        console.print(f"[bold red]YAMNet unavailable:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except SoundAnalysisError as exc:
        console.print(f"[bold red]Sound analysis error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


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
            f"Processing: {result.duration_seconds:.2f}s\n"
            f"Language: {_format_language(result)}",
            title="Done",
            border_style="green",
        )
    )


def _print_language_hint(result: TranscriptionResult) -> None:
    console.print(f"[dim]Language: {_format_language(result)}[/dim]")


def _format_language(result: TranscriptionResult) -> str:
    if result.detected_language is None:
        return "unknown"
    if result.language_probability is None:
        return result.detected_language
    return f"{result.detected_language} ({result.language_probability:.0%})"


def _build_transcript_output(result: TranscriptionResult) -> str:
    output = f"[Detected language: {_format_language(result)}]\n\n{result.text}".strip()
    if result.context_analyses:
        analyses = "\n\n".join(
            f"[Block {index}] {analysis.text}" for index, analysis in enumerate(result.context_analyses, start=1)
        )
        output = f"{output}\n\n[Context analyses]\n{analyses}".strip()
    return output


def _run_context_analysis(
    conversation_buffer: list[dict[str, object]],
    analyzer: OllamaAnalyzer | None,
    live: Live,
    chunks: list[TranscriptChunk],
    progress: Progress,
) -> ContextAnalysis | None:
    if analyzer is None:
        return None

    live.stop()
    try:
        with console.status("[bold yellow]⚡ Đang phân tích ngữ cảnh đoạn...[/]"):
            analysis = analyzer.analyze_context(conversation_buffer)
        console.print(
            Panel(
                analysis.text,
                title="📊 TỔNG KẾT NGỮ CẢNH ĐOẠN",
                border_style="green",
            )
        )
        return analysis
    except OllamaUnavailable as exc:
        console.print(f"[yellow]Bỏ qua phân tích ngữ cảnh:[/yellow] {exc}")
        return None
    finally:
        live.start(refresh=True)
        live.update(Group(progress, build_transcript_table(chunks)))


def _analyze_sounds_or_exit(file_path: Path, threshold: float) -> SoundAnalysisResult:
    try:
        classifier = YAMNetSoundClassifier(threshold=threshold)
        with console.status("[bold blue]Running YAMNet sound analysis...[/bold blue]"):
            return classifier.analyze_file(file_path)
    except SoundClassifierUnavailable as exc:
        console.print(f"[bold red]YAMNet unavailable:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except SoundAnalysisError as exc:
        console.print(f"[bold red]Sound analysis error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


def _analyze_sounds_for_transcription(
    file_path: Path,
    threshold: float,
    top_k: int,
    required: bool,
) -> SoundAnalysisResult | None:
    try:
        classifier = YAMNetSoundClassifier(threshold=threshold, top_k=top_k)
        with console.status("[bold blue]Running raw-audio YAMNet timeline...[/bold blue]"):
            return classifier.analyze_file(file_path)
    except (SoundClassifierUnavailable, SoundAnalysisError) as exc:
        if required:
            console.print(f"[bold red]YAMNet error:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc
        console.print(f"[yellow]YAMNet timeline skipped:[/yellow] {exc}")
        return None


if __name__ == "__main__":
    app()
