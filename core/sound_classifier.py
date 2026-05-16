from __future__ import annotations

import csv
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from pydub import AudioSegment


YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"
YAMNET_SAMPLE_RATE = 16_000
YAMNET_FRAME_SECONDS = 0.48

EMOTION_CUE_KEYWORDS = (
    "laughter",
    "laugh",
    "giggle",
    "chuckle",
    "crying",
    "sob",
    "whimper",
    "sigh",
    "groan",
    "moan",
    "screaming",
    "shout",
    "yell",
)

VOICE_KEYWORDS = ("speech", "conversation", "narration", "monologue", "dialogue")
HUMAN_NON_SPEECH_KEYWORDS = (
    "cough",
    "sneeze",
    "sniff",
    "breathing",
    "throat clearing",
    "hiccup",
    "snoring",
    "chewing",
    "clapping",
    "applause",
    "finger snapping",
)
AMBIENT_KEYWORDS = (
    "silence",
    "inside",
    "outside",
    "noise",
    "static",
    "wind",
    "rain",
    "water",
    "traffic",
    "vehicle",
    "engine",
    "keyboard",
    "typing",
    "door",
    "knock",
    "alarm",
    "siren",
)


@dataclass(frozen=True)
class SoundEvent:
    start: float
    end: float
    label: str
    score: float
    category: str


@dataclass(frozen=True)
class SoundAnalysisResult:
    events: list[SoundEvent]
    duration_seconds: float
    processing_seconds: float


class SoundClassifierUnavailable(RuntimeError):
    """Raised when optional YAMNet dependencies are not installed."""


class SoundAnalysisError(RuntimeError):
    """Raised when sound analysis cannot read or process the audio file."""


class YAMNetSoundClassifier:
    def __init__(
        self,
        model_url: str = YAMNET_MODEL_URL,
        top_k: int = 5,
        threshold: float = 0.25,
    ) -> None:
        self.model_url = model_url
        self.top_k = top_k
        self.threshold = threshold
        self._model: object | None = None
        self._class_names: list[str] | None = None

    @property
    def model(self) -> object:
        if self._model is None:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            try:
                import tensorflow_hub as hub
            except ImportError as exc:
                raise SoundClassifierUnavailable(
                    _build_yamnet_dependency_message()
                ) from exc
            self._model = hub.load(self.model_url)
        return self._model

    @property
    def class_names(self) -> list[str]:
        if self._class_names is None:
            class_map_path = self.model.class_map_path().numpy()
            self._class_names = _load_class_names(class_map_path)
        return self._class_names

    def analyze_file(self, file_path: Path) -> SoundAnalysisResult:
        started_at = time.perf_counter()
        model = self.model
        waveform, duration_seconds = _load_waveform(file_path)
        scores, _embeddings, _spectrogram = model(waveform)
        events = self._scores_to_events(scores.numpy())
        elapsed = time.perf_counter() - started_at
        return SoundAnalysisResult(
            events=events,
            duration_seconds=duration_seconds,
            processing_seconds=elapsed,
        )

    def _scores_to_events(self, scores: object) -> list[SoundEvent]:
        import numpy as np

        frame_events: list[SoundEvent] = []
        for frame_index, frame_scores in enumerate(scores):
            ranked_indices = np.argsort(frame_scores)[::-1][: self.top_k]
            start = frame_index * YAMNET_FRAME_SECONDS
            end = start + YAMNET_FRAME_SECONDS
            for class_index in ranked_indices:
                score = float(frame_scores[class_index])
                if score < self.threshold:
                    continue
                label = self.class_names[int(class_index)]
                frame_events.append(
                    SoundEvent(
                        start=start,
                        end=end,
                        label=label,
                        score=score,
                        category=categorize_sound(label),
                    )
                )

        return merge_adjacent_events(frame_events)


def categorize_sound(label: str) -> str:
    normalized = label.lower()
    if any(keyword in normalized for keyword in EMOTION_CUE_KEYWORDS):
        return "emotion-cue"
    if any(keyword in normalized for keyword in VOICE_KEYWORDS):
        return "voice"
    if any(keyword in normalized for keyword in HUMAN_NON_SPEECH_KEYWORDS):
        return "human-non-speech"
    if any(keyword in normalized for keyword in ("music", "singing", "song")):
        return "music"
    if any(keyword in normalized for keyword in AMBIENT_KEYWORDS):
        return "ambient"
    return "other"


def is_non_speech_event(event: SoundEvent) -> bool:
    return event.category != "voice"


def filter_non_speech_events(events: Iterable[SoundEvent]) -> list[SoundEvent]:
    return [event for event in events if is_non_speech_event(event)]


def events_overlapping_window(
    events: Iterable[SoundEvent],
    start: float,
    end: float,
    min_overlap_seconds: float = 0.05,
) -> list[SoundEvent]:
    """Return events that overlap a transcript chunk by at least a small amount."""
    overlapping: list[SoundEvent] = []
    for event in events:
        overlap = min(end, event.end) - max(start, event.start)
        if overlap >= min_overlap_seconds:
            overlapping.append(event)
    return overlapping


def merge_adjacent_events(events: Sequence[SoundEvent], max_gap_seconds: float = 0.55) -> list[SoundEvent]:
    if not events:
        return []

    sorted_events = sorted(events, key=lambda event: (event.label, event.start))
    merged: list[SoundEvent] = []
    current = sorted_events[0]

    for event in sorted_events[1:]:
        is_same_sound = event.label == current.label and event.category == current.category
        is_adjacent = event.start - current.end <= max_gap_seconds
        if is_same_sound and is_adjacent:
            current = SoundEvent(
                start=current.start,
                end=max(current.end, event.end),
                label=current.label,
                score=max(current.score, event.score),
                category=current.category,
            )
            continue
        merged.append(current)
        current = event

    merged.append(current)
    return sorted(merged, key=lambda event: event.start)


def format_sound_events(events: Iterable[SoundEvent]) -> str:
    lines = []
    for event in events:
        lines.append(
            f"[{_format_timestamp(event.start)} -> {_format_timestamp(event.end)}] "
            f"{event.label} ({event.category}, {event.score:.2f})"
        )
    return "\n".join(lines)


def format_non_speech_notes(events: Iterable[SoundEvent]) -> str:
    non_speech_events = filter_non_speech_events(events)
    if not non_speech_events:
        return "No non-speech sound events detected."
    return format_sound_events(non_speech_events)


def _load_waveform(file_path: Path) -> tuple[object, float]:
    import numpy as np

    audio_path = file_path.expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception as exc:
        raise SoundAnalysisError(
            "Could not read audio for YAMNet analysis. Install FFmpeg and ensure the file is valid."
        ) from exc
    duration_seconds = len(audio) / 1000
    audio = audio.set_channels(1).set_frame_rate(YAMNET_SAMPLE_RATE).set_sample_width(2)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        audio.export(temp_path, format="wav")
        converted = AudioSegment.from_wav(temp_path)
        samples = np.array(converted.get_array_of_samples(), dtype=np.float32)
        waveform = samples / 32768.0
        return waveform, duration_seconds
    finally:
        temp_path.unlink(missing_ok=True)


def _load_class_names(class_map_path: bytes | str) -> list[str]:
    path = class_map_path.decode("utf-8") if isinstance(class_map_path, bytes) else class_map_path
    with open(path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return [row["display_name"] for row in reader]


def _format_timestamp(seconds: float) -> str:
    whole_seconds = int(seconds)
    minutes, sec = divmod(whole_seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minute:02d}:{sec:02d}"
    return f"{minute:02d}:{sec:02d}"


def _build_yamnet_dependency_message() -> str:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 14):
        return (
            f"YAMNet needs TensorFlow/TensorFlow Hub, but this venv is Python {version}. "
            "TensorFlow Windows wheels currently target Python 3.10-3.13. "
            "Create a Python 3.13 venv, then run: pip install -r requirements.txt; "
            "pip install -r requirements-yamnet.txt"
        )
    return (
        "YAMNet needs optional dependencies. Install them with: "
        "pip install -r requirements-yamnet.txt"
    )
