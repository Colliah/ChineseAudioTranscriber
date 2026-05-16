from __future__ import annotations

import math
import re
import tempfile
import time
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Protocol, Sequence

from faster_whisper import WhisperModel
from pydub import AudioSegment

from core.nlp_analyzer import ContextAnalysis
from core.sound_classifier import SoundEvent, events_overlapping_window
from utils.audio_helper import AudioAnalyzer


INITIAL_PROMPT = (
    "Transcribe clearly with accurate punctuation. The audio may contain English, "
    "Vietnamese, or Chinese. Convert all Chinese text to Simplified Chinese. "
    "Preserve proper nouns when possible."
)

SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a"}
MAX_AUDIO_SECONDS = 15 * 60
LanguageCode = Literal["auto", "en", "vi", "zh"]


@dataclass(frozen=True)
class TranscriptChunk:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptionResult:
    chunks: list[TranscriptChunk]
    text: str
    duration_seconds: float
    audio_seconds: float
    detected_language: str | None
    language_probability: float | None
    context_analyses: list[ContextAnalysis]


@dataclass(frozen=True)
class WordInfo:
    start: float
    end: float
    word: str


ProgressCallback = Callable[[float], None]
ChunkCallback = Callable[[TranscriptChunk], None]
ContextAnalysisCallback = Callable[[ContextAnalysis], None]
ContextAnalysisRunner = Callable[[list[dict[str, Any]]], ContextAnalysis | None]


class AudioInputBackend(Protocol):
    def read(self, chunk_size: int) -> bytes:
        ...

    def close(self) -> None:
        ...


class AudioValidationError(ValueError):
    """Raised when an input audio file cannot be processed."""


class MicrophoneError(RuntimeError):
    """Raised when microphone capture cannot be initialized or completed."""


class FasterWhisperTranscriber:
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cpu",
        compute_type: str = "int8",
        initial_prompt: str = INITIAL_PROMPT,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.initial_prompt = initial_prompt
        self._model: WhisperModel | None = None

    @property
    def model(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe_file(
        self,
        file_path: Path,
        language: LanguageCode = "auto",
        on_progress: ProgressCallback | None = None,
        on_chunk: ChunkCallback | None = None,
        sound_events: Sequence[SoundEvent] | None = None,
        context_analysis_enabled: bool = False,
        context_window_seconds: float = 60.0,
        context_analysis_runner: ContextAnalysisRunner | None = None,
        on_context_analysis: ContextAnalysisCallback | None = None,
    ) -> TranscriptionResult:
        audio_path, audio_seconds = self.validate_audio_file(file_path)
        started_at = time.perf_counter()

        segments, _info = self.model.transcribe(
            str(audio_path),
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 250,
            },
            word_timestamps=True,
            initial_prompt=self.initial_prompt,
            beam_size=5,
            language=None if language == "auto" else language,
        )

        chunks = self._consume_segments(
            segments=segments,
            audio_path=audio_path,
            audio_seconds=audio_seconds,
            on_progress=on_progress,
            on_chunk=on_chunk,
            sound_events=sound_events or [],
            context_analysis_enabled=context_analysis_enabled,
            context_window_seconds=context_window_seconds,
            context_analysis_runner=context_analysis_runner,
            on_context_analysis=on_context_analysis,
        )
        elapsed = time.perf_counter() - started_at
        return TranscriptionResult(
            chunks=chunks,
            text=self.format_chunks(chunks),
            duration_seconds=elapsed,
            audio_seconds=audio_seconds,
            detected_language=getattr(_info, "language", None),
            language_probability=getattr(_info, "language_probability", None),
            context_analyses=getattr(self, "_last_context_analyses", []),
        )

    def listen_once(
        self,
        language: LanguageCode = "auto",
        silence_seconds: float = 1.0,
        max_record_seconds: float = 60.0,
        sample_rate: int = 16_000,
        chunk_size: int = 1024,
        vad_threshold: int = 500,
    ) -> TranscriptionResult:
        wav_path = self._record_until_silence(
            silence_seconds=silence_seconds,
            max_record_seconds=max_record_seconds,
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            vad_threshold=vad_threshold,
        )
        try:
            return self.transcribe_file(wav_path, language=language)
        finally:
            wav_path.unlink(missing_ok=True)

    def validate_audio_file(self, file_path: Path) -> tuple[Path, float]:
        audio_path = file_path.expanduser().resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
            raise AudioValidationError(f"Unsupported audio type. Use one of: {supported}")

        try:
            audio = AudioSegment.from_file(audio_path)
        except Exception as exc:
            raise AudioValidationError(
                "Could not read audio. Ensure ffmpeg is installed and the file is valid."
            ) from exc

        audio_seconds = len(audio) / 1000
        if audio_seconds > MAX_AUDIO_SECONDS:
            raise AudioValidationError("Audio must be under 15 minutes.")
        return audio_path, audio_seconds

    def _consume_segments(
        self,
        segments: Iterable[object],
        audio_path: Path,
        audio_seconds: float,
        on_progress: ProgressCallback | None,
        on_chunk: ChunkCallback | None,
        sound_events: Sequence[SoundEvent],
        context_analysis_enabled: bool,
        context_window_seconds: float,
        context_analysis_runner: ContextAnalysisRunner | None,
        on_context_analysis: ContextAnalysisCallback | None,
    ) -> list[TranscriptChunk]:
        chunks: list[TranscriptChunk] = []
        pending_words: list[WordInfo] = []
        last_end = 0.0
        conversation_buffer: list[dict[str, Any]] = []
        buffered_duration = 0.0
        context_analyses: list[ContextAnalysis] = []
        audio_analyzer = AudioAnalyzer() if context_analysis_enabled else None
        self._last_context_analyses = context_analyses

        for segment in segments:
            words = self._extract_words(segment)
            if words:
                for word in words:
                    pending_words.append(word)
                    last_end = max(last_end, word.end)
                    if self._should_flush_chunk(pending_words):
                        chunk = self._words_to_chunk(pending_words, sound_events)
                        chunks.append(chunk)
                        if on_chunk is not None:
                            on_chunk(chunk)
                        buffered_duration = self._append_context_buffer(
                            conversation_buffer,
                            buffered_duration,
                            chunk,
                            audio_path,
                            audio_analyzer,
                            sound_events,
                        )
                        buffered_duration = self._maybe_run_context_analysis(
                            conversation_buffer,
                            buffered_duration,
                            context_window_seconds,
                            context_analysis_runner,
                            on_context_analysis,
                            context_analyses,
                        )
                        pending_words = []
            else:
                text = str(getattr(segment, "text", "")).strip()
                if text:
                    text = normalize_simplified_chinese(text)
                    start = float(getattr(segment, "start", last_end))
                    end = float(getattr(segment, "end", start))
                    chunk = TranscriptChunk(start=start, end=end, text=text)
                    chunks.append(chunk)
                    if on_chunk is not None:
                        on_chunk(chunk)
                    buffered_duration = self._append_context_buffer(
                        conversation_buffer,
                        buffered_duration,
                            chunk,
                            audio_path,
                            audio_analyzer,
                            sound_events,
                        )
                    buffered_duration = self._maybe_run_context_analysis(
                        conversation_buffer,
                        buffered_duration,
                        context_window_seconds,
                        context_analysis_runner,
                        on_context_analysis,
                        context_analyses,
                    )
                    last_end = max(last_end, end)

            if on_progress is not None:
                on_progress(min(last_end, audio_seconds))

        if pending_words:
            chunk = self._words_to_chunk(pending_words, sound_events)
            chunks.append(chunk)
            if on_chunk is not None:
                on_chunk(chunk)
            buffered_duration = self._append_context_buffer(
                conversation_buffer,
                buffered_duration,
                chunk,
                audio_path,
                audio_analyzer,
                sound_events,
            )
            self._maybe_run_context_analysis(
                conversation_buffer,
                buffered_duration,
                context_window_seconds,
                context_analysis_runner,
                on_context_analysis,
                context_analyses,
            )
            last_end = max(last_end, chunk.end)

        self._maybe_run_context_analysis(
            conversation_buffer,
            buffered_duration,
            context_window_seconds,
            context_analysis_runner,
            on_context_analysis,
            context_analyses,
            force=True,
        )

        if on_progress is not None:
            on_progress(audio_seconds)

        return chunks

    def _append_context_buffer(
        self,
        conversation_buffer: list[dict[str, Any]],
        buffered_duration: float,
        chunk: TranscriptChunk,
        audio_path: Path,
        audio_analyzer: AudioAnalyzer | None,
        sound_events: Sequence[SoundEvent],
    ) -> float:
        if audio_analyzer is None:
            return buffered_duration

        audio_features = audio_analyzer.analyze_segment(audio_path, chunk.start, chunk.end)
        chunk_events = events_overlapping_window(sound_events, chunk.start, chunk.end)
        conversation_buffer.append(
            {
                "timestamp": f"[{format_timestamp(chunk.start)} -> {format_timestamp(chunk.end)}]",
                "text": chunk.text,
                "audio_meta": audio_features.to_meta_text(),
                "events": format_context_events(chunk_events, chunk.text),
            }
        )
        return buffered_duration + max(0.0, chunk.end - chunk.start)

    def _maybe_run_context_analysis(
        self,
        conversation_buffer: list[dict[str, Any]],
        buffered_duration: float,
        context_window_seconds: float,
        context_analysis_runner: ContextAnalysisRunner | None,
        on_context_analysis: ContextAnalysisCallback | None,
        context_analyses: list[ContextAnalysis],
        force: bool = False,
    ) -> float:
        if not conversation_buffer or context_analysis_runner is None:
            return buffered_duration
        if not force and buffered_duration < context_window_seconds:
            return buffered_duration

        analysis = context_analysis_runner(list(conversation_buffer))
        conversation_buffer.clear()
        if analysis is not None:
            context_analyses.append(analysis)
            if on_context_analysis is not None:
                on_context_analysis(analysis)
        return 0.0

    @staticmethod
    def _extract_words(segment: object) -> list[WordInfo]:
        words = getattr(segment, "words", None) or []
        extracted: list[WordInfo] = []
        for word in words:
            text = str(getattr(word, "word", "")).strip()
            if not text:
                continue
            start = float(getattr(word, "start", getattr(segment, "start", 0.0)))
            end = float(getattr(word, "end", getattr(segment, "end", start)))
            extracted.append(WordInfo(start=start, end=end, word=text))
        return extracted

    @staticmethod
    def _should_flush_chunk(words: Sequence[WordInfo]) -> bool:
        if not words:
            return False
        duration = words[-1].end - words[0].start
        last_word = words[-1].word.rstrip()
        sentence_endings = (".", "!", "?", ";", "\u3002", "\uff01", "\uff1f", "\uff1b")
        has_sentence_end = last_word.endswith(sentence_endings)
        return duration >= 5.0 and has_sentence_end

    @staticmethod
    def _words_to_chunk(
        words: Sequence[WordInfo],
        sound_events: Sequence[SoundEvent] = (),
    ) -> TranscriptChunk:
        text = build_interpolated_text(words, sound_events)
        text = (
            text.replace(" ,", ",")
            .replace(" .", ".")
            .replace(" !", "!")
            .replace(" ?", "?")
            .replace(" ;", ";")
            .replace(" :", ":")
        )
        text = normalize_simplified_chinese(text)
        return TranscriptChunk(start=words[0].start, end=words[-1].end, text=text)

    @staticmethod
    def format_chunks(chunks: Sequence[TranscriptChunk]) -> str:
        return "\n".join(
            f"[{format_timestamp(chunk.start)} -> {format_timestamp(chunk.end)}] {chunk.text}"
            for chunk in chunks
        )

    def _record_until_silence(
        self,
        silence_seconds: float,
        max_record_seconds: float,
        sample_rate: int,
        chunk_size: int,
        vad_threshold: int,
    ) -> Path:
        backend = self._open_audio_input(sample_rate, chunk_size)
        frames: list[bytes] = []
        speech_started = False
        silent_chunks = 0
        required_silent_chunks = max(1, math.ceil(silence_seconds * sample_rate / chunk_size))
        max_chunks = max(1, math.ceil(max_record_seconds * sample_rate / chunk_size))

        try:
            for _ in range(max_chunks):
                data = backend.read(chunk_size)
                is_speech = self._is_speech(data, vad_threshold)

                if is_speech:
                    speech_started = True
                    silent_chunks = 0
                    frames.append(data)
                    continue

                if speech_started:
                    frames.append(data)
                    silent_chunks += 1
                    if silent_chunks >= required_silent_chunks:
                        break

            if not frames:
                raise MicrophoneError("No speech was detected.")

            return self._write_temp_wav(frames, sample_rate)
        except OSError as exc:
            raise MicrophoneError(
                "Could not access the microphone. Check device availability and permissions."
            ) from exc
        finally:
            backend.close()

    @staticmethod
    def _open_audio_input(sample_rate: int, chunk_size: int) -> AudioInputBackend:
        try:
            return PyAudioInputBackend(sample_rate=sample_rate, chunk_size=chunk_size)
        except ImportError:
            pass
        except OSError:
            raise

        try:
            return SoundDeviceInputBackend(sample_rate=sample_rate, chunk_size=chunk_size)
        except ImportError as exc:
            raise MicrophoneError(
                "No microphone backend is installed. Install sounddevice or PyAudio."
            ) from exc

    @staticmethod
    def _is_speech(data: bytes, threshold: int) -> bool:
        samples = array("h")
        samples.frombytes(data)
        if not samples:
            return False
        square_sum = sum(sample * sample for sample in samples)
        rms = math.sqrt(square_sum / len(samples))
        return rms >= threshold

    @staticmethod
    def _write_temp_wav(frames: Sequence[bytes], sample_rate: int) -> Path:
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()

        with wave.open(str(temp_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"".join(frames))

        return temp_path


def format_timestamp(seconds: float) -> str:
    whole_seconds = int(seconds)
    milliseconds = int((seconds - whole_seconds) * 1000)
    minutes, sec = divmod(whole_seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minute:02d}:{sec:02d}.{milliseconds:03d}"
    return f"{minute:02d}:{sec:02d}.{milliseconds:03d}"


def normalize_simplified_chinese(text: str) -> str:
    """Convert Chinese text to Simplified Chinese when zhconv is available."""
    try:
        from zhconv import convert
    except ImportError:
        return text
    return str(convert(text, "zh-cn"))


def build_interpolated_text(
    words: Sequence[WordInfo],
    sound_events: Sequence[SoundEvent],
) -> str:
    """Insert emotion-cue tags into the exact word-timestamp position."""
    if not words:
        return ""

    emotion_events = [
        event
        for event in events_overlapping_window(sound_events, words[0].start, words[-1].end)
        if event.category == "emotion-cue"
    ]
    if not emotion_events:
        return " ".join(word.word.strip() for word in words).strip()

    insertions: dict[int, list[SoundEvent]] = {}
    for event in sorted(emotion_events, key=lambda item: (item.start, item.label)):
        position = find_emotion_insert_position(words, event.start)
        insertions.setdefault(position, []).append(event)

    tokens: list[str] = []
    for index, word in enumerate(words):
        tokens.extend(emotion_tag_from_event(event) for event in insertions.get(index, []))
        tokens.append(word.word.strip())
    tokens.extend(emotion_tag_from_event(event) for event in insertions.get(len(words), []))
    return " ".join(token for token in tokens if token).strip()


def find_emotion_insert_position(words: Sequence[WordInfo], timestamp: float) -> int:
    """Return the token index where an event timestamp belongs."""
    if timestamp <= words[0].start:
        return 0

    for index in range(len(words) - 1):
        current_word = words[index]
        next_word = words[index + 1]
        if current_word.end <= timestamp <= next_word.start:
            return index + 1
        if current_word.start <= timestamp <= current_word.end:
            midpoint = current_word.start + ((current_word.end - current_word.start) / 2)
            return index if timestamp < midpoint else index + 1

    last_word = words[-1]
    if last_word.start <= timestamp <= last_word.end:
        midpoint = last_word.start + ((last_word.end - last_word.start) / 2)
        return len(words) - 1 if timestamp < midpoint else len(words)
    return len(words)


def emotion_tag_from_event(event: SoundEvent) -> str:
    label = event.label.split(",", maxsplit=1)[0].strip()
    label = re.sub(r"[^A-Za-z0-9 _-]+", "", label).strip()
    label = label.replace(" ", "")
    return f"[{label or 'Emotion'}]"


def extract_event_tokens(text: str) -> list[str]:
    """Extract bracketed event markers such as [Laughter] or [Sigh]."""
    return [token.strip() for token in re.findall(r"\[([^\]]+)\]", text) if token.strip()]


def format_context_events(events: Sequence[SoundEvent], text: str) -> list[str] | str:
    """Format raw-audio sound events plus any explicit Whisper event tokens."""
    formatted = [
        (
            f"{event.label} ({event.category}, score={event.score:.2f}, "
            f"{format_timestamp(event.start)}->{format_timestamp(event.end)})"
        )
        for event in events
    ]
    if not formatted:
        formatted.extend(f"Transcript token: {token}" for token in extract_event_tokens(text))
    return formatted if formatted else "None"


class PyAudioInputBackend:
    def __init__(self, sample_rate: int, chunk_size: int) -> None:
        import pyaudio

        self._pyaudio = pyaudio.PyAudio()
        try:
            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk_size,
            )
        except Exception:
            self._pyaudio.terminate()
            raise

    def read(self, chunk_size: int) -> bytes:
        return bytes(self._stream.read(chunk_size, exception_on_overflow=False))

    def close(self) -> None:
        self._stream.stop_stream()
        self._stream.close()
        self._pyaudio.terminate()


class SoundDeviceInputBackend:
    def __init__(self, sample_rate: int, chunk_size: int) -> None:
        import sounddevice

        self._stream = sounddevice.RawInputStream(
            samplerate=sample_rate,
            blocksize=chunk_size,
            channels=1,
            dtype="int16",
        )
        self._stream.start()

    def read(self, chunk_size: int) -> bytes:
        data, _overflowed = self._stream.read(chunk_size)
        return bytes(data)

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
