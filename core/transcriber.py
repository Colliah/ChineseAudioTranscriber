from __future__ import annotations

import math
import tempfile
import time
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

from faster_whisper import WhisperModel
from pydub import AudioSegment


INITIAL_PROMPT = (
    "Transcribe clearly with accurate punctuation. Context may include English "
    "and Chinese. Convert all Chinese text to Simplified Chinese. Preserve "
    "proper nouns when possible."
)

SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a"}
MAX_AUDIO_SECONDS = 15 * 60


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


@dataclass(frozen=True)
class WordInfo:
    start: float
    end: float
    word: str


ProgressCallback = Callable[[float], None]
ChunkCallback = Callable[[TranscriptChunk], None]


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
        on_progress: ProgressCallback | None = None,
        on_chunk: ChunkCallback | None = None,
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
        )

        chunks = self._consume_segments(
            segments=segments,
            audio_seconds=audio_seconds,
            on_progress=on_progress,
            on_chunk=on_chunk,
        )
        elapsed = time.perf_counter() - started_at
        return TranscriptionResult(
            chunks=chunks,
            text=self.format_chunks(chunks),
            duration_seconds=elapsed,
            audio_seconds=audio_seconds,
        )

    def listen_once(
        self,
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
            return self.transcribe_file(wav_path)
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
        audio_seconds: float,
        on_progress: ProgressCallback | None,
        on_chunk: ChunkCallback | None,
    ) -> list[TranscriptChunk]:
        chunks: list[TranscriptChunk] = []
        pending_words: list[WordInfo] = []
        last_end = 0.0

        for segment in segments:
            words = self._extract_words(segment)
            if words:
                for word in words:
                    pending_words.append(word)
                    last_end = max(last_end, word.end)
                    if self._should_flush_chunk(pending_words):
                        chunk = self._words_to_chunk(pending_words)
                        chunks.append(chunk)
                        if on_chunk is not None:
                            on_chunk(chunk)
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
                    last_end = max(last_end, end)

            if on_progress is not None:
                on_progress(min(last_end, audio_seconds))

        if pending_words:
            chunk = self._words_to_chunk(pending_words)
            chunks.append(chunk)
            if on_chunk is not None:
                on_chunk(chunk)
            last_end = max(last_end, chunk.end)

        if on_progress is not None:
            on_progress(audio_seconds)

        return chunks

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
    def _words_to_chunk(words: Sequence[WordInfo]) -> TranscriptChunk:
        text = " ".join(word.word.strip() for word in words).strip()
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
