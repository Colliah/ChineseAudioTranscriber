from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioFeatureSummary:
    pitch_label: str
    volume_label: str
    average_pitch_hz: float | None
    rms_energy: float | None

    def to_meta_text(self) -> str:
        pitch = self.pitch_label
        volume = self.volume_label
        if self.average_pitch_hz is not None:
            pitch = f"{pitch} ({self.average_pitch_hz:.0f} Hz)"
        if self.rms_energy is not None:
            volume = f"{volume} ({self.rms_energy:.3f} RMS)"
        return f"Pitch: {pitch}, Volume: {volume}"


class AudioAnalyzer:
    """Lightweight pitch and energy extraction for short transcript chunks."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        low_pitch_hz: float = 120.0,
        high_pitch_hz: float = 260.0,
        low_rms: float = 0.015,
        high_rms: float = 0.075,
    ) -> None:
        self.sample_rate = sample_rate
        self.low_pitch_hz = low_pitch_hz
        self.high_pitch_hz = high_pitch_hz
        self.low_rms = low_rms
        self.high_rms = high_rms

    def analyze_segment(self, audio_path: Path, start: float, end: float) -> AudioFeatureSummary:
        try:
            import librosa
            import numpy as np
        except ImportError:
            return AudioFeatureSummary(
                pitch_label="Unavailable",
                volume_label="Unavailable",
                average_pitch_hz=None,
                rms_energy=None,
            )

        duration = max(0.05, end - start)
        try:
            samples, sample_rate = librosa.load(
                audio_path,
                sr=self.sample_rate,
                mono=True,
                offset=max(0.0, start),
                duration=duration,
            )
        except Exception:
            return AudioFeatureSummary(
                pitch_label="Unavailable",
                volume_label="Unavailable",
                average_pitch_hz=None,
                rms_energy=None,
            )

        if len(samples) == 0:
            return AudioFeatureSummary(
                pitch_label="Unavailable",
                volume_label="Unavailable",
                average_pitch_hz=None,
                rms_energy=None,
            )

        rms_energy = float(np.mean(librosa.feature.rms(y=samples)))
        pitch_values, _voiced_flag, _voiced_prob = librosa.pyin(
            samples,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sample_rate,
        )
        voiced_pitch = pitch_values[~np.isnan(pitch_values)]
        average_pitch = float(np.mean(voiced_pitch)) if len(voiced_pitch) else None

        return AudioFeatureSummary(
            pitch_label=self._label_pitch(average_pitch),
            volume_label=self._label_volume(rms_energy),
            average_pitch_hz=average_pitch,
            rms_energy=rms_energy,
        )

    def _label_pitch(self, average_pitch_hz: float | None) -> str:
        if average_pitch_hz is None:
            return "Unknown"
        if average_pitch_hz >= self.high_pitch_hz:
            return "High"
        if average_pitch_hz <= self.low_pitch_hz:
            return "Low"
        return "Normal"

    def _label_volume(self, rms_energy: float | None) -> str:
        if rms_energy is None:
            return "Unknown"
        if rms_energy >= self.high_rms:
            return "High"
        if rms_energy <= self.low_rms:
            return "Low"
        return "Normal"
