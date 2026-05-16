from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"


@dataclass(frozen=True)
class ContextAnalysis:
    text: str
    model: str


class OllamaUnavailable(RuntimeError):
    """Raised when Ollama cannot be reached or returns an invalid response."""


class OllamaAnalyzer:
    def __init__(
        self,
        model: str = "llama3",
        api_url: str = DEFAULT_OLLAMA_URL,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.model = model
        self.api_url = api_url
        self.timeout_seconds = timeout_seconds

    def analyze_context(self, conversation_buffer: Sequence[dict[str, Any]]) -> ContextAnalysis:
        try:
            import requests
        except ImportError as exc:
            raise OllamaUnavailable(
                "Thieu dependency `requests`. Hay chay `pip install -r requirements.txt`."
            ) from exc

        payload = {
            "model": self.model,
            "prompt": self._build_prompt(conversation_buffer),
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_ctx": 4096,
            },
        }
        try:
            response = requests.post(self.api_url, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaUnavailable(
                "Khong ket noi duoc Ollama tai localhost:11434. Hay chay `ollama serve` "
                "va bao dam model da duoc pull, vi du `ollama pull llama3`."
            ) from exc

        data = response.json()
        analysis = str(data.get("response", "")).strip()
        if not analysis:
            raise OllamaUnavailable("Ollama tra ve phan hoi rong.")
        return ContextAnalysis(text=analysis, model=self.model)

    def _build_prompt(self, conversation_buffer: Sequence[dict[str, Any]]) -> str:
        return (
            "Ban la chuyen gia phan tich hoi thoai, cam xuc, sac thai va ngu canh.\n"
            "Du lieu gom transcript theo timestamp, metadata am thanh nhe (pitch/volume), "
            "va cac event am thanh phat hien tu audio goc bang YAMNet.\n\n"
            "Dieu quan trong: cac event nhu Laughter, Giggle, Cough, Sigh, Crying, "
            "Applause, Music, Background noise la bang chung am thanh rieng, khong phai "
            "noi dung loi noi. Hay dung chung de suy luan sac thai, cam xuc, do cang "
            "thang, su mia mai, su do du, hoac thay doi tam trang neu co co so.\n"
            "Khong duoc bo qua truong `events` neu no khac `None`. Neu khong du bang "
            "chung, hay noi ro la tin hieu chua du manh thay vi khang dinh qua muc.\n\n"
            "Tra loi bang MOT doan van tieng Viet ngan gon, ro rang, huu ich. "
            "Khong dung bullet list.\n\n"
            f"Du lieu doan hoi thoai:\n{list(conversation_buffer)}"
        )
