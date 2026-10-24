from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from typing import Any

import numpy as np

from .languages import supports_any_benchmark_language
from .streaming_stt import SAMPLE_RATE, resample_audio

LIVEKIT_TURN_DETECTOR_MINI_MODEL_ID = "turn-detector-v1-mini"
LIVEKIT_TURN_DETECTOR_MINI_LOCAL_REVISION = "livekit-local-inference"
LIVEKIT_TURN_DETECTOR_MINI_AUDIO_SEC = 1.2
LIVEKIT_LOCAL_INFERENCE_PACKAGE = "livekit-local-inference"
LIVEKIT_TURN_DETECTOR_MINI_WORKERS_ENV = "LIVEKIT_TURN_DETECTOR_MINI_WORKERS"
LIVEKIT_TURN_DETECTOR_MINI_MAX_DEFAULT_WORKERS = 8


class LiveKitTurnDetectorMiniAdapter:
    display_name = "LiveKit Turn Detector v1-mini"
    score_point = 0.2

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        audio_sec: float = LIVEKIT_TURN_DETECTOR_MINI_AUDIO_SEC,
        num_workers: int | None = None,
    ) -> None:
        self.model_id = LIVEKIT_TURN_DETECTOR_MINI_MODEL_ID
        self.revision = LIVEKIT_TURN_DETECTOR_MINI_LOCAL_REVISION
        self.sample_rate = int(sample_rate)
        self.audio_sec = float(audio_sec)
        self.max_audio_sec = self.audio_sec
        self.max_samples = int(round(self.sample_rate * self.audio_sec))
        self.num_workers = _resolve_num_workers(num_workers)
        self._thread_local = threading.local()
        self._models: list[Any] = []
        self._models_lock = threading.Lock()
        self._executor: ThreadPoolExecutor | None = None
        version = _local_inference_version()
        self.adapter_id = f"livekit/{self.model_id}-{self.revision}-{version}"
        self.model = _load_local_eot_model() if self.num_workers == 1 else None
        if self.num_workers > 1:
            self._executor = ThreadPoolExecutor(
                max_workers=self.num_workers,
                thread_name_prefix="lk-turn-detector-mini",
            )

    def predict_batch(self, batch):
        if self._executor is None:
            return [self._predict_item(item) for item in batch]
        return list(self._executor.map(self._predict_item, batch))

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        for model in self._models:
            close = getattr(model, "close", None)
            if callable(close):
                close()
        self._models.clear()
        if self.model is not None:
            close = getattr(self.model, "close", None)
            if callable(close):
                close()
            self.model = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def supports_language(self, lang_code: str) -> bool:
        return supports_any_benchmark_language(lang_code)

    def _predict_item(self, item: dict[str, Any]) -> float:
        pcm = _prepare_pcm16_tail(
            item,
            sample_rate=self.sample_rate,
            max_samples=self.max_samples,
        )
        return float(self._worker_model().predict(pcm))

    def _worker_model(self):
        if self.model is not None:
            return self.model
        model = getattr(self._thread_local, "model", None)
        if model is None:
            model = _load_local_eot_model()
            self._thread_local.model = model
            with self._models_lock:
                self._models.append(model)
        return model


def _resolve_num_workers(num_workers: int | None) -> int:
    if num_workers is None:
        env_value = os.getenv(LIVEKIT_TURN_DETECTOR_MINI_WORKERS_ENV)
        if env_value is not None and env_value.strip():
            num_workers = int(env_value)
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        num_workers = min(LIVEKIT_TURN_DETECTOR_MINI_MAX_DEFAULT_WORKERS, max(1, cpu_count))
    value = int(num_workers)
    if value <= 0:
        raise ValueError("LiveKit Turn Detector mini num_workers must be positive.")
    return value


def _prepare_pcm16_tail(item: dict[str, Any], *, sample_rate: int, max_samples: int) -> np.ndarray:
    audio = item.get("audio") or {}
    arr = np.asarray(audio["array"], dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    elif arr.ndim != 1:
        raise ValueError(f"LiveKit Turn Detector mini adapter expects mono audio, got shape={arr.shape}")

    input_sample_rate = int(audio["sampling_rate"])
    if input_sample_rate != sample_rate:
        arr = resample_audio(arr, input_sample_rate, sample_rate)

    if max_samples > 0:
        arr = arr[-max_samples:]
    elif max_samples == 0:
        arr = arr[:0]
    return _float_audio_to_pcm16(arr)


def _float_audio_to_pcm16(array: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(array, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    return np.ascontiguousarray(pcm)


def _load_local_eot_model():
    try:
        from livekit.local_inference import EOT
    except ImportError as exc:
        raise RuntimeError(
            "LiveKit Turn Detector mini requires `livekit-local-inference` "
            "(installed by `livekit-agents>=1.6.0rc1`)."
        ) from exc
    return EOT()


def _local_inference_version() -> str:
    try:
        return metadata.version(LIVEKIT_LOCAL_INFERENCE_PACKAGE)
    except metadata.PackageNotFoundError:
        return "unknown"
