from __future__ import annotations

import re
import unicodedata
import warnings

import numpy as np

from .languages import supports_any_benchmark_language

DEFAULT_TEXT_MODEL_ID = "livekit/turn-detector"
DEFAULT_TEXT_MODEL_REVISION = "v0.4.1-intl"


class LiveKitTextTurnDetectorAdapter:
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_TEXT_MODEL_ID,
        revision: str = DEFAULT_TEXT_MODEL_REVISION,
        max_length: int = 128,
        onnx_filename: str = "model_q8.onnx",
        onnx_subfolder: str = "onnx",
    ) -> None:
        warnings.warn(
            "LiveKit text adapters are deprecated; use "
            "eot_harness.livekit_turn_detector_adapter:LiveKitTurnDetectorAdapter "
            "or eot_harness.livekit_turn_detector_mini_adapter:LiveKitTurnDetectorMiniAdapter.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.model_id = model_id
        self.revision = revision
        self.max_length = int(max_length)
        self.adapter_id = f"{model_id}-{revision}" if revision else model_id
        self._score_cache: dict[str, float] = {}

        onnx_path = _download_onnx_path(
            model_id=model_id,
            revision=revision,
            onnx_filename=onnx_filename,
            onnx_subfolder=onnx_subfolder,
        )
        self.tokenizer = _load_tokenizer(model_id=model_id, revision=revision)
        self.tokenizer.padding_side = "left"
        self.session = _make_ort_session(onnx_path)

    def supports_language(self, lang_code: str) -> bool:
        return supports_any_benchmark_language(lang_code)

    def predict_batch(self, batch):
        prompts = [self._format_chat_input(item.get("messages") or []) for item in batch]
        scores = [0.0] * len(batch)
        uncached_by_prompt: dict[str, list[int]] = {}
        for idx, prompt in enumerate(prompts):
            if not prompt:
                continue
            cached = self._score_cache.get(prompt)
            if cached is not None:
                scores[idx] = cached
                continue
            uncached_by_prompt.setdefault(prompt, []).append(idx)

        if not uncached_by_prompt:
            return scores

        uncached_prompts = list(uncached_by_prompt)
        scored = self._batch_text_probs(uncached_prompts)
        for prompt, score in zip(uncached_prompts, scored, strict=True):
            value = float(score)
            self._score_cache[prompt] = value
            for idx in uncached_by_prompt[prompt]:
                scores[idx] = value
        return scores

    def _batch_text_probs(self, texts: list[str]) -> list[float]:
        token_rows: list[np.ndarray] = []
        for text in texts:
            enc = self.tokenizer(
                text,
                add_special_tokens=False,
                return_tensors="np",
                max_length=self.max_length,
                truncation=True,
            )
            token_rows.append(enc["input_ids"].astype(np.int64)[0])

        buckets: dict[int, list[tuple[int, np.ndarray]]] = {}
        for idx, ids in enumerate(token_rows):
            buckets.setdefault(int(ids.shape[0]), []).append((idx, ids))

        scores = [0.0] * len(texts)
        for items in buckets.values():
            batch_ids = np.stack([ids for _, ids in items], axis=0).astype(np.int64)
            output = self.session.run(None, {"input_ids": batch_ids})[0]
            logits = np.asarray(output).reshape(len(items), -1)[:, -1]
            for (original_idx, _), logit in zip(items, logits, strict=True):
                scores[original_idx] = float(logit)
        return scores

    def _format_chat_input(self, messages: list[dict]) -> str:
        chat: list[dict[str, str]] = []
        for message in messages:
            role = str(message.get("role") or "").strip()
            if role not in {"assistant", "user"}:
                continue
            content = _normalize_text(str(message.get("content") or ""))
            if not content:
                continue
            chat.append({"role": role, "content": content})

        if not chat:
            return ""
        text = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=False,
            add_special_tokens=False,
            tokenize=False,
        )
        cut = text.rfind("<|im_end|>")
        if cut != -1:
            text = text[:cut]
        return text


class LiveKitText2Adapter(LiveKitTextTurnDetectorAdapter):
    def __init__(self) -> None:
        super().__init__(
            model_id="livekit/qwen-davidai-panels-multilingual",
            revision="main",
        )


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text.lower())
    text = "".join(ch for ch in text if not (unicodedata.category(ch).startswith("P") and ch not in ["'", "-"]))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _download_onnx_path(*, model_id: str, revision: str, onnx_filename: str, onnx_subfolder: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=model_id,
        filename=onnx_filename,
        subfolder=onnx_subfolder,
        revision=revision,
    )


def _load_tokenizer(*, model_id: str, revision: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        truncation_side="left",
    )


def _make_ort_session(onnx_path: str):
    import onnxruntime as ort

    return ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )
