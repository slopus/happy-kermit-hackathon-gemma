"""Minimal public adapter contract for the EoT harness.

Adapters are intentionally duck-typed. The harness owns all timing semantics:

- it slices audio causally up to time t
- it applies transcript lag
- it constructs the causal message history

Adapters only receive already-prepared causal inputs and return p(eot).
Adapters may also expose `score_point`, the silence duration used by default
when computing metrics for that model. Adapters without `score_point` are
evaluated over all span scores: scalar metrics use max p(eot), and policy
metrics use the first threshold crossing.
"""

from __future__ import annotations

import importlib


def validate_adapter(adapter) -> None:
    """Fail fast if an adapter does not implement the public contract."""

    adapter_id = getattr(adapter, "adapter_id", None)
    if not isinstance(adapter_id, str) or not adapter_id.strip():
        raise TypeError("Adapter must define a non-empty string attribute `adapter_id`.")

    predict_batch = getattr(adapter, "predict_batch", None)
    if not callable(predict_batch):
        raise TypeError("Adapter must define a callable method `predict_batch(batch)`.")


def validate_streaming_adapter(adapter) -> None:
    """Fail fast if a streaming adapter does not implement the public contract."""

    adapter_id = getattr(adapter, "adapter_id", None)
    if not isinstance(adapter_id, str) or not adapter_id.strip():
        raise TypeError("Adapter must define a non-empty string attribute `adapter_id`.")

    predict_turn = getattr(adapter, "predict_turn", None)
    if not callable(predict_turn):
        raise TypeError(
            "Streaming adapter must define a callable method "
            "`predict_turn(row, *, inference_interval)`.",
        )


def instantiate_adapter(ref: str):
    """Instantiate an adapter from `module:attr` or `module.attr` without validating shape."""

    if ":" in ref:
        module_name, attr_name = ref.split(":", 1)
    else:
        module_name, _, attr_name = ref.rpartition(".")
        if not module_name or not attr_name:
            raise ValueError("Adapter reference must be `module:attr` or `module.attr`.")

    module = importlib.import_module(module_name)
    target = getattr(module, attr_name)
    return target() if isinstance(target, type) else target


def load_adapter(ref: str):
    """Load a pointwise batch adapter from `module:attr` or `module.attr`."""

    adapter = instantiate_adapter(ref)
    validate_adapter(adapter)
    return adapter


def load_streaming_adapter(ref: str):
    """Load a streaming adapter from `module:attr` or `module.attr`."""

    adapter = instantiate_adapter(ref)
    validate_streaming_adapter(adapter)
    return adapter
