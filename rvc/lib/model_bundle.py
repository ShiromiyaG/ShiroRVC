from __future__ import annotations

import io
import os
from os import PathLike
from pathlib import Path
from typing import Any

import zstandard as zstd


MODEL_BUNDLE_EXTENSION = ".srvc"
MODEL_BUNDLE_EXTENSIONS = (MODEL_BUNDLE_EXTENSION,)
MODEL_FILE_EXTENSIONS = (".pth", MODEL_BUNDLE_EXTENSION)
MODEL_BUNDLE_FORMAT = "shiromiya-rvc-model-bundle"
MODEL_BUNDLE_VERSION = 1


def is_model_bundle(path: str | PathLike[str] | None) -> bool:
    """Return whether a path uses the model bundle extension."""
    if not path:
        return False
    return Path(path).suffix.lower() in MODEL_BUNDLE_EXTENSIONS


def is_model_file(path: str | PathLike[str] | None) -> bool:
    """Return whether a path is a regular checkpoint or a model bundle."""
    if not path:
        return False
    return Path(path).suffix.lower() in MODEL_FILE_EXTENSIONS


def load_model_bundle(path: str | PathLike[str]) -> dict[str, Any]:
    """Load and validate a Zstandard-compressed model bundle."""
    import torch

    bundle_path = Path(path)
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Model bundle not found: {bundle_path}")

    with bundle_path.open("rb") as compressed_file:
        decompressor = zstd.ZstdDecompressor()
        with decompressor.stream_reader(compressed_file) as reader:
            payload = reader.read()

    bundle_data = torch.load(
        io.BytesIO(payload),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(bundle_data, dict):
        raise ValueError("The model bundle must contain a dictionary.")
    _validate_bundle_data(bundle_data)
    return bundle_data


def _validate_bundle_data(bundle_data: dict[str, Any]) -> None:
    models = bundle_data.get("models")
    if "models" in bundle_data:
        if not isinstance(models, dict) or not models:
            raise ValueError("The model bundle does not contain any models.")
        for model_name, model_entry in models.items():
            if not isinstance(model_name, str) or not model_name.strip():
                raise ValueError("The model bundle contains an invalid model name.")
            if not isinstance(model_entry, dict) or not isinstance(
                model_entry.get("model_state"), dict
            ):
                raise ValueError(f"Model entry '{model_name}' is invalid.")
        return

    if isinstance(bundle_data.get("model_state"), dict) or "config" in bundle_data:
        return

    raise ValueError("The file is not a recognized RVC model bundle.")


def save_model_bundle(
    path: str | PathLike[str],
    bundle_data: dict[str, Any],
    compression_level: int = 3,
) -> Path:
    """Write a bundle atomically using Zstandard compression."""
    import torch

    if not isinstance(bundle_data, dict):
        raise TypeError("The model bundle must be a dictionary.")
    _validate_bundle_data(bundle_data)

    compression_level = int(compression_level)
    if not 1 <= compression_level <= 22:
        raise ValueError("Compression level must be between 1 and 22.")

    bundle_path = Path(path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = bundle_path.with_name(f".{bundle_path.name}.tmp")

    try:
        compressor = zstd.ZstdCompressor(level=compression_level)
        with temporary_path.open("wb") as compressed_file:
            with compressor.stream_writer(compressed_file) as writer:
                torch.save(bundle_data, writer)
        os.replace(temporary_path, bundle_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return bundle_path


def get_bundle_models(bundle_data: dict[str, Any]) -> dict[str, Any]:
    """Return the multi-model mapping, or an empty mapping for single bundles."""
    models = bundle_data.get("models")
    return models if isinstance(models, dict) else {}


def get_bundle_model_state(
    bundle_data: dict[str, Any],
    model_name: str | None = None,
) -> dict[str, Any] | None:
    """Return one checkpoint state from a multi- or single-model bundle."""
    models = get_bundle_models(bundle_data)
    if models:
        if not model_name or model_name not in models:
            return None
        model_entry = models[model_name]
        state = model_entry.get("model_state") if isinstance(model_entry, dict) else None
        return state if isinstance(state, dict) else None

    state = bundle_data.get("model_state")
    if isinstance(state, dict):
        return state
    return bundle_data if "config" in bundle_data else None
