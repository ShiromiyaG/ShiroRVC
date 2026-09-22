"""YAML configs for the style model."""

from __future__ import annotations

import os

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "style_flow")


def load_config(path: str) -> dict:
    """Read a YAML config.  ``inherit: other.yaml`` (relative to the file)
    is loaded first and overridden key by key, one level deep per section."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    parent = data.pop("inherit", None)
    if parent:
        base = load_config(os.path.join(os.path.dirname(os.path.abspath(path)), parent))
        for key, value in data.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key] = {**base[key], **value}
            else:
                base[key] = value
        data = base
    return data


def default_config_path(name: str) -> str:
    return os.path.normpath(os.path.join(CONFIG_DIR, name))
