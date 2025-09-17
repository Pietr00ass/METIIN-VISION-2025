"""Utilities for loading and saving shared configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

CONFIG_FILE = Path("config.json")
DEFAULT_TELEPORT_SEQUENCE: List[int] = list(range(1, 9))


def load_config() -> Dict[str, Any]:
    """Load the full configuration file.

    Returns an empty dictionary if the file does not exist or contains
    invalid JSON. The function logs a warning when the JSON cannot be
    decoded.
    """

    if not CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Nie udało się wczytać pliku %s: %s", CONFIG_FILE, exc)
        return {}


def save_config(data: Dict[str, Any]) -> None:
    """Persist *data* to :data:`CONFIG_FILE`."""

    CONFIG_FILE.write_text(
        json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8"
    )


def load_teleport_sequence(config: Dict[str, Any] | None = None) -> List[int]:
    """Return the configured teleport sequence.

    Falls back to :data:`DEFAULT_TELEPORT_SEQUENCE` if no custom sequence is
    defined. Invalid entries (non-integer values or indices outside the
    1..8 range) are ignored.
    """

    data = config if config is not None else load_config()
    teleport_config = data.get("teleport", {})
    raw_sequence = teleport_config.get("sequence")
    if not isinstance(raw_sequence, list):
        return DEFAULT_TELEPORT_SEQUENCE.copy()

    cleaned: List[int] = []
    for value in raw_sequence:
        try:
            index = int(value)
        except (TypeError, ValueError):
            logger.debug("Pomijam niepoprawną pozycję teleportu: %r", value)
            continue
        if not 1 <= index <= 8:
            logger.debug("Pomijam pozycję teleportu poza zakresem 1..8: %s", index)
            continue
        if index not in cleaned:
            cleaned.append(index)
    if not cleaned:
        return DEFAULT_TELEPORT_SEQUENCE.copy()
    return cleaned
