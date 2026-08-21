"""Guards for the translation catalogs.

The failure mode these exist for is silent: :func:`gettext.translation` is
called with ``fallback=True``, so a catalog that was never compiled, or one
whose ``msgid`` no longer matches the source, produces English instead of an
error.  Nobody files a bug about an interface that looks like it is working.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rvc.lib import i18n  # noqa: E402
from tools import i18n_tool  # noqa: E402


def _template() -> dict:
    messages = i18n_tool.parse_po(i18n_tool.POT_PATH)
    messages.pop("", None)
    return messages


def test_every_shipped_language_has_a_compiled_catalog():
    for tag in i18n.LANGUAGES:
        assert i18n.is_available(tag), (
            f"{tag} has no compiled catalog. Run: python tools/i18n_tool.py all"
        )


def test_template_is_current():
    """The ``.pot`` matches what the sources actually ask to translate.

    Regenerated into a scratch copy rather than compared by timestamp: a stale
    template silently drops every string added since it was written.
    """
    before = i18n_tool.POT_PATH.read_text(encoding="utf-8")
    try:
        i18n_tool.extract()
        after = i18n_tool.POT_PATH.read_text(encoding="utf-8")
    finally:
        i18n_tool.POT_PATH.write_text(before, encoding="utf-8")

    def messages(text: str) -> set[str]:
        # Only the msgids matter; the header carries a creation timestamp that
        # changes on every run and means nothing here.
        return {
            line for line in text.splitlines()
            if line.startswith(("msgid ", '"'))
            and not line.startswith(('#', '"POT-Creation-Date'))
        }

    assert messages(before) == messages(after), (
        "locales/shiromiya.pot is out of date. Run: python tools/i18n_tool.py all"
    )


@pytest.mark.parametrize("tag", [t for t in i18n.LANGUAGES if t != i18n.DEFAULT_LANGUAGE])
def test_translations_keep_their_placeholders(tag: str):
    """``{name}`` in a msgid has to survive into its translation.

    A dropped or renamed placeholder is a ``KeyError`` at the ``.format()``
    call, in one language only, on whatever screen happens to use it.
    """
    import re

    placeholders = lambda text: set(re.findall(r"\{(\w+)\}", text))  # noqa: E731

    po_path = i18n.LOCALE_DIR / tag / "LC_MESSAGES" / f"{i18n_tool.DOMAIN}.po"
    entries = i18n_tool.parse_po(po_path)
    entries.pop("", None)

    problems = []
    for msgid, entry in entries.items():
        expected = placeholders(msgid)
        for translated in [entry["msgstr"], *entry.get("plurals", [])]:
            if translated and placeholders(translated) != expected:
                problems.append((msgid, translated))
    assert not problems, f"placeholder mismatch in {tag}: {problems}"


def test_catalog_actually_translates():
    """End to end: a known string comes back in Portuguese."""
    previous = i18n.current_language()
    try:
        i18n.install("pt_BR")
        assert i18n._("Settings") == "Configurações"
        # A regional tag with no catalog of its own falls back to its base
        # language rather than all the way to English.
        assert i18n.resolve("pt_PT") in i18n.LANGUAGES
    finally:
        i18n.install(previous)


def test_normalisation_of_language_tags():
    assert i18n.normalize("pt-br") == "pt_BR"
    assert i18n.normalize("pt_BR.UTF-8") == "pt_BR"
    assert i18n.normalize("C") is None
    assert i18n.normalize("") is None


def test_explicit_choice_beats_the_system(monkeypatch):
    monkeypatch.delenv("RVC_LANGUAGE", raising=False)
    assert i18n.resolve(explicit="en", stored="pt_BR") == "en"
    assert i18n.resolve(explicit=None, stored="pt_BR") == "pt_BR"
    monkeypatch.setenv("RVC_LANGUAGE", "pt_BR")
    assert i18n.resolve(explicit=None, stored="en") == "pt_BR"
    assert i18n.resolve(explicit="en", stored=None) == "en"


def test_argv_flag_parsing():
    assert i18n.language_from_argv(["app.py", "--language", "pt_BR"]) == "pt_BR"
    assert i18n.language_from_argv(["app.py", "--language=pt_BR"]) == "pt_BR"
    assert i18n.language_from_argv(["app.py", "--share"]) is None


def test_the_cli_is_not_translated():
    """``core.py`` is a machine interface; the GUI parses its output."""
    source = (ROOT / "core.py").read_text(encoding="utf-8")
    assert "from rvc.lib.i18n import" not in source
    assert "core.py" not in " ".join(i18n_tool.SOURCE_DIRS)


def test_compiling_is_reproducible(tmp_path):
    """``tools/i18n_tool.py compile`` runs clean from a shell."""
    result = subprocess.run(
        [sys.executable, "tools/i18n_tool.py", "compile"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
