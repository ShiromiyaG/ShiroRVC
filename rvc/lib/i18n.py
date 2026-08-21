"""Translation for every surface of the application.

One catalog, one backend (:mod:`gettext` from the standard library), shared by
the Gradio interface, the native GUI and anything under ``rvc/``.  The CLI in
``core.py`` deliberately stays English -- its output is parsed by the GUI and
quoted in bug reports, so it is a machine interface first.

Two rules make the difference between this working and silently falling back to
English forever:

1. :func:`install` has to run **before** any user interface is constructed.
   Gradio builds its tabs at import time and Qt builds its views in the widget
   constructors, so both freeze whatever ``_()`` returned at that moment.
2. Nothing translated may live at module scope.  Use :func:`N_` to mark such a
   string for extraction and call :func:`_` at the point of use.
"""

from __future__ import annotations

import gettext as _gettext
import os
import sys
from pathlib import Path

#: Catalog (``.mo``) base name, i.e. ``locales/<lang>/LC_MESSAGES/<DOMAIN>.mo``.
DOMAIN = "shiromiya"

#: ``locales/`` at the application root, three levels up from ``rvc/lib/``.
LOCALE_DIR = Path(__file__).resolve().parents[2] / "locales"

#: Languages the application ships, in the order a selector should list them.
#: The value is what a human picks in a menu; the key is what goes on disk, in
#: ``--language`` and in the preferences file.
LANGUAGES: dict[str, str] = {
    "en": "English",
    "pt_BR": "Português (Brasil)",
}

DEFAULT_LANGUAGE = "en"

_translation: _gettext.NullTranslations = _gettext.NullTranslations()
_current: str = DEFAULT_LANGUAGE


def _(message: str) -> str:
    """Translate ``message`` into the installed language."""
    return _translation.gettext(message)


def N_(message: str) -> str:  # noqa: N802 - the conventional gettext spelling
    """Mark a string for extraction without translating it yet.

    For strings that have to be defined at module scope (tables, constants,
    argument defaults).  Wrap the *use* site in :func:`_`.
    """
    return message


def ngettext(singular: str, plural: str, count: int) -> str:
    """Plural-aware lookup.  ``count`` picks the form, not the substitution."""
    return _translation.ngettext(singular, plural, count)


def normalize(language: str | None) -> str | None:
    """Canonicalise a language tag to the spelling used on disk.

    ``pt-br``, ``pt_BR.UTF-8`` and ``pt-BR`` all mean ``pt_BR``.
    """
    if not language:
        return None
    tag = language.strip().replace("-", "_").split(".")[0].split("@")[0]
    if not tag or tag in ("C", "POSIX"):
        return None
    parts = tag.split("_")
    if len(parts) == 1:
        return parts[0].lower()
    return f"{parts[0].lower()}_{parts[1].upper()}"


def _fallback_chain(language: str) -> list[str]:
    """``pt_BR`` -> ``["pt_BR", "pt", "en"]``.

    gettext walks this list and takes the first catalog that exists, so a
    regional variant degrades to its base language before it degrades to
    English.
    """
    chain = [language]
    if "_" in language:
        chain.append(language.split("_")[0])
    if DEFAULT_LANGUAGE not in chain:
        chain.append(DEFAULT_LANGUAGE)
    return chain


def system_language() -> str:
    """The user's preferred *display* language, e.g. ``pt_BR``.

    Deliberately not :func:`locale.getdefaultlocale`: on Windows that reports
    the *format* locale (the one that decides decimal separators), so a machine
    in Brazil running an English Windows would be handed a Portuguese interface
    against the user's explicit choice.  It is also deprecated since 3.11.
    """
    if sys.platform == "win32":
        tag = _windows_ui_language()
        if tag:
            return tag
    # POSIX, and Windows when someone has set these by hand.  gettext reads
    # them itself, but only these -- which on Windows are almost never set,
    # hence the explicit probe above and the explicit ``languages=`` below.
    for variable in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        value = os.environ.get(variable)
        tag = normalize((value or "").split(":")[0])
        if tag:
            return tag
    return DEFAULT_LANGUAGE


def _windows_ui_language() -> str | None:
    """First entry of ``GetUserPreferredUILanguages``.

    That list is what the Settings app calls "Windows display language", which
    is the question being asked here.
    """
    try:
        import ctypes

        MUI_LANGUAGE_NAME = 0x8
        kernel32 = ctypes.windll.kernel32
        count = ctypes.c_ulong()
        size = ctypes.c_ulong(0)
        if not kernel32.GetUserPreferredUILanguages(
            MUI_LANGUAGE_NAME, ctypes.byref(count), None, ctypes.byref(size)
        ):
            return None
        buffer = ctypes.create_unicode_buffer(size.value)
        if not kernel32.GetUserPreferredUILanguages(
            MUI_LANGUAGE_NAME, ctypes.byref(count), buffer, ctypes.byref(size)
        ):
            return None
        # A double-NUL-terminated multi-string; ``.value`` stops at the first
        # NUL, which is the highest-priority language.
        return normalize(buffer.value)
    except Exception:
        return None  # no ctypes, or a Windows without the MUI API


def resolve(explicit: str | None = None, stored: str | None = None) -> str:
    """Decide which language to install.

    Precedence, highest first::

        explicit (--language)  >  RVC_LANGUAGE  >  stored preference
                               >  system display language  >  English

    ``stored`` being ``None`` is meaningful and is not the same as ``"en"``: it
    means the user has never touched the selector, so the interface should keep
    following the operating system.  Someone who deliberately chose English
    stays on English even after switching Windows to Spanish.
    """
    for candidate in (explicit, os.environ.get("RVC_LANGUAGE"), stored):
        tag = normalize(candidate)
        if tag:
            return tag if tag in LANGUAGES else _closest(tag)
    return _closest(system_language())


def _closest(tag: str) -> str:
    """Best shipped language for a tag we may not have a catalog for."""
    if tag in LANGUAGES:
        return tag
    base = tag.split("_")[0]
    for shipped in LANGUAGES:
        if shipped.split("_")[0] == base:
            return shipped
    return DEFAULT_LANGUAGE


def install(language: str | None = None) -> str:
    """Load the catalog for ``language``.  Returns what was actually installed.

    Safe to call with a language that has no catalog: ``fallback=True`` means a
    missing ``.mo`` yields untranslated English rather than an exception.  That
    is also what happens when a release ships ``.po`` files without compiling
    them, so :mod:`tools.i18n_tool` is wired into both build paths.
    """
    global _translation, _current
    resolved = _closest(normalize(language) or DEFAULT_LANGUAGE)
    _translation = _gettext.translation(
        DOMAIN,
        localedir=str(LOCALE_DIR),
        languages=_fallback_chain(resolved),
        fallback=True,
    )
    _current = resolved
    return resolved


def install_resolved(explicit: str | None = None, stored: str | None = None) -> str:
    """:func:`resolve` then :func:`install`, which is what callers want."""
    return install(resolve(explicit, stored))


def current_language() -> str:
    return _current


def language_name(tag: str) -> str:
    return LANGUAGES.get(tag, tag)


def is_available(tag: str) -> bool:
    """Whether a compiled catalog actually exists for ``tag``."""
    if tag == DEFAULT_LANGUAGE:
        return True  # the source language needs no catalog
    return (LOCALE_DIR / tag / "LC_MESSAGES" / f"{DOMAIN}.mo").is_file()


def language_from_argv(argv: list[str] | None = None) -> str | None:
    """``--language pt_BR`` or ``--language=pt_BR`` from a raw argument list.

    Read before argparse exists, because the Gradio interface has to install
    the catalog before it imports the tabs that build the widgets.
    """
    args = sys.argv if argv is None else argv
    for index, argument in enumerate(args):
        if argument == "--language" and index + 1 < len(args):
            return args[index + 1]
        if argument.startswith("--language="):
            return argument.split("=", 1)[1]
    return None
