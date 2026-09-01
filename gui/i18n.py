"""Translation for the native interface: the one place ``gui/`` names ``rvc``.

The catalog lives in ``rvc/lib/i18n.py`` so the Gradio interface, the native
one and anything under ``rvc/`` share one set of ``.po`` files instead of two
that can drift. ``tests/test_gui_isolation.py`` exempts this module by name:
``rvc.lib.i18n`` is stdlib ``gettext`` plus a path, imports nothing and starts
nothing, so importing it does not break ``gui/``'s deletability the way
reaching into torch or ``core.py`` would.
"""

from __future__ import annotations

from rvc.lib.i18n import (  # noqa: F401 - re-exported deliberately
    DEFAULT_LANGUAGE,
    LANGUAGES,
    N_,
    _,
    current_language,
    install,
    install_resolved,
    is_available,
    language_from_argv,
    language_name,
    ngettext,
    normalize,
    resolve,
)

__all__ = [
    "DEFAULT_LANGUAGE",
    "LANGUAGES",
    "N_",
    "_",
    "current_language",
    "install",
    "install_resolved",
    "is_available",
    "language_from_argv",
    "language_name",
    "ngettext",
    "normalize",
    "resolve",
]
