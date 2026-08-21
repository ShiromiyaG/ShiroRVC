"""Translation for the native interface: the one place ``gui/`` names ``rvc``.

The catalog itself lives in ``rvc/lib/i18n.py`` so that the Gradio interface,
the native one and anything under ``rvc/`` share one set of ``.po`` files -- a
string translated once is translated everywhere it appears.  Reimplementing the
lookup here to avoid the import would buy nothing and cost the thing that makes
the design worth having: two catalogs that drift.

``tests/test_gui_isolation.py`` exempts this module by name, and only this one.
The rules it enforces are about the *backend* -- views and widgets must not
reach past ``gui/services/`` into torch, ``core.py`` or the training code, so
that the window can be built without a worker and ``gui/`` stays deletable.
``rvc.lib.i18n`` is stdlib ``gettext`` and a path; it imports nothing, starts
nothing, and deleting ``gui/`` still leaves it working for the other two
interfaces.  Widgets import this rather than ``gui.services`` for the same
reason: it is presentation, not a service.
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
