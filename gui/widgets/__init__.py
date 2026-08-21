"""Presentation-only building blocks.

Nothing in here imports :mod:`gui.services`; widgets take values and emit
signals, and the views wire them to the backend.  That split is what keeps a
widget testable without a worker process behind it.
"""

#: Qt's "no maximum" sentinel, which PySide6 does not export.  Needed wherever
#: a maximum size is set temporarily -- to animate a widget's height, say --
#: and then has to be given back.
QWIDGETSIZE_MAX = (1 << 24) - 1
