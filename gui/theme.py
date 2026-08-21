"""Palette and stylesheet for the application.

The stylesheet lives in ``resources/style.qss`` with ``@token`` placeholders
that this module substitutes, so a palette change is data rather than a pile of
string concatenation, and the QSS stays readable in an editor.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette

from .services import paths

ACCENTS = {
    "violet": "#8b7cf6",
    "teal": "#2dd4bf",
    "amber": "#f59e0b",
    "rose": "#fb7185",
    "sky": "#38bdf8",
}

DARK = {
    "bg": "#101014",
    "bg_alt": "#16161c",
    "surface": "#1c1c24",
    "surface_alt": "#22222c",
    "border": "#2e2e3a",
    "border_strong": "#3d3d4d",
    "text": "#e8e8ef",
    "text_dim": "#9a9aab",
    "text_faint": "#6b6b7d",
    "input": "#14141a",
    #: The pale core of a slider handle.  Not ``text``: that is near-black in
    #: the light theme, which turned the handle into a black dot on a violet
    #: ring.  A knob has to read as raised in both themes, so it is pale in
    #: both -- the ring carries the accent, not the core.
    "knob": "#f2f2f7",
    "success": "#4ade80",
    "warning": "#fbbf24",
    "danger": "#f87171",
    "shadow": "rgba(0, 0, 0, 110)",
}

LIGHT = {
    "bg": "#f4f4f7",
    "bg_alt": "#ffffff",
    "surface": "#ffffff",
    "surface_alt": "#f7f7fa",
    "border": "#dedee6",
    "border_strong": "#c4c4d0",
    "text": "#1b1b22",
    "text_dim": "#5c5c6b",
    "text_faint": "#8a8a99",
    "input": "#ffffff",
    "knob": "#ffffff",
    "success": "#16a34a",
    "warning": "#d97706",
    "danger": "#dc2626",
    "shadow": "rgba(0, 0, 0, 28)",
}


def tokens(mode: str = "dark", accent: str = "violet") -> dict[str, str]:
    """Palette for a theme.

    Deliberately independent of the compositor backdrop: that is toggled with
    a window attribute, not with colours, so switching it never has to re-apply
    the stylesheet.
    """
    base = dict(DARK if mode == "dark" else LIGHT)
    colour = ACCENTS.get(accent, ACCENTS["violet"])
    base["accent"] = colour
    base["accent_hover"] = _shift(colour, 18 if mode == "dark" else -12)
    base["accent_pressed"] = _shift(colour, -18)
    # Selection and hover washes are the accent at low alpha, which keeps them
    # correct against both backgrounds without a second hand-picked colour.
    base["accent_soft"] = _alpha(colour, 38)
    base["accent_ghost"] = _alpha(colour, 20)
    base["mode"] = mode
    return base


def _shift(hex_colour: str, amount: int) -> str:
    colour = QColor(hex_colour)
    return colour.lighter(100 + amount).name() if amount >= 0 else colour.darker(100 - amount).name()


def _alpha(hex_colour: str, alpha: int) -> str:
    colour = QColor(hex_colour)
    return f"rgba({colour.red()}, {colour.green()}, {colour.blue()}, {alpha})"


def stylesheet(mode: str = "dark", accent: str = "violet") -> str:
    """The themed sheet.  Belongs on the main window, not on the application.

    Measured on this layout: applying it to the ``QApplication`` costs 2.1 s,
    because Qt then re-resolves every rule against every widget in the process
    -- ~1500 of them.  The identical sheet on the main window's subtree, which
    is all of the same widgets, costs 0.3 s and renders identically.  So the
    scope is not an implementation detail; it is the theme switch's speed.
    """
    source = (paths.RESOURCE_DIR / "style.qss").read_text(encoding="utf-8")
    values = tokens(mode, accent)
    values.update(_glyph_urls(values, mode, accent))
    for key, value in values.items():
        source = source.replace(f"@{key}@", str(value))
    return source


def _glyph_urls(values: dict[str, str], mode: str, accent: str) -> dict[str, str]:
    """Render the few glyphs QSS needs as images, and return their URLs.

    Qt's stylesheet engine cannot draw a chevron or a tick from CSS: both need
    a zero-sized box with borders, which it renders as a filled block.  Leaving
    them to Fusion means a heavier, differently-weighted glyph sitting beside a
    hand-drawn icon set, and a checkbox whose "checked" state is a plain filled
    square.  Rasterising ours once per theme is what makes them match.

    Written under the GUI's own state directory and keyed by theme, so a
    palette switch does not reuse the previous theme's colour.  A failure here
    is not worth a broken window: the QSS falls back to no image, which is what
    it looked like before.
    """
    from .widgets import icons

    wanted = {
        "chevron_glyph": ("chevron", values["text_dim"], 2.0),
        "chevron_glyph_accent": ("chevron", values["accent"], 2.0),
        "check_glyph": ("check", "#ffffff", 2.4),
        "check_glyph_dim": ("check", values["text_faint"], 2.4),
        "disclosure_glyph": ("chevron", values["text_dim"], 2.0),
    }
    urls: dict[str, str] = {}
    try:
        directory = paths.STATE_DIR / "glyphs"
        directory.mkdir(parents=True, exist_ok=True)
        for token, (name, colour, width) in wanted.items():
            target = directory / f"{token}-{mode}-{accent}.png"
            if not target.exists():
                pixmap = icons.pixmap(name, colour, 18, width)
                if pixmap.isNull() or not pixmap.save(str(target)):
                    return dict.fromkeys(wanted, "none")
            # QSS wants forward slashes even on Windows.
            urls[token] = f"url({target.as_posix()})"
    except Exception:  # noqa: BLE001 - styling must never stop the window
        return dict.fromkeys(wanted, "none")
    return urls


def global_stylesheet() -> str:
    """The application-wide sheet: tooltips, and nothing else worth 2.1 s.

    Deliberately free of literal colours -- it is set once at startup and
    follows the theme through :func:`apply_palette` instead of being re-applied.
    """
    return (paths.RESOURCE_DIR / "global.qss").read_text(encoding="utf-8")


def apply_palette(app, mode: str = "dark") -> None:
    """Set the QPalette too, not just the stylesheet.

    QSS does not reach native pieces like tooltips, the text cursor or the
    dialog chrome; without this they stay light-on-light in dark mode.
    """
    values = DARK if mode == "dark" else LIGHT
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(values["bg"]))
    palette.setColor(QPalette.WindowText, QColor(values["text"]))
    palette.setColor(QPalette.Base, QColor(values["input"]))
    palette.setColor(QPalette.AlternateBase, QColor(values["surface_alt"]))
    palette.setColor(QPalette.Text, QColor(values["text"]))
    palette.setColor(QPalette.Button, QColor(values["surface"]))
    palette.setColor(QPalette.ButtonText, QColor(values["text"]))
    palette.setColor(QPalette.ToolTipBase, QColor(values["surface"]))
    palette.setColor(QPalette.ToolTipText, QColor(values["text"]))
    palette.setColor(QPalette.PlaceholderText, QColor(values["text_faint"]))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(values["text_faint"]))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(values["text_faint"]))
    app.setPalette(palette)


def monospace_font(size: int = 9) -> QFont:
    """A fixed-pitch face for the log console and numeric readouts."""
    for family in ("Cascadia Mono", "Consolas", "JetBrains Mono", "DejaVu Sans Mono"):
        if family in QFontDatabase.families():
            font = QFont(family, size)
            break
    else:
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        font.setPointSize(size)
    font.setStyleHint(QFont.Monospace)
    return font
