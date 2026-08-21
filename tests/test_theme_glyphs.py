"""The stylesheet's rasterised glyphs.

Qt's stylesheet engine cannot draw a chevron or a tick: both need a zero-sized
box with CSS borders, which it renders as a filled block.  So the combo arrow
and the checkbox tick are our own icons rendered to PNGs and referenced as
images.  That is a small amount of machinery between the theme and the window,
and the failure modes are quiet ones -- an unsubstituted ``@token@`` in the
sheet makes Qt drop the whole rule, and a glyph rendered in the wrong theme's
colour is invisible rather than wrong.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("PySide6", reason="the Qt interface is optional")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _without_comments(sheet: str) -> str:
    """QSS with ``/* ... */`` removed.

    The sheet documents its own ``@token@`` placeholder mechanism, and names
    the malformed selector it warns against, so scanning the raw text finds
    both in prose and reports them as defects.
    """
    return re.sub(r"/\*.*?\*/", "", sheet, flags=re.S)


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_every_token_is_substituted(app, mode):
    """A leftover ``@name@`` makes Qt discard the rule it appears in."""
    from gui import theme

    sheet = _without_comments(theme.stylesheet(mode, "violet"))
    leftovers = set(re.findall(r"@[a-z_]+@", sheet))
    assert not leftovers, f"unsubstituted tokens: {sorted(leftovers)}"


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_the_glyph_images_exist_on_disk(app, mode):
    from gui import theme

    sheet = theme.stylesheet(mode, "violet")
    urls = re.findall(r"image:\s*url\(([^)]+)\)", sheet)
    assert urls, "no glyph images were referenced"
    for url in urls:
        assert Path(url).is_file(), f"{url} was referenced but not written"


def test_themes_do_not_share_a_glyph_file(app):
    """Keyed by theme, or a switch reuses the previous palette's colour."""
    from gui import theme

    dark = set(re.findall(r"image:\s*url\(([^)]+)\)", theme.stylesheet("dark", "violet")))
    light = set(re.findall(r"image:\s*url\(([^)]+)\)", theme.stylesheet("light", "violet")))
    assert dark and light
    assert not (dark & light), "the two themes point at the same PNG"


def test_accents_do_not_share_a_glyph_file(app):
    from gui import theme

    violet = set(re.findall(r"image:\s*url\(([^)]+)\)", theme.stylesheet("dark", "violet")))
    teal = set(re.findall(r"image:\s*url\(([^)]+)\)", theme.stylesheet("dark", "teal")))
    assert not (violet & teal)


def test_a_broken_glyph_render_does_not_break_the_window(app, monkeypatch, tmp_path):
    """Styling is not worth a failed startup; the sheet falls back to no image."""
    from gui import theme
    from gui.services import paths
    from gui.widgets import icons

    def explode(*args, **kwargs):
        raise RuntimeError("no raster backend")

    # Into a fresh state directory, or the already-written PNGs are reused and
    # the renderer is never reached.
    monkeypatch.setattr(paths, "STATE_DIR", tmp_path)
    monkeypatch.setattr(icons, "pixmap", explode)

    sheet = theme.stylesheet("dark", "violet")
    assert "@chevron_glyph@" not in sheet, "a failed render left a raw token behind"
    assert "image: none" in sheet


def test_the_knob_colour_is_pale_in_both_themes(app):
    """It used to be ``text``, which is near-black in the light theme -- the
    slider handle came out as a black dot inside a violet ring."""
    from PySide6.QtGui import QColor

    from gui import theme

    for mode in ("dark", "light"):
        knob = QColor(theme.tokens(mode)["knob"])
        assert knob.lightness() > 200, f"{mode} knob is not pale: {knob.name()}"


def test_the_combo_arrow_is_declared_once(app):
    """Two rules for the same subcontrol drew two arrows per dropdown.

    ``QComboBox:hover::down-arrow`` -- the pseudo-class before the subcontrol
    -- made Qt add a second arrow at the combo's natural width rather than
    restyle the first.
    """
    sheet = _without_comments(
        (ROOT / "gui" / "resources" / "style.qss").read_text(encoding="utf-8")
    )
    assert "QComboBox:hover::down-arrow" not in sheet, (
        "pseudo-class before subcontrol: this draws a second arrow"
    )


def test_the_slider_handle_stays_round(app):
    """Qt measures ``border-radius`` against the content box.

    Setting it to half the *outer* size squares the handle off, which is what
    a radius larger than half the width does.
    """
    sheet = (ROOT / "gui" / "resources" / "style.qss").read_text(encoding="utf-8")
    block = re.search(
        r"QSlider::handle:horizontal \{(.*?)\}", sheet, re.S
    )
    assert block, "the handle rule went missing"
    body = block.group(1)
    width = int(re.search(r"width:\s*(\d+)px", body).group(1))
    radius = int(re.search(r"border-radius:\s*(\d+)px", body).group(1))
    assert radius * 2 <= width, f"radius {radius} exceeds half of width {width}"
