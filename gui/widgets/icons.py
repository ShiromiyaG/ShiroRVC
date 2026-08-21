"""Line icons drawn with QPainter.

No image files and no icon font: a stroked path scales cleanly to any DPI, and
recolouring it for the current theme is a pen change rather than a second set
of assets.  Everything is defined on a 24x24 grid with a 2 px stroke, which is
what keeps them looking like one family.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt
from PySide6.QtGui import (
    QColor,
    QIcon,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)

GRID = 24.0


def _waveform(path: QPainterPath) -> None:
    """Inference: a level meter."""
    for x, height in ((4, 4), (8, 8), (12, 10), (16, 7), (20, 3)):
        path.moveTo(x, 12 - height)
        path.lineTo(x, 12 + height)


def _speech(path: QPainterPath) -> None:
    """Text to speech: a bubble with a tail."""
    path.addRoundedRect(QRectF(3, 4, 18, 13), 4, 4)
    path.moveTo(8, 17)
    path.lineTo(7, 21)
    path.lineTo(12, 17)


def _trend(path: QPainterPath) -> None:
    """Training: axes with a falling loss curve."""
    path.moveTo(4, 3)
    path.lineTo(4, 20)
    path.lineTo(21, 20)
    path.moveTo(7, 16)
    path.cubicTo(10, 16, 10, 7, 13, 7)
    path.cubicTo(16, 7, 17, 12, 20, 12)


def _sliders(path: QPainterPath) -> None:
    """Utilities: a small control surface."""
    for y in (7, 12, 17):
        path.moveTo(4, y)
        path.lineTo(20, y)
    for x, y in ((9, 7), (15, 12), (7, 17)):
        path.addEllipse(QPointF(x, y), 2.4, 2.4)


def _play(path: QPainterPath) -> None:
    path.moveTo(7, 4)
    path.lineTo(20, 12)
    path.lineTo(7, 20)
    path.closeSubpath()


def _pause(path: QPainterPath) -> None:
    path.addRoundedRect(QRectF(7, 5, 3.4, 14), 1.4, 1.4)
    path.addRoundedRect(QRectF(14, 5, 3.4, 14), 1.4, 1.4)


#: Geometry of the reload ring, shared by its two passes.
_REFRESH_CENTRE = 12.0
_REFRESH_RADIUS = 7.0
_REFRESH_START = 55.0
#: Negative sweeps clockwise in Qt's angle convention, which is the direction a
#: reload glyph reads as turning.  290 leaves a 70-degree gap at the top for
#: the arrowhead to sit in.
_REFRESH_SWEEP = -290.0


def _refresh_end() -> tuple[float, float, float, float]:
    """``(x, y, tangent_x, tangent_y)`` at the open end of the ring."""
    angle = math.radians(_REFRESH_START + _REFRESH_SWEEP)
    x = _REFRESH_CENTRE + _REFRESH_RADIUS * math.cos(angle)
    y = _REFRESH_CENTRE - _REFRESH_RADIUS * math.sin(angle)
    # Qt's y axis points down, so the counter-clockwise tangent is
    # (-sin, -cos) and a clockwise sweep flips it.
    direction = -1.0 if _REFRESH_SWEEP < 0 else 1.0
    return x, y, direction * math.sin(angle), direction * math.cos(angle)


def _refresh(path: QPainterPath) -> None:
    """Reload: an open ring. The arrowhead is the filled pass below."""
    box = QRectF(
        _REFRESH_CENTRE - _REFRESH_RADIUS,
        _REFRESH_CENTRE - _REFRESH_RADIUS,
        _REFRESH_RADIUS * 2,
        _REFRESH_RADIUS * 2,
    )
    path.arcMoveTo(box, _REFRESH_START)
    path.arcTo(box, _REFRESH_START, _REFRESH_SWEEP)


def _refresh_details(path: QPainterPath) -> None:
    """A solid arrowhead on the ring's tangent.

    Filled rather than two stroked barbs, which is what this was first drawn
    as.  On a curve one barb always lands along the arc it is attached to, so
    only the other reads as a barb, and at the 15 px this renders at the pair
    merged into a bar stuck to the ring.  A triangle has no such degenerate
    orientation.
    """
    x, y, tangent_x, tangent_y = _refresh_end()
    normal_x, normal_y = -tangent_y, tangent_x

    ahead, behind, half_width = 3.0, 1.4, 2.8
    path.moveTo(x + tangent_x * ahead, y + tangent_y * ahead)
    path.lineTo(
        x - tangent_x * behind + normal_x * half_width,
        y - tangent_y * behind + normal_y * half_width,
    )
    path.lineTo(
        x - tangent_x * behind - normal_x * half_width,
        y - tangent_y * behind - normal_y * half_width,
    )
    path.closeSubpath()


def _close(path: QPainterPath) -> None:
    """Dismiss: a plain cross.

    Drawn corner to corner rather than inset, so at the 13 px the audio
    player's clear button renders it there is enough length for the two
    strokes to read as a cross instead of as a blob.
    """
    path.moveTo(6.5, 6.5)
    path.lineTo(17.5, 17.5)
    path.moveTo(17.5, 6.5)
    path.lineTo(6.5, 17.5)


#: Where the speaker cone ends and the sound waves radiate from.
_VOLUME_MOUTH = 10.5


def _volume(path: QPainterPath) -> None:
    """Volume: the radiating waves. The cone is the filled pass below."""
    for radius in (4.0, 7.0):
        box = QRectF(
            _VOLUME_MOUTH - radius, 12.0 - radius, radius * 2, radius * 2
        )
        path.arcMoveTo(box, 45.0)
        path.arcTo(box, 45.0, -90.0)


def _volume_details(path: QPainterPath) -> None:
    """The speaker cone, filled.

    Stroked, its own outline closes the narrow throat between the box and the
    cone into a smudge below about 20 px; as a solid it keeps its silhouette
    all the way down.
    """
    path.moveTo(3.0, 9.5)
    path.lineTo(6.5, 9.5)
    path.lineTo(_VOLUME_MOUTH, 5.5)
    path.lineTo(_VOLUME_MOUTH, 18.5)
    path.lineTo(6.5, 14.5)
    path.lineTo(3.0, 14.5)
    path.closeSubpath()


def _folder(path: QPainterPath) -> None:
    path.moveTo(3, 19)
    path.lineTo(3, 6)
    path.lineTo(9.5, 6)
    path.lineTo(11.5, 8.6)
    path.lineTo(21, 8.6)
    path.lineTo(21, 19)
    path.closeSubpath()


def _search(path: QPainterPath) -> None:
    path.addEllipse(QPointF(10.5, 10.5), 6.2, 6.2)
    path.moveTo(15.2, 15.2)
    path.lineTo(20.5, 20.5)


def _download(path: QPainterPath) -> None:
    path.moveTo(12, 3)
    path.lineTo(12, 15)
    path.moveTo(7.4, 10.6)
    path.lineTo(12, 15.3)
    path.lineTo(16.6, 10.6)
    path.moveTo(4, 20)
    path.lineTo(20, 20)


def _chevron(path: QPainterPath) -> None:
    """The combo box arrow.

    Qt's stylesheet engine cannot draw a triangle from CSS borders -- the
    zero-sized box that needs comes out as a filled block -- so the arrow used
    to be left to Fusion, which draws a heavier glyph than anything else in
    this set.  Rendered to a pixmap and referenced as an image, it matches.
    """
    path.moveTo(7.5, 10)
    path.lineTo(12, 14.5)
    path.lineTo(16.5, 10)


def _chevron_right(path: QPainterPath) -> None:
    path.moveTo(10, 7.5)
    path.lineTo(14.5, 12)
    path.lineTo(10, 16.5)


def _check(path: QPainterPath) -> None:
    path.moveTo(6, 12.4)
    path.lineTo(10.2, 16.4)
    path.lineTo(18, 8)


def _stop(path: QPainterPath) -> None:
    path.addRoundedRect(QRectF(6, 6, 12, 12), 2.4, 2.4)


def _spark(path: QPainterPath) -> None:
    """Used for anything "start": a run arrow with motion."""
    path.moveTo(13, 2.5)
    path.lineTo(6, 13)
    path.lineTo(11.5, 13)
    path.lineTo(10.5, 21.5)
    path.lineTo(18, 10.5)
    path.lineTo(12.5, 10.5)
    path.closeSubpath()


def _sun(path: QPainterPath) -> None:
    path.addEllipse(QPointF(12, 12), 4.4, 4.4)
    for x1, y1, x2, y2 in (
        (12, 2.5, 12, 5), (12, 19, 12, 21.5), (2.5, 12, 5, 12), (19, 12, 21.5, 12),
        (5.4, 5.4, 7.2, 7.2), (16.8, 16.8, 18.6, 18.6),
        (18.6, 5.4, 16.8, 7.2), (7.2, 16.8, 5.4, 18.6),
    ):
        path.moveTo(x1, y1)
        path.lineTo(x2, y2)


def _moon(path: QPainterPath) -> None:
    path.moveTo(20, 14.5)
    path.cubicTo(18.6, 15.2, 17, 15.6, 15.4, 15.6)
    path.cubicTo(10.4, 15.6, 8.4, 12.4, 8.4, 8.6)
    path.cubicTo(8.4, 6.6, 9, 5, 10, 3.6)
    path.cubicTo(6, 4.6, 3.4, 8.2, 3.4, 12.4)
    path.cubicTo(3.4, 17.4, 7.4, 21, 12.4, 21)
    path.cubicTo(15.8, 21, 18.8, 18.4, 20, 14.5)


def _terminal(path: QPainterPath) -> None:
    path.addRoundedRect(QRectF(2.5, 4.5, 19, 15), 3, 3)
    path.moveTo(7, 10)
    path.lineTo(10.5, 12.6)
    path.lineTo(7, 15.2)
    path.moveTo(13, 15.4)
    path.lineTo(17.5, 15.4)


def _shiba(path: QPainterPath) -> None:
    """The brand mark: a shiba's head, front on.

    Drawn rather than scaled from ``assets/logo.png`` because the logo is a
    detailed illustration -- fur, headphones, a microphone -- and at 20 px it
    collapses into an orange smudge.  A stroked outline survives the size and
    matches the rest of the icon family, which the raster never could.
    """
    # The previous version read as a cat, and the reasons were all proportion
    # rather than detail: tall narrow ears set close, a deep notch between them,
    # and a long muzzle tapering to a point.  Those are feline proportions, and
    # no amount of eye and nose detail argues a viewer out of a silhouette.
    #
    # A shiba, front on, is the opposite on every one of them: the ears are
    # *short* and wide-based, the skull between them is broad and nearly flat,
    # and the muzzle is short and blunt with a wide chin.  Those three, plus the
    # cheek ruff flaring wider than the skull, are what carry at 16 px.
    path.moveTo(4.4, 11.2)           # left temple
    path.lineTo(5.2, 4.8)            # up the outer edge of the left ear
    path.quadTo(5.5, 3.6, 6.7, 4.3)  # rounded tip, not a spike
    path.lineTo(10.0, 6.9)           # inner edge, down to the skull
    # Broad, barely domed forehead. The old brow dipped into a V here, which is
    # the single most cat-like line in the drawing.
    path.cubicTo(11.0, 6.5, 13.0, 6.5, 14.0, 6.9)
    path.lineTo(17.3, 4.3)
    path.quadTo(18.5, 3.6, 18.8, 4.8)
    path.lineTo(19.6, 11.2)          # right temple
    # Cheek ruff, flaring wider than the skull before tucking in. A cat's face
    # narrows continuously from ear to chin; a shiba's steps out here first, and
    # that bulge is what the eye reads as a thick coat.
    path.cubicTo(21.2, 14.1, 19.7, 16.0, 17.1, 16.5)
    path.cubicTo(16.1, 16.7, 15.5, 17.2, 15.2, 18.0)
    # Short, blunt muzzle with a wide chin -- a rounded base, not a point.
    path.cubicTo(14.8, 19.6, 13.5, 20.5, 12.0, 20.5)
    path.cubicTo(10.5, 20.5, 9.2, 19.6, 8.8, 18.0)
    path.cubicTo(8.5, 17.2, 7.9, 16.7, 6.9, 16.5)
    path.cubicTo(4.3, 16.0, 2.8, 14.1, 4.4, 11.2)
    path.closeSubpath()


def _shiba_details(path: QPainterPath) -> None:
    """Inner ears, eyes and nose, filled so they hold at small sizes."""
    # Inner ears follow the shorter outer ear rather than sitting inside it as
    # a separate wedge, which is what made the old pair read as eyebrows.
    path.moveTo(6.5, 5.95)
    path.lineTo(8.9, 7.95)
    path.lineTo(6.0, 8.85)
    path.closeSubpath()
    path.moveTo(17.5, 5.95)
    path.lineTo(15.1, 7.95)
    path.lineTo(18.0, 8.85)
    path.closeSubpath()

    # Slanted wedges, outer corner high. Round eyes are the other half of why
    # the old mark read feline -- a cat's eye is large and circular in the face,
    # a shiba's is small, narrow and angled up toward the ear.  A wedge also
    # survives 16 px, where an outlined almond turns into a smudge.
    path.moveTo(7.2, 10.7)
    path.lineTo(10.3, 11.4)
    path.lineTo(7.6, 12.5)
    path.closeSubpath()
    path.moveTo(16.8, 10.7)
    path.lineTo(13.7, 11.4)
    path.lineTo(16.4, 12.5)
    path.closeSubpath()

    # Nose: a small rounded trapezoid high on the muzzle. The old downward
    # triangle pointed at the chin and the two merged into one blob.
    path.moveTo(10.9, 14.8)
    path.quadTo(12.0, 14.15, 13.1, 14.8)
    path.quadTo(12.85, 16.2, 12.0, 16.2)
    path.quadTo(11.15, 16.2, 10.9, 14.8)
    path.closeSubpath()


def _chip(path: QPainterPath) -> None:
    """GPU / device."""
    path.addRoundedRect(QRectF(6, 6, 12, 12), 2, 2)
    for offset in (9.5, 12, 14.5):
        path.moveTo(offset, 2.5)
        path.lineTo(offset, 6)
        path.moveTo(offset, 18)
        path.lineTo(offset, 21.5)
        path.moveTo(2.5, offset)
        path.lineTo(6, offset)
        path.moveTo(18, offset)
        path.lineTo(21.5, offset)


DRAWINGS = {
    "waveform": _waveform,
    "speech": _speech,
    "trend": _trend,
    "sliders": _sliders,
    "play": _play,
    "pause": _pause,
    "stop": _stop,
    "refresh": _refresh,
    "close": _close,
    "volume": _volume,
    "folder": _folder,
    "search": _search,
    "download": _download,
    "chevron": _chevron,
    "chevron_right": _chevron_right,
    "check": _check,
    "spark": _spark,
    "sun": _sun,
    "moon": _moon,
    "terminal": _terminal,
    "chip": _chip,
    "shiba": _shiba,
}

#: Shapes that read as solid marks rather than outlines.
FILLED = {"play", "pause", "stop", "spark", "moon"}

#: A second, filled pass drawn over the stroked outline, for shapes that need a
#: solid mark next to a stroked one: the shiba's eyes and nose disappear below
#: about 24 px as thin outlines, and the reload arrowhead needs a fill to keep
#: its point at any size.
DETAILS = {
    "shiba": _shiba_details,
    "refresh": _refresh_details,
    "volume": _volume_details,
}


def pixmap(name: str, colour: str, size: int = 20, width: float = 1.9) -> QPixmap:
    """Render one icon at a device-independent size."""
    scale = size / GRID
    canvas = QPixmap(size, size)
    canvas.setDevicePixelRatio(1.0)
    canvas.fill(Qt.transparent)

    drawing = DRAWINGS.get(name)
    if drawing is None:
        return canvas

    path = QPainterPath()
    drawing(path)

    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.scale(scale, scale)

    tint = QColor(colour)
    if name in FILLED:
        painter.setPen(Qt.NoPen)
        painter.setBrush(tint)
    else:
        pen = QPen(tint, width)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
    painter.drawPath(path)

    detail = DETAILS.get(name)
    if detail is not None:
        detail_path = QPainterPath()
        detail(detail_path)
        painter.setPen(Qt.NoPen)
        painter.setBrush(tint)
        painter.drawPath(detail_path)

    painter.end()
    return canvas


def icon(name: str, colour: str, active_colour: str | None = None, size: int = 20) -> QIcon:
    """A QIcon with a separate colour for the selected and disabled states.

    Qt will happily grey out a coloured pixmap on its own, but the result is a
    washed-out tint rather than the theme's own muted colour, so the states are
    supplied explicitly.
    """
    result = QIcon()
    base = pixmap(name, colour, size)
    result.addPixmap(base, QIcon.Normal, QIcon.Off)
    if active_colour:
        highlighted = pixmap(name, active_colour, size)
        result.addPixmap(highlighted, QIcon.Selected, QIcon.Off)
        result.addPixmap(highlighted, QIcon.Active, QIcon.Off)
        result.addPixmap(highlighted, QIcon.Normal, QIcon.On)
    return result


def default_size() -> QSize:
    return QSize(20, 20)


#: Sizes Windows and the Qt window manager actually ask for.
_RASTER_SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)


def _opaque_bounds(image: QImage, probe: int = 64) -> QRect:
    """Bounding box of the non-transparent pixels, in the image's coordinates.

    Scanned on a small thumbnail rather than the full image: a 1254x1254 logo
    is 1.5 million per-pixel calls from Python, and two percent of precision is
    invisible once the result is scaled down to 16 px anyway.
    """
    if not image.hasAlphaChannel():
        return image.rect()

    thumbnail = image.scaled(
        probe, probe, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
    ).convertToFormat(QImage.Format_ARGB32)

    left, top = probe, probe
    right = bottom = -1
    for y in range(probe):
        for x in range(probe):
            if (thumbnail.pixel(x, y) >> 24) & 0xFF > 8:
                left = min(left, x)
                top = min(top, y)
                right = max(right, x)
                bottom = max(bottom, y)

    if right < 0:
        return image.rect()

    scale_x = image.width() / probe
    scale_y = image.height() / probe
    x = int(left * scale_x)
    y = int(top * scale_y)
    width = max(1, int((right - left + 1) * scale_x))
    height = max(1, int((bottom - top + 1) * scale_y))

    # Squared off around the centre. An icon source that is a pixel or two off
    # square makes KeepAspectRatio return 15x16 where 16x16 was asked for, and
    # the window manager then has no exact match to pick.
    side = min(max(width, height), image.width(), image.height())
    x = max(0, min(x - (side - width) // 2, image.width() - side))
    y = max(0, min(y - (side - height) // 2, image.height() - side))
    return QRect(x, y, side, side)


def raster_icon(path: str) -> QIcon:
    """A multi-size QIcon from an image file, trimmed of transparent margin.

    Two things a bare ``QIcon(path)`` gets wrong for a window icon.  The logo
    carries about 11% of empty margin on each side, which at 16 px in a title
    bar throws away a quarter of the available pixels; and letting Qt scale one
    1254 px source on demand is softer than supplying each size explicitly.
    """
    image = QImage(path)
    if image.isNull():
        return QIcon()

    bounds = _opaque_bounds(image)
    if bounds != image.rect():
        image = image.copy(bounds)

    icon = QIcon()
    for size in _RASTER_SIZES:
        icon.addPixmap(
            QPixmap.fromImage(
                image.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        )
    return icon
