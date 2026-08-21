"""Platform window effects, where the platform offers real ones.

Qt Widgets can fake depth with painted shadows, but it cannot blur what is
*behind* a window -- only the compositor can.  On Windows 11 the DWM will do
it for us, which is both better looking and free, since the desktop compositor
is already sampling that region.

Everything here degrades to a no-op.  A missing API, an older build or a
non-Windows host must cost nothing but the effect.  There is no equivalent to
reach for elsewhere: a system backdrop is the compositor's to draw, and X11 and
Wayland have no portable way to ask for one.

Note the absence of ``ctypes.wintypes``.  Importing it on Linux raises -- it
declares ``VARIANT_BOOL`` with a type code only the Windows build of ``_ctypes``
knows -- and this module is imported unconditionally by ``gui.app``, so the
whole interface would fail to start there.  ``HWND`` is a ``void *`` anyway.
"""

from __future__ import annotations

import ctypes
import sys

#: DwmSetWindowAttribute keys.
_DWMWA_USE_IMMERSIVE_DARK_MODE = 20
_DWMWA_WINDOW_CORNER_PREFERENCE = 33
_DWMWA_SYSTEMBACKDROP_TYPE = 38

#: DWM_SYSTEMBACKDROP_TYPE values.
BACKDROP_AUTO = 0
BACKDROP_NONE = 1
BACKDROP_MICA = 2
BACKDROP_ACRYLIC = 3
BACKDROP_TABBED = 4

BACKDROPS = {
    "none": BACKDROP_NONE,
    "mica": BACKDROP_MICA,
    "acrylic": BACKDROP_ACRYLIC,
    "tabbed": BACKDROP_TABBED,
}

#: DWMWA_WINDOW_CORNER_PREFERENCE: 2 == round.
_CORNER_ROUND = 2

#: The system backdrop attribute landed in Windows 11 22H2.  Below that the
#: call succeeds against a different attribute number and does something else,
#: so the build is checked rather than the return value.
_BACKDROP_MIN_BUILD = 22621
_DARK_TITLEBAR_MIN_BUILD = 18985


def windows_build() -> int:
    if sys.platform != "win32":
        return 0
    try:
        return int(sys.getwindowsversion().build)
    except (AttributeError, ValueError):
        return 0


def supports_backdrop() -> bool:
    return windows_build() >= _BACKDROP_MIN_BUILD


def _set_attribute(handle: int, attribute: int, value: int) -> bool:
    try:
        dwm = ctypes.windll.dwmapi
    except (AttributeError, OSError):
        return False
    payload = ctypes.c_int(value)
    try:
        result = dwm.DwmSetWindowAttribute(
            ctypes.c_void_p(handle),
            ctypes.c_uint(attribute),
            ctypes.byref(payload),
            ctypes.sizeof(payload),
        )
    except (OSError, ctypes.ArgumentError):
        return False
    return result == 0


def apply_titlebar_theme(window, dark: bool) -> bool:
    """Match the native title bar to the application theme.

    Without this the frame stays light while the window below it is dark,
    which is the single most obvious sign that an application is not really
    themed.
    """
    if windows_build() < _DARK_TITLEBAR_MIN_BUILD:
        return False
    handle = int(window.winId())
    return _set_attribute(handle, _DWMWA_USE_IMMERSIVE_DARK_MODE, 1 if dark else 0)


class _Margins(ctypes.Structure):
    _fields_ = [
        ("cxLeftWidth", ctypes.c_int),
        ("cxRightWidth", ctypes.c_int),
        ("cyTopHeight", ctypes.c_int),
        ("cyBottomHeight", ctypes.c_int),
    ]


def _extend_frame(handle: int, enable: bool) -> bool:
    """Extend the DWM frame across the client area, or put it back.

    This is what actually makes a system backdrop visible.  Setting
    ``DWMWA_SYSTEMBACKDROP_TYPE`` on its own does nothing: measured against a
    window with a transparent client area, Mica and Acrylic both come out as
    flat black until the frame is extended over that area.

    Qt's ``WA_TranslucentBackground`` produces the same result as extending the
    frame, but it is a creation-time attribute -- toggling it forces the window
    to be destroyed and rebuilt, which on a window this size is seconds of
    blank white.  This call works on a live window, so the backdrop can be
    switched from the sidebar without a restart.
    """
    try:
        dwm = ctypes.windll.dwmapi
    except (AttributeError, OSError):
        return False
    # -1 on every edge means "the whole client area"; zeros restore the normal
    # frame.
    margins = _Margins(-1, -1, -1, -1) if enable else _Margins(0, 0, 0, 0)
    try:
        return dwm.DwmExtendFrameIntoClientArea(
            ctypes.c_void_p(handle), ctypes.byref(margins)
        ) == 0
    except (OSError, ctypes.ArgumentError):
        return False


def apply_backdrop(window, backdrop: str) -> bool:
    """Ask the compositor to blur what is behind the window.

    ``acrylic`` is a strong blur of whatever sits behind the window.  ``mica``
    tints with the desktop wallpaper instead and is far subtler by design --
    on a dark wallpaper in dark mode it is close to invisible, which is the
    effect working, not failing.

    The window's own background has to be transparent for either to show; the
    main window paints it that way itself, in ``MainWindow.paintEvent``.
    """
    if not supports_backdrop():
        return False
    value = BACKDROPS.get(backdrop, BACKDROP_NONE)
    handle = int(window.winId())
    _set_attribute(handle, _DWMWA_WINDOW_CORNER_PREFERENCE, _CORNER_ROUND)
    _extend_frame(handle, value != BACKDROP_NONE)
    return _set_attribute(handle, _DWMWA_SYSTEMBACKDROP_TYPE, value)
