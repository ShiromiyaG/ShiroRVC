"""One module per top-level screen.

Views own layout and validation; they reach the backend only through
:mod:`gui.services.engine`, and report progress upwards with signals so the
main window remains the single place that knows about the status bar.
"""
