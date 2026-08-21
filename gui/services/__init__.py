"""The one place in the GUI that is allowed to know the host application exists.

Views and widgets import from here; they never import ``core`` or ``rvc``
themselves.  When the backend's signatures move, exactly one directory has to
follow them.
"""
