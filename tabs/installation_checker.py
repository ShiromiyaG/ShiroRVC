import sys

from rvc.lib.paths import ROOT

app_root = str(ROOT)


class InstallationError(Exception):
    def __init__(self, message="InstallationError"):
        self.message = message
        super().__init__(self.message)


def check_installation():
    # Windows only: these are the paths its launch scripts and tools choke on.
    if sys.platform != "win32":
        return
    if "OneDrive" in app_root:
        raise InstallationError(
            "Installation Error: The installation folder is located in OneDrive. Please move Shiromiya RVC Fork to a different folder."
        )
    if " " in app_root:
        raise InstallationError(
            "Installation Error: The installation folder contains spaces. Please move Shiromiya RVC Fork to a folder without spaces in its path."
        )
    try:
        app_root.encode("ascii")
    except UnicodeEncodeError:
        raise InstallationError(
            "Installation Error: The installation folder contains non-ASCII characters. Please move Shiromiya RVC Fork to a folder with only ASCII characters in its path."
        )
