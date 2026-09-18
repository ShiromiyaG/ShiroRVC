import sys
import os

now_dir = os.getcwd()
sys.path.append(now_dir)


class InstallationError(Exception):
    def __init__(self, message="InstallationError"):
        self.message = message
        super().__init__(self.message)


def check_installation():
    # Windows only: these are the paths its launch scripts and tools choke on.
    if sys.platform != "win32":
        return
    if "OneDrive" in now_dir:
        raise InstallationError(
            "Installation Error: The current working directory is located in OneDrive. Please move Shiromiya RVC Fork to a different folder."
        )
    if " " in now_dir:
        raise InstallationError(
            "Installation Error: The current working directory contains spaces. Please move Shiromiya RVC Fork to a folder without spaces in its path."
        )
    try:
        now_dir.encode("ascii")
    except UnicodeEncodeError:
        raise InstallationError(
            "Installation Error: The current working directory contains non-ASCII characters. Please move Shiromiya RVC Fork to a folder with only ASCII characters in its path."
        )
