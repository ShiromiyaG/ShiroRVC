import time
import logging
from tensorboard import program

from rvc.lib.terminal import success

log_path = "logs"


def launch_tensorboard_pipeline():
    logging.getLogger("root").setLevel(logging.WARNING)
    logging.getLogger("tensorboard").setLevel(logging.WARNING)

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", log_path])
    url = tb.launch()

    # The query string pins the four losses worth watching, so the link is
    # given whole rather than as a bare host:port.
    pinned = (
        "?pinnedCards=%5B%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22loss"
        "%2Fg%2Ftotal%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22"
        "loss%2Fd%2Ftotal%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A"
        "%22loss%2Fg%2Fkl%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A"
        "%22loss%2Fg%2Fmel%22%7D%5D"
    )
    success(f"TensorBoard is up at {url}{pinned}", tag="[TENSORBOARD]")

    while True:
        time.sleep(600)
