import asyncio
import os
import sys

import edge_tts

# Run as a script, so the repository root is not on sys.path by itself.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from rvc.lib.terminal import install_rich_print

install_rich_print()


async def main():
    tts_file = str(sys.argv[1])
    text = str(sys.argv[2])
    voice = str(sys.argv[3])
    rate = int(sys.argv[4])
    output_file = str(sys.argv[5])

    rates = f"+{rate}%" if rate >= 0 else f"{rate}%"
    if tts_file and os.path.exists(tts_file):
        text = ""
        try:
            with open(tts_file, "r", encoding="utf-8") as file:
                text = file.read()
        except UnicodeDecodeError:
            with open(tts_file, "r") as file:
                text = file.read()
    await edge_tts.Communicate(text, voice, rate=rates).save(output_file)


if __name__ == "__main__":
    asyncio.run(main())
