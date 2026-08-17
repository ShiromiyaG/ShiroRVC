import re
import unicodedata


def format_title(title: str) -> str:
    formatted_title = unicodedata.normalize("NFC", title)
    formatted_title = re.sub(r"[\u2500-\u257F]+", "", formatted_title)
    formatted_title = re.sub(
        r"[^\w\s.-]",
        "",
        formatted_title,
        flags=re.UNICODE,
    )
    return re.sub(r"\s+", "_", formatted_title)
