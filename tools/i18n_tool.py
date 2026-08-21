"""Translation catalog toolchain, with no dependencies outside the stdlib.

``pybabel`` and the GNU gettext binaries do the same three jobs, but both are
an install step that has to succeed on a contributor's Windows box before they
can fix a typo -- and the Windows one is not a ``pip install``.  The formats
here are the standard ones (``.pot``/``.po``/``.mo``), so Weblate, Crowdin and
Poedit all still work; only the tool is local.

    python tools/i18n_tool.py extract              # sources     -> .pot
    python tools/i18n_tool.py update               # .pot        -> every .po
    python tools/i18n_tool.py update --language pt_BR
    python tools/i18n_tool.py compile              # .po         -> .mo
    python tools/i18n_tool.py stats                # what is still untranslated

``compile`` runs from the Windows build script and the release workflow.  It
has to: a release that ships ``.po`` files without ``.mo`` files falls back to
English without raising anything, which is a bug nobody reports because the
application looks like it is working.
"""

from __future__ import annotations

import argparse
import ast
import re
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCALE_DIR = ROOT / "locales"
DOMAIN = "shiromiya"
POT_PATH = LOCALE_DIR / f"{DOMAIN}.pot"

#: Where translatable strings live.  ``core.py`` is absent on purpose: the CLI
#: stays English (see rvc/lib/i18n.py), and so does everything under rvc/train/
#: whose stdout the GUI parses.
SOURCE_DIRS = ("tabs", "gui", "rvc/infer", "rvc/lib")

EXCLUDE_PARTS = {"__pycache__", ".git", "env", "logs", ".tmp"}

#: Functions whose first string argument is a message.  ``N_`` marks strings
#: defined at module scope that are translated later at the point of use.
KEYWORDS = {"_", "N_", "gettext"}
#: ``ngettext(singular, plural, count)``.
PLURAL_KEYWORDS = {"ngettext"}


# -- extraction ------------------------------------------------------------


class _Collector(ast.NodeVisitor):
    """Finds ``_("...")`` calls and records where each message came from."""

    def __init__(self, relative_path: str):
        self.path = relative_path
        #: ``msgid -> {"plural": str|None, "locations": [(path, line)]}``
        self.messages: dict[str, dict] = {}

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast API
        name = _called_name(node.func)
        if name in KEYWORDS:
            self._record(node, _literal(node.args[0]) if node.args else None)
        elif name in PLURAL_KEYWORDS and len(node.args) >= 2:
            self._record(
                node,
                _literal(node.args[0]),
                plural=_literal(node.args[1]),
            )
        self.generic_visit(node)

    def _record(self, node: ast.Call, msgid: str | None, plural: str | None = None) -> None:
        # ``_(SOME_CONSTANT)`` is the documented deferred pattern: the string
        # was already extracted from the ``N_()`` that defined it, and this
        # call only resolves it at the right moment.
        if msgid is None and node.args and isinstance(node.args[0], (ast.Name, ast.Attribute)):
            return
        # Anything else non-literal -- an f-string, a concatenation -- cannot
        # be extracted at all, and is almost always a mistake, so it is
        # reported rather than skipped quietly.
        if msgid is None:
            print(
                f"  warning: {self.path}:{node.lineno}: "
                "translated call with a non-literal argument; use _() on the "
                "literal and format afterwards",
                file=sys.stderr,
            )
            return
        entry = self.messages.setdefault(msgid, {"plural": None, "locations": []})
        if plural:
            entry["plural"] = plural
        entry["locations"].append((self.path, node.lineno))


def _called_name(func: ast.expr) -> str | None:
    """``_`` from ``_(...)`` and ``i18n._(...)`` alike."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _literal(node: ast.expr | None) -> str | None:
    """The value of a string literal, including implicit concatenation."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _source_files() -> list[Path]:
    files: list[Path] = []
    for directory in SOURCE_DIRS:
        base = ROOT / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if EXCLUDE_PARTS & set(path.parts):
                continue
            files.append(path)
    return files


def extract() -> dict[str, dict]:
    """Scan the sources and write ``locales/shiromiya.pot``."""
    messages: dict[str, dict] = {}
    for path in _source_files():
        relative = path.relative_to(ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as error:
            print(f"  skipped {relative}: {error}", file=sys.stderr)
            continue
        collector = _Collector(relative)
        collector.visit(tree)
        for msgid, found in collector.messages.items():
            entry = messages.setdefault(msgid, {"plural": None, "locations": []})
            entry["plural"] = entry["plural"] or found["plural"]
            entry["locations"].extend(found["locations"])

    LOCALE_DIR.mkdir(parents=True, exist_ok=True)
    POT_PATH.write_text(_render_pot(messages), encoding="utf-8")
    print(f"extract: {len(messages)} messages -> {POT_PATH.relative_to(ROOT)}")
    return messages


def _render_pot(messages: dict[str, dict]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M%z")
    lines = [
        "# Translation template for ShiroRVC.",
        "# Generated by tools/i18n_tool.py -- do not edit by hand.",
        "#",
        'msgid ""',
        'msgstr ""',
        '"Project-Id-Version: ShiroRVC\\n"',
        f'"POT-Creation-Date: {stamp}\\n"',
        '"MIME-Version: 1.0\\n"',
        '"Content-Type: text/plain; charset=UTF-8\\n"',
        '"Content-Transfer-Encoding: 8bit\\n"',
        "",
    ]
    for msgid in sorted(messages):
        entry = messages[msgid]
        for path, line in sorted(set(entry["locations"])):
            lines.append(f"#: {path}:{line}")
        lines.append(f"msgid {_quote(msgid)}")
        if entry["plural"]:
            lines.append(f"msgid_plural {_quote(entry['plural'])}")
            lines.append('msgstr[0] ""')
            lines.append('msgstr[1] ""')
        else:
            lines.append('msgstr ""')
        lines.append("")
    return "\n".join(lines)


def _quote(value: str) -> str:
    """PO string literal.  Multi-line messages use the empty-first-line form."""
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\t", "\\t")
        .replace("\n", "\\n\n")
    )
    if "\n" in escaped.rstrip("\n") or escaped.count("\\n") > 1:
        parts = [part for part in escaped.split("\n")]
        if parts and parts[-1] == "":
            parts.pop()
        return '""\n' + "\n".join(f'"{part}"' for part in parts)
    return '"' + escaped.replace("\n", "") + '"'


# -- .po parsing and updating ----------------------------------------------


_PO_LINE = re.compile(r'^\s*(msgid_plural|msgid|msgstr(?:\[\d+\])?)\s+(.*)$')


def parse_po(path: Path) -> dict[str, dict]:
    """Read a ``.po`` into ``{msgid: {"msgstr": ..., "plurals": [...], "fuzzy": bool}}``."""
    entries: dict[str, dict] = {}
    if not path.is_file():
        return entries

    current: dict[str, object] = {}
    key: str | None = None
    fuzzy = False

    def flush() -> None:
        nonlocal current, key, fuzzy
        msgid = current.get("msgid")
        if msgid is not None:
            entries[str(msgid)] = {
                "msgstr": current.get("msgstr", ""),
                "plural": current.get("msgid_plural"),
                "plurals": current.get("plurals", []),
                "fuzzy": fuzzy,
            }
        current = {}
        key = None
        fuzzy = False

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            if current:
                flush()
            continue
        if line.startswith("#~"):
            continue  # obsolete entry
        if line.startswith("#"):
            if "fuzzy" in line and line.startswith("#,"):
                fuzzy = True
            continue
        match = _PO_LINE.match(line)
        if match:
            key = match.group(1)
            value = _unquote(match.group(2))
            if key.startswith("msgstr["):
                index = int(key[7:-1])
                plurals = current.setdefault("plurals", [])  # type: ignore[assignment]
                while len(plurals) <= index:  # type: ignore[arg-type]
                    plurals.append("")  # type: ignore[union-attr]
                plurals[index] = value  # type: ignore[index]
                key = f"plural:{index}"
            else:
                current[key] = value
            continue
        if line.startswith('"') and key:
            value = _unquote(line)
            if key.startswith("plural:"):
                index = int(key.split(":")[1])
                current["plurals"][index] += value  # type: ignore[index]
            else:
                current[key] = str(current.get(key, "")) + value
    if current:
        flush()
    return entries


def _unquote(text: str) -> str:
    text = text.strip()
    if not (text.startswith('"') and text.endswith('"')):
        return ""
    body = text[1:-1]
    out: list[str] = []
    index = 0
    while index < len(body):
        char = body[index]
        if char == "\\" and index + 1 < len(body):
            index += 1
            out.append({"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}.get(body[index], body[index]))
        else:
            out.append(char)
        index += 1
    return "".join(out)


PO_HEADER = """# {name} translation for ShiroRVC.
# Translate the msgstr lines; leave an empty msgstr to fall back to English.
#
msgid ""
msgstr ""
"Project-Id-Version: ShiroRVC\\n"
"Language: {language}\\n"
"MIME-Version: 1.0\\n"
"Content-Type: text/plain; charset=UTF-8\\n"
"Content-Transfer-Encoding: 8bit\\n"
"Plural-Forms: {plural_forms}\\n"
"""

PLURAL_FORMS = {
    "pt_BR": "nplurals=2; plural=(n > 1);",
    "pt": "nplurals=2; plural=(n > 1);",
}
DEFAULT_PLURAL_FORMS = "nplurals=2; plural=(n != 1);"


def update(languages: list[str]) -> None:
    """Regenerate each ``.po`` from the template, keeping existing translations."""
    if not POT_PATH.is_file():
        extract()
    template = parse_po(POT_PATH)
    template.pop("", None)

    for language in languages:
        po_path = LOCALE_DIR / language / "LC_MESSAGES" / f"{DOMAIN}.po"
        existing = parse_po(po_path)
        po_path.parent.mkdir(parents=True, exist_ok=True)

        kept = new = 0
        lines = [
            PO_HEADER.format(
                name=language,
                language=language,
                plural_forms=PLURAL_FORMS.get(language, DEFAULT_PLURAL_FORMS),
            )
        ]
        for msgid in sorted(template):
            entry = existing.get(msgid)
            translated = entry["msgstr"] if entry else ""
            plurals = entry["plurals"] if entry else []
            if translated or plurals:
                kept += 1
            else:
                new += 1
            lines.append(f"msgid {_quote(msgid)}")
            if template[msgid].get("plural"):
                lines.append(f"msgid_plural {_quote(str(template[msgid]['plural']))}")
                forms = list(plurals) + ["", ""]
                lines.append(f"msgstr[0] {_quote(forms[0])}")
                lines.append(f"msgstr[1] {_quote(forms[1])}")
            else:
                lines.append(f"msgstr {_quote(translated)}")
            lines.append("")

        po_path.write_text("\n".join(lines), encoding="utf-8")
        print(
            f"update: {po_path.relative_to(ROOT)} -- "
            f"{kept} translated, {new} untranslated, {len(template)} total"
        )


# -- .mo writing ------------------------------------------------------------


def _write_mo(entries: dict[str, dict], path: Path, language: str) -> int:
    """Write the binary catalog gettext actually reads.

    Format: magic, two counts, two offset tables, then the string blobs.  Only
    translated entries go in -- an empty msgstr in the file has to mean "fall
    back", and a zero-length translation in the ``.mo`` would render as an
    empty label instead.
    """
    items: list[tuple[bytes, bytes]] = []
    header = (
        f"Project-Id-Version: ShiroRVC\n"
        f"Language: {language}\n"
        "MIME-Version: 1.0\n"
        "Content-Type: text/plain; charset=UTF-8\n"
        "Content-Transfer-Encoding: 8bit\n"
        f"Plural-Forms: {PLURAL_FORMS.get(language, DEFAULT_PLURAL_FORMS)}\n"
    )
    items.append((b"", header.encode("utf-8")))

    for msgid, entry in entries.items():
        if not msgid or entry.get("fuzzy"):
            continue
        if entry.get("plural"):
            forms = [form for form in entry.get("plurals", []) if form]
            if len(forms) < 2:
                continue
            key = msgid + "\x00" + str(entry["plural"])
            value = "\x00".join(entry["plurals"])
        else:
            if not entry.get("msgstr"):
                continue
            key, value = msgid, str(entry["msgstr"])
        items.append((key.encode("utf-8"), value.encode("utf-8")))

    items.sort(key=lambda pair: pair[0])
    count = len(items)
    keys_start = 7 * 4 + 16 * count  # header + both offset tables
    offsets: list[tuple[int, int, int, int]] = []
    keys = bytearray()
    values = bytearray()
    for key, value in items:
        offsets.append((len(key), keys_start + len(keys), len(value), 0))
        keys += key + b"\x00"
    values_start = keys_start + len(keys)
    packed: list[tuple[int, int, int, int]] = []
    for (key_length, key_offset, value_length, _unused), (_key, value) in zip(offsets, items):
        packed.append((key_length, key_offset, value_length, values_start + len(values)))
        values += value + b"\x00"

    blob = struct.pack(
        "<Iiiiiii",
        0x950412DE,  # magic, little-endian
        0,           # revision
        count,
        7 * 4,       # offset of the original-string table
        7 * 4 + 8 * count,  # offset of the translation table
        0,           # hash table size: unused, gettext copes
        0,           # hash table offset
    )
    blob += b"".join(struct.pack("<ii", length, offset) for length, offset, _l, _o in packed)
    blob += b"".join(struct.pack("<ii", length, offset) for _l, _o, length, offset in packed)
    blob += bytes(keys) + bytes(values)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return count - 1  # the header entry is not a message


def compile_catalogs(languages: list[str] | None = None) -> int:
    """Compile every ``.po`` under ``locales/`` into its ``.mo``."""
    total = 0
    for po_path in sorted(LOCALE_DIR.glob(f"*/LC_MESSAGES/{DOMAIN}.po")):
        language = po_path.parents[1].name
        if languages and language not in languages:
            continue
        entries = parse_po(po_path)
        entries.pop("", None)
        written = _write_mo(entries, po_path.with_suffix(".mo"), language)
        total += written
        print(f"compile: {language} -- {written} translated messages")
    if total == 0:
        print("compile: nothing compiled; run extract/update first", file=sys.stderr)
    return total


def stats() -> None:
    template = parse_po(POT_PATH)
    template.pop("", None)
    for po_path in sorted(LOCALE_DIR.glob(f"*/LC_MESSAGES/{DOMAIN}.po")):
        entries = parse_po(po_path)
        entries.pop("", None)
        done = sum(1 for entry in entries.values() if entry["msgstr"] or entry["plurals"])
        # A plural entry carries its translations in ``plurals``, not
        # ``msgstr``; checking only the latter reports every plural as missing.
        missing = [
            msgid
            for msgid in template
            if not (
                entries.get(msgid, {}).get("msgstr")
                or any(entries.get(msgid, {}).get("plurals") or [])
            )
        ]
        percent = 100 * done / len(template) if template else 100
        print(f"{po_path.parents[1].name}: {done}/{len(template)} ({percent:.0f}%)")
        for msgid in missing[:20]:
            print(f"    untranslated: {msgid[:70]!r}")
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("extract", help="scan the sources into locales/shiromiya.pot")
    update_parser = sub.add_parser("update", help="merge the template into the .po files")
    update_parser.add_argument("--language", action="append", dest="languages")
    compile_parser = sub.add_parser("compile", help="build the .mo files")
    compile_parser.add_argument("--language", action="append", dest="languages")
    sub.add_parser("stats", help="report translation coverage")
    sub.add_parser("all", help="extract, update and compile in one go")

    args = parser.parse_args()
    known = [name for name in _shipped_languages() if name != "en"]

    if args.command == "extract":
        extract()
    elif args.command == "update":
        update(args.languages or known)
    elif args.command == "compile":
        compile_catalogs(args.languages)
    elif args.command == "stats":
        stats()
    elif args.command == "all":
        extract()
        update(known)
        compile_catalogs(None)
    return 0


def _shipped_languages() -> list[str]:
    """Languages from the catalog module, so the two cannot drift apart."""
    sys.path.insert(0, str(ROOT))
    from rvc.lib.i18n import LANGUAGES

    return list(LANGUAGES)


if __name__ == "__main__":
    raise SystemExit(main())
