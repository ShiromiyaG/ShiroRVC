"""ShiroRVC model bundles: several ``.pth`` models and their indexes in one file.

A bundle is a zstd frame holding a ``torch.save`` of
``{"format", "version", "models": {name: {"model_state", "index_data", "index_meta"}}}``,
followed by a zstd *skippable* frame with a small JSON manifest -- names,
speakers, vocoder -- so a bundle can be listed without decompressing it.
Every zstd reader steps over a skippable frame, so builds that predate the
manifest still open these files.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import re
import shutil
import struct
from os import PathLike
from pathlib import Path
from typing import Any

import zstandard as zstd


MODEL_BUNDLE_EXTENSION = ".srvc"
MODEL_BUNDLE_EXTENSIONS = (MODEL_BUNDLE_EXTENSION,)
MODEL_FILE_EXTENSIONS = (".pth", MODEL_BUNDLE_EXTENSION)
MODEL_BUNDLE_FORMAT = "shiromiya-rvc-model-bundle"
#: 2 added the manifest and stores each index file verbatim, footer included.
#: The payload is otherwise laid out as in 1, which older builds still read.
MODEL_BUNDLE_VERSION = 2

#: The manifest's frame: a zstd skippable frame whose data ends in
#: ``<json length, u32 LE><magic>``, so it can be found from the end.
_SKIPPABLE_FRAME = struct.Struct("<II")
_SKIPPABLE_MAGIC = 0x184D2A5E
_MANIFEST_TAIL = struct.Struct("<I4s")
_MANIFEST_MAGIC = b"SRVM"


def is_model_bundle(path: str | PathLike[str] | None) -> bool:
    """Return whether a path uses the model bundle extension."""
    if not path:
        return False
    return Path(path).suffix.lower() in MODEL_BUNDLE_EXTENSIONS


def is_model_file(path: str | PathLike[str] | None) -> bool:
    """Return whether a path is a regular checkpoint or a model bundle."""
    if not path:
        return False
    return Path(path).suffix.lower() in MODEL_FILE_EXTENSIONS


# Directories that live inside ``logs/<model>/`` but never contain a model:
# preprocessing output, extracted features and TensorBoard runs.  A single
# trained model can leave hundreds of thousands of files in these, which is
# enough to make a naive recursive scan of ``logs/`` take seconds.
TRAINING_ARTIFACT_DIRS = frozenset(
    {
        "sliced_audios",
        "sliced_audios_16k",
        "extracted",
        "f0",
        "f0_voiced",
        "eval",
        "validation_samples",
        "zips",
        "__pycache__",
        ".torchinductor",
    }
)


def walk_models(root: str | PathLike[str], skip=TRAINING_ARTIFACT_DIRS):
    """``os.walk`` over a model tree with training-artifact directories pruned.

    Yields the same ``(dirpath, dirnames, filenames)`` triples as ``os.walk``,
    so callers only have to swap the call.  Arbitrary nesting of real model
    folders still works; only the known artifact directories are skipped.

    Note this walks top-down (pruning is impossible bottom-up), so the yield
    order is shallowest-first rather than deepest-first.
    """
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [name for name in dirnames if name not in skip]
        yield dirpath, dirnames, filenames


def _safe_globals() -> list:
    """What a bundle may hold beyond tensors and plain containers: numpy arrays.

    The index is stored as one, and ``weights_only`` refuses them by default.
    Allowing exactly these keeps a ``.srvc`` from running code when opened --
    an unrestricted unpickle would, and bundles are files people download.
    Both module paths, since numpy 1 and numpy 2 pickle the helper under
    different names.
    """
    import numpy as np

    try:
        from numpy._core.multiarray import _reconstruct
    except ImportError:  # numpy < 2
        from numpy.core.multiarray import _reconstruct

    return [
        (_reconstruct, "numpy._core.multiarray._reconstruct"),
        (_reconstruct, "numpy.core.multiarray._reconstruct"),
        np.ndarray,
        np.dtype,
        # Restoring the array's dtype builds an instance of this class, which
        # ``get_unsafe_globals_in_checkpoint`` does not list.
        type(np.dtype(np.uint8)),
    ]


def load_model_bundle(path: str | PathLike[str]) -> dict[str, Any]:
    """Load and validate a bundle, refusing anything but tensors, plain
    containers and numpy arrays."""
    import torch

    bundle_path = Path(path)
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Model bundle not found: {bundle_path}")

    with bundle_path.open("rb") as compressed_file:
        decompressor = zstd.ZstdDecompressor()
        with decompressor.stream_reader(compressed_file) as reader:
            payload = reader.read()

    try:
        with torch.serialization.safe_globals(_safe_globals()):
            bundle_data = torch.load(
                io.BytesIO(payload), map_location="cpu", weights_only=True
            )
    except pickle.UnpicklingError as error:
        raise ValueError(
            f"{bundle_path.name} holds objects a model bundle never contains, "
            f"so it was not loaded: {error}"
        ) from error
    del payload

    if not isinstance(bundle_data, dict):
        raise ValueError("The model bundle must contain a dictionary.")
    _validate_bundle_data(bundle_data)
    return bundle_data


def default_model_name(names) -> str | None:
    """The model a bundle opens on when none is chosen: the first by name,
    which is also what the model pickers show first."""
    names = sorted(names)
    return names[0] if names else None


def _validate_bundle_data(bundle_data: dict[str, Any]) -> None:
    bundle_format = bundle_data.get("format")
    if bundle_format is not None and bundle_format != MODEL_BUNDLE_FORMAT:
        raise ValueError(f"Not a ShiroRVC model bundle (format {bundle_format!r}).")
    version = bundle_data.get("version")
    if version is not None and int(version) > MODEL_BUNDLE_VERSION:
        raise ValueError(
            f"This bundle is format version {version}; this build reads up to "
            f"{MODEL_BUNDLE_VERSION}. Update ShiroRVC to open it."
        )

    models = bundle_data.get("models")
    if "models" in bundle_data:
        if not isinstance(models, dict) or not models:
            raise ValueError("The model bundle does not contain any models.")
        for model_name, model_entry in models.items():
            if not isinstance(model_name, str) or not model_name.strip():
                raise ValueError("The model bundle contains an invalid model name.")
            if not isinstance(model_entry, dict) or not isinstance(
                model_entry.get("model_state"), dict
            ):
                raise ValueError(f"Model entry '{model_name}' is invalid.")
        return

    if isinstance(bundle_data.get("model_state"), dict) or "config" in bundle_data:
        return

    raise ValueError("The file is not a recognized RVC model bundle.")


def save_model_bundle(
    path: str | PathLike[str],
    bundle_data: dict[str, Any],
    compression_level: int = 3,
) -> Path:
    """Write a bundle atomically using Zstandard compression."""
    import torch

    if not isinstance(bundle_data, dict):
        raise TypeError("The model bundle must be a dictionary.")
    _validate_bundle_data(bundle_data)

    compression_level = int(compression_level)
    if not 1 <= compression_level <= 22:
        raise ValueError("Compression level must be between 1 and 22.")

    bundle_path = Path(path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = bundle_path.with_name(f".{bundle_path.name}.tmp")

    try:
        compressor = zstd.ZstdCompressor(level=compression_level)
        with temporary_path.open("wb") as compressed_file:
            with compressor.stream_writer(compressed_file, closefd=False) as writer:
                torch.save(bundle_data, writer)
            compressed_file.write(_manifest_frame(_manifest(bundle_data)))
        os.replace(temporary_path, bundle_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return bundle_path


def _model_info(state: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "speakers_id": state.get("speakers_id"),
        "vocoder": state.get("vocoder"),
        "sample_rate": state.get("sr"),
        "has_index": entry.get("index_data") is not None,
    }


def _manifest(bundle_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": MODEL_BUNDLE_FORMAT,
        "version": MODEL_BUNDLE_VERSION,
        "models": {
            name: _model_info(entry.get("model_state") or {}, entry)
            for name, entry in get_bundle_models(bundle_data).items()
        },
    }


def _manifest_frame(manifest: dict[str, Any]) -> bytes:
    body = json.dumps(manifest, default=str).encode("utf-8")
    data = body + _MANIFEST_TAIL.pack(len(body), _MANIFEST_MAGIC)
    return _SKIPPABLE_FRAME.pack(_SKIPPABLE_MAGIC, len(data)) + data


def read_bundle_manifest(path: str | PathLike[str]) -> dict[str, Any] | None:
    """The manifest from the end of a bundle, without decompressing anything.

    ``None`` for a bundle written before the manifest existed, one from a newer
    format, or one that cannot be read -- callers then load the bundle.
    """
    try:
        with Path(path).open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            if size < _SKIPPABLE_FRAME.size + _MANIFEST_TAIL.size:
                return None
            handle.seek(size - _MANIFEST_TAIL.size)
            length, magic = _MANIFEST_TAIL.unpack(handle.read(_MANIFEST_TAIL.size))
            start = size - _MANIFEST_TAIL.size - length - _SKIPPABLE_FRAME.size
            if magic != _MANIFEST_MAGIC or start < 0:
                return None
            handle.seek(start)
            frame_magic, frame_size = _SKIPPABLE_FRAME.unpack(
                handle.read(_SKIPPABLE_FRAME.size)
            )
            if frame_magic != _SKIPPABLE_MAGIC or frame_size != length + _MANIFEST_TAIL.size:
                return None
            manifest = json.loads(handle.read(length))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict) or not isinstance(manifest.get("models"), dict):
        return None
    if int(manifest.get("version", 0)) > MODEL_BUNDLE_VERSION:
        return None
    return manifest


def bundle_model_names(path: str | PathLike[str]) -> list[str]:
    """Model names in a bundle, sorted; from the manifest when there is one."""
    manifest = read_bundle_manifest(path)
    if manifest is not None:
        return sorted(manifest["models"])
    return sorted(get_bundle_models(load_model_bundle(path)))


def bundle_model_info(
    path: str | PathLike[str],
    model_name: str | None = None,
) -> dict[str, Any]:
    """``speakers_id``, ``vocoder``, ``sample_rate`` and ``has_index`` of one model.

    From the manifest when there is one.  ``model_name`` defaults to the first
    model; an unknown name gives ``{}``.
    """
    manifest = read_bundle_manifest(path)
    if manifest is None:
        bundle_data = load_model_bundle(path)
        if not get_bundle_models(bundle_data):  # the older single-model layout
            state = get_bundle_model_state(bundle_data) or {}
            return _model_info(state, bundle_data)
        manifest = _manifest(bundle_data)
    models = manifest["models"]
    return models.get(model_name or default_model_name(models), {})


def speaker_ids(path: str | PathLike[str], model_name: str | None = None) -> list[int]:
    """Speaker ids a ``.pth``, or one model of a bundle, can be asked for.

    ``[0]`` when the checkpoint records no speaker count.  A bundle is read
    from its manifest when it has one; a ``.pth`` is loaded weights-only.
    """
    if is_model_bundle(path):
        count = bundle_model_info(path, model_name).get("speakers_id")
    else:
        import torch

        state = torch.load(str(path), map_location="cpu", weights_only=True)
        count = state.get("speakers_id") if isinstance(state, dict) else None
    return list(range(count)) if count else [0]


def get_bundle_models(bundle_data: dict[str, Any]) -> dict[str, Any]:
    """Return the multi-model mapping, or an empty mapping for single bundles."""
    models = bundle_data.get("models")
    return models if isinstance(models, dict) else {}


def get_bundle_model_state(
    bundle_data: dict[str, Any],
    model_name: str | None = None,
) -> dict[str, Any] | None:
    """Return one checkpoint state from a multi- or single-model bundle."""
    models = get_bundle_models(bundle_data)
    if models:
        if not model_name or model_name not in models:
            return None
        model_entry = models[model_name]
        state = model_entry.get("model_state") if isinstance(model_entry, dict) else None
        return state if isinstance(state, dict) else None

    state = bundle_data.get("model_state")
    if isinstance(state, dict):
        return state
    return bundle_data if "config" in bundle_data else None


#: Vocoder labels Applio's synthesizer builds; it builds any other as HiFi-GAN.
APPLIO_VOCODERS = frozenset({"HiFi-GAN", "MRF HiFi-GAN", "RefineGAN", "RefineGAN2"})


def _link_or_copy(source: Path, target: Path) -> None:
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def extract_model_bundle(
    path: str | PathLike[str],
    output_dir: str | PathLike[str],
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Write each model in a bundle back out as ``<name>/<name>.pth`` and ``.index``.

    That is the layout Applio pairs by name.  The ``.pth`` is the checkpoint
    that went into the bundle, and the index carries this fork's metadata as a
    footer, which Applio ignores.  An index shared by every model is written
    once and hard-linked into the other folders.  Returns one report per model:
    ``name``, ``pth``, ``index`` (or ``None``), ``vocoder`` and ``metric``.
    """
    import faiss
    import numpy as np
    import torch

    from rvc.lib import index_meta

    bundle_data = load_model_bundle(path)
    entries = get_bundle_models(bundle_data) or {
        # The older single-model format keeps the index beside the state.
        Path(path).stem: {
            "model_state": get_bundle_model_state(bundle_data),
            "index_data": bundle_data.get("index_data"),
            "index_meta": bundle_data.get("index_meta"),
        }
    }

    output_dir = Path(output_dir)
    if not overwrite:
        clashes = [
            str(output_dir / name / f"{name}{suffix}")
            for name in entries
            for suffix in (".pth", ".index")
            if (output_dir / name / f"{name}{suffix}").exists()
        ]
        if clashes:
            raise FileExistsError("Already exists: " + ", ".join(clashes))

    # ``torch.load`` keeps a shared object shared, so a single-index bundle
    # hands every model the same array.
    written: dict[int, tuple[Path, str]] = {}
    reports = []
    for name, entry in entries.items():
        folder = output_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        state = entry["model_state"]
        pth_path = folder / f"{name}.pth"
        torch.save(state, pth_path)
        report = {
            "name": name,
            "pth": pth_path,
            "index": None,
            "vocoder": str(state.get("vocoder", "HiFi-GAN")),
            "metric": None,
        }

        index_data = entry.get("index_data")
        if index_data is not None:
            index_path = folder / f"{name}.index"
            if id(index_data) in written:
                source, metric = written[id(index_data)]
                _link_or_copy(source, index_path)
            else:
                # Version 1 bundles hold FAISS's serialisation, version 2 the
                # file itself; both are a valid .index as they stand.
                np.asarray(index_data, dtype=np.uint8).tofile(index_path)
                payload = entry.get("index_meta")
                meta = index_meta.from_bytes(bytes(payload)) if payload is not None else None
                if meta is not None:
                    index_meta.write(index_path, meta)
                else:
                    meta = index_meta.read(index_path)
                metric = (
                    meta.metric
                    if meta is not None
                    else index_meta.legacy_meta(faiss.read_index(str(index_path))).metric
                )
                written[id(index_data)] = (index_path, metric)
            report["index"] = index_path
            report["metric"] = metric
        reports.append(report)
    return reports


def extraction_report(reports: list[dict[str, Any]]) -> list[str]:
    """One line per extracted model, plus a note where Applio will not take it as-is."""
    lines = []
    for report in reports:
        index = report["index"].name if report["index"] else "no index"
        lines.append(f"{report['name']}: {report['pth'].name} + {index}")
        if report["vocoder"] not in APPLIO_VOCODERS:
            lines.append(
                f"  Applio does not know the vocoder '{report['vocoder']}' and "
                "builds HiFi-GAN in its place, so the model only loads there if "
                "its generator is HiFi-GAN."
            )
        if report["metric"] == "cosine":
            lines.append(
                "  Cosine index: Applio loads it but weights and scales its "
                "matches as if it were L2. Rebuild it as L2 for use there."
            )
    return lines


def resolve_bundle_path(
    output_path: str | None,
    logs_dir: str | PathLike[str],
    pth_paths: list[Path],
) -> Path:
    """Where a new bundle goes: ``output_path``, or ``logs/<first model>.srvc``.

    A bare file name lands in ``logs_dir``; a folder gets the default name.
    """
    logs_dir = Path(logs_dir)
    default_stem = Path(pth_paths[0]).stem if pth_paths else "bundle"
    default_name = f"{default_stem}_multi" if len(pth_paths) > 1 else default_stem
    default_filename = f"{default_name}{MODEL_BUNDLE_EXTENSION}"

    if not output_path or not str(output_path).strip():
        return logs_dir / default_filename

    candidate = Path(str(output_path).strip()).expanduser()
    if not candidate.is_absolute() and candidate.parent == Path("."):
        candidate = logs_dir / candidate

    is_directory = candidate.is_dir() or (
        not candidate.suffix and candidate.parent != Path(".")
    )
    if is_directory:
        return candidate / default_filename

    return candidate.with_suffix(MODEL_BUNDLE_EXTENSION)


def _serialize_index(index_path: Path) -> Any:
    """The index file's bytes as they are, footer included.

    ``faiss.deserialize_index`` stops where the index ends, as ``read_index``
    does, so the footer rides along without FAISS having to load and
    re-serialise the index here.  It is still opened once, mmapped where the
    index type allows, only to refuse a file that is not an index at all.
    """
    import faiss
    import numpy as np

    if index_path.suffix.lower() != ".index":
        raise ValueError(f"Not an index file: {index_path.name}")
    if not index_path.is_file():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    try:
        try:
            faiss.read_index(str(index_path), faiss.IO_FLAG_MMAP)
        except RuntimeError:
            faiss.read_index(str(index_path))
    except RuntimeError as error:
        raise ValueError(f"{index_path.name} is not a readable FAISS index: {error}") from error
    return np.fromfile(index_path, dtype=np.uint8)


def _serialize_index_meta(index_path: Path) -> Any:
    """The index's metadata, if it has any.

    Redundant with the footer :func:`_serialize_index` now carries, and kept
    for builds that predate it: they read the metadata from here only.
    """
    from rvc.lib import index_meta

    meta = index_meta.read(str(index_path))
    return index_meta.to_bytes(meta) if meta is not None else None


#: The suffix training gives exported weights; see ``small_model_naming``.
_EPOCH_SUFFIX = re.compile(r"_\d+e_\d+s$", re.IGNORECASE)
#: The suffix ``extract_index`` gives a one-speaker index.
_SPEAKER_SUFFIX = re.compile(r"_spk\d+$", re.IGNORECASE)


def _pair_indexes(
    pth_paths: list[Path],
    index_paths: list[Path],
) -> tuple[dict[Path, Path], list[str]]:
    """Pair each model with an index by file name; also say what did not pair.

    In order of preference: the same name; the model's name without the
    ``_<N>e_<N>s`` training suffix (``voz_10e_850s.pth`` takes ``voz.index``);
    that name plus ``_spk<N>``, when exactly one such index was given.  Two
    equally good candidates attach neither rather than a guess.
    """
    by_stem = {path.stem.casefold(): path for path in index_paths}
    pairs: dict[Path, Path] = {}
    notes: list[str] = []
    explained: set[Path] = set()
    for pth_path in pth_paths:
        stem = pth_path.stem.casefold()
        base = _EPOCH_SUFFIX.sub("", stem)
        chosen = by_stem.get(stem) or by_stem.get(base)
        if chosen is None:
            speakers = sorted(
                path
                for key, path in by_stem.items()
                if key != base and _SPEAKER_SUFFIX.sub("", key) == base
            )
            if len(speakers) == 1:
                chosen = speakers[0]
            elif speakers:
                explained.update(speakers)
                notes.append(
                    f"{pth_path.stem}: several indexes fit "
                    f"({', '.join(path.name for path in speakers)}), so none was "
                    f"attached. Rename the one to use to {pth_path.stem}.index."
                )
        if chosen is not None:
            pairs[pth_path] = chosen

    used = set(pairs.values())
    for index_path in index_paths:
        if index_path not in used and index_path not in explained:
            notes.append(
                f"{index_path.name}: not attached to any model -- its name fits "
                "none of the .pth files. Name it after one, or use Single index."
            )
    return pairs, notes


def create_model_bundle(
    pth_paths: list[str | PathLike[str]],
    index_paths: list[str | PathLike[str]] | None,
    output_path: str | PathLike[str],
    use_single_index: bool = False,
    compression_level: int = 3,
) -> list[str]:
    """Bundle ``.pth`` models, each with the index its file name pairs it with.

    See :func:`_pair_indexes` for the pairing; ``use_single_index`` attaches
    the one given index to every model instead.  Model names come from the
    ``.pth`` file names.  Returns one line per model saying what was attached,
    then a line for each index left out; raises ``ValueError`` or
    ``FileNotFoundError`` on bad input before anything is written.
    """
    import torch

    pth_paths = [Path(path).expanduser() for path in pth_paths]
    index_paths = [Path(path).expanduser() for path in index_paths or []]
    if not pth_paths:
        raise ValueError("At least one .pth file is required.")
    if not 1 <= int(compression_level) <= 22:
        raise ValueError("Compression level must be between 1 and 22.")

    names: set[str] = set()
    for pth_path in pth_paths:
        if pth_path.suffix.lower() != ".pth":
            raise ValueError(f"Expected a .pth file, got {pth_path.name}.")
        if not pth_path.is_file():
            raise FileNotFoundError(f"PTH file not found: {pth_path}")
        if pth_path.stem.casefold() in names:
            raise ValueError(f"Duplicate model name: {pth_path.stem}")
        names.add(pth_path.stem.casefold())

    stems: set[str] = set()
    for index_path in index_paths:
        if index_path.stem.casefold() in stems:
            raise ValueError(f"Multiple index files share the name {index_path.stem}")
        stems.add(index_path.stem.casefold())

    notes: list[str] = []
    if use_single_index:
        if len(index_paths) != 1:
            raise ValueError("Single index requires exactly one .index file.")
        pairs = {pth_path: index_paths[0] for pth_path in pth_paths}
    else:
        pairs, notes = _pair_indexes(pth_paths, index_paths)

    # One array per index file, however many models it serves: the pickle
    # then stores it once, where a copy per model would store it again each time.
    serialized: dict[Path, tuple[Any, Any]] = {}
    models: dict[str, dict[str, Any]] = {}
    report: list[str] = []
    for pth_path in sorted(pth_paths, key=lambda path: path.name.casefold()):
        entry: dict[str, Any] = {
            "model_state": torch.load(str(pth_path), map_location="cpu", weights_only=True)
        }
        index_path = pairs.get(pth_path)
        if index_path is not None:
            if index_path not in serialized:
                serialized[index_path] = (
                    _serialize_index(index_path),
                    _serialize_index_meta(index_path),
                )
            index_data, meta = serialized[index_path]
            entry["index_data"] = index_data
            if meta is not None:
                entry["index_meta"] = meta
            report.append(f"{pth_path.stem}: {index_path.name} attached")
        else:
            report.append(f"{pth_path.stem}: weights only")
        models[pth_path.stem] = entry
    report.extend(notes)

    save_model_bundle(
        output_path,
        {"format": MODEL_BUNDLE_FORMAT, "version": MODEL_BUNDLE_VERSION, "models": models},
        int(compression_level),
    )
    return report
