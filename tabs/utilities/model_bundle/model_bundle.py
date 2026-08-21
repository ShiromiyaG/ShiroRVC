from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from rvc.lib.model_bundle import (
    MODEL_BUNDLE_EXTENSION,
    MODEL_BUNDLE_FORMAT,
    MODEL_BUNDLE_VERSION,
    save_model_bundle,
)

from rvc.lib.i18n import _


def _upload_path(uploaded_file: Any) -> Path:
    """Return a filesystem path for a Gradio upload or a plain path value."""
    return Path(getattr(uploaded_file, "name", uploaded_file)).expanduser()


def _uploaded_paths(uploaded_files: list[Any] | None) -> list[Path]:
    return [_upload_path(uploaded_file) for uploaded_file in uploaded_files or []]


def _project_logs_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "logs"


def _resolve_output_path(
    output_path: str | None,
    logs_dir: Path,
    default_stem: str,
    model_count: int,
) -> Path:
    default_name = f"{default_stem}_multi" if model_count > 1 else default_stem
    default_filename = f"{default_name}{MODEL_BUNDLE_EXTENSION}"

    if not output_path or not output_path.strip():
        return logs_dir / default_filename

    candidate = Path(output_path.strip()).expanduser()
    if not candidate.is_absolute() and candidate.parent == Path("."):
        candidate = logs_dir / candidate

    is_directory = candidate.is_dir() or (
        not candidate.suffix and candidate.parent != Path(".")
    )
    if is_directory:
        return candidate / default_filename

    return candidate.with_suffix(MODEL_BUNDLE_EXTENSION)


def _serialize_index(index_path: Path) -> Any:
    import faiss

    if index_path.suffix.lower() != ".index":
        raise ValueError(f"Not an index file: {index_path.name}")
    if not index_path.is_file():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    return faiss.serialize_index(faiss.read_index(str(index_path)))


def _serialize_index_meta(index_path: Path) -> Any:
    """The index's sidecar, if it has one.

    A bundle is a single file, so a sidecar that lives next to the ``.index`` on
    disk has nowhere to go unless it is carried inside.  Without this the
    retrieval's temporal continuity would work from a folder and quietly stop
    working from a bundle of the same model.
    """
    from rvc.lib import index_meta

    meta = index_meta.read(str(index_path))
    return index_meta.to_bytes(meta) if meta is not None else None


def _matching_indexes(index_paths: list[Path]) -> dict[str, Path]:
    matches: dict[str, Path] = {}
    for index_path in index_paths:
        key = index_path.stem.casefold()
        if key in matches:
            raise ValueError(
                f"Multiple index files use the same model name: {index_path.stem}"
            )
        matches[key] = index_path
    return matches


def run_create_model_bundle_script(
    pth_files,
    index_files,
    output_path,
    use_single_index,
    comp_level,
):
    """Create a ShiroRVC model bundle from uploaded model files."""
    pth_paths = _uploaded_paths(pth_files)
    if not pth_paths:
        return "Error: upload at least one .pth file."

    try:
        return create_model_bundle(
            pth_paths,
            _uploaded_paths(index_files),
            output_path,
            bool(use_single_index),
            int(comp_level),
        )
    except Exception as error:
        return f"Error: {error}"


def model_bundle_tab():
    """Build the model bundle utility interface."""
    with gr.Column():
        gr.Markdown(
            _(
                "# Model Bundle Maker\n"
                "Combine one or more RVC `.pth` models into a compressed "
                "`{extension}` bundle.\n\n"
                "- Speaker names come from the `.pth` filenames.\n"
                "- Matching `.index` files are attached automatically.\n"
                "- Single Index binds one uploaded index to every model.\n"
            ).format(extension=MODEL_BUNDLE_EXTENSION)
        )
        pth_input = gr.File(
            label=_("Upload PTH file(s)"),
            file_types=[".pth"],
            file_count="multiple",
        )
        index_input = gr.File(
            label=_("Upload index file(s) (optional)"),
            file_types=[".index"],
            file_count="multiple",
        )
        with gr.Row():
            use_single_index_checkbox = gr.Checkbox(
                label=_("Single index"),
                info=_("Attach one uploaded index to every model."),
                value=False,
            )
            compression_slider = gr.Slider(
                minimum=1,
                maximum=22,
                step=1,
                value=3,
                label=_("Compression level"),
                info=_("3 balances size and speed."),
            )
        output_path_input = gr.Textbox(
            label=_("Output path"),
            info=_("Empty saves to logs using the first model name."),
            placeholder=f"Example: D:/models/my_bundle{MODEL_BUNDLE_EXTENSION}",
            interactive=True,
        )
        bundle_output_info = gr.Textbox(
            label=_("Output information"),
            value="",
            max_lines=10,
            interactive=False,
        )
        bundle_create_button = gr.Button(_("Create model bundle"), variant="primary")

        bundle_create_button.click(
            fn=run_create_model_bundle_script,
            inputs=[
                pth_input,
                index_input,
                output_path_input,
                use_single_index_checkbox,
                compression_slider,
            ],
            outputs=[bundle_output_info],
        )


def create_model_bundle(
    pth_paths: list[Path],
    index_paths: list[Path] | None = None,
    output_path: str | None = None,
    use_single_index: bool = False,
    comp_level: int = 3,
) -> str:
    """Build and save a validated, compressed multi-model bundle."""
    import torch

    index_paths = index_paths or []
    if not pth_paths:
        return "Error: at least one .pth file is required."
    if not 1 <= int(comp_level) <= 22:
        return "Error: compression level must be between 1 and 22."

    try:
        normalized_pth_paths = [Path(path).expanduser() for path in pth_paths]
        normalized_index_paths = [Path(path).expanduser() for path in index_paths]
        model_names: set[str] = set()
        for pth_path in normalized_pth_paths:
            if pth_path.suffix.lower() != ".pth":
                return f"Error: expected a .pth file, got {pth_path.name}."
            if not pth_path.is_file():
                return f"Error: PTH file not found: {pth_path}"
            model_key = pth_path.stem.casefold()
            if model_key in model_names:
                return f"Error: duplicate model name: {pth_path.stem}"
            model_names.add(model_key)

        matching_indexes = _matching_indexes(normalized_index_paths)
        serialized_single_index = None
        serialized_single_meta = None
        if use_single_index:
            if len(normalized_index_paths) != 1:
                return "Error: Single index requires exactly one .index file."
            serialized_single_index = _serialize_index(normalized_index_paths[0])
            serialized_single_meta = _serialize_index_meta(normalized_index_paths[0])

        models: dict[str, dict[str, Any]] = {}
        log_messages: list[str] = []
        for pth_path in sorted(normalized_pth_paths, key=lambda path: path.name.casefold()):
            model_key = pth_path.stem
            model_entry: dict[str, Any] = {
                "model_state": torch.load(
                    str(pth_path),
                    map_location="cpu",
                    weights_only=True,
                )
            }

            if use_single_index:
                model_entry["index_data"] = serialized_single_index
                if serialized_single_meta is not None:
                    model_entry["index_meta"] = serialized_single_meta
                log_messages.append(f"{model_key}: single index attached")
            else:
                index_path = matching_indexes.get(pth_path.stem.casefold())
                if index_path:
                    model_entry["index_data"] = _serialize_index(index_path)
                    meta_payload = _serialize_index_meta(index_path)
                    if meta_payload is not None:
                        model_entry["index_meta"] = meta_payload
                    log_messages.append(f"{model_key}: {index_path.name} attached")
                else:
                    log_messages.append(f"{model_key}: weights only")

            models[model_key] = model_entry

        bundle_data = {
            "format": MODEL_BUNDLE_FORMAT,
            "version": MODEL_BUNDLE_VERSION,
            "models": models,
        }
        logs_dir = _project_logs_dir()
        final_path = _resolve_output_path(
            output_path,
            logs_dir,
            normalized_pth_paths[0].stem,
            len(normalized_pth_paths),
        )
        save_model_bundle(final_path, bundle_data, int(comp_level))

        details = "\n".join(log_messages)
        return f"Success!\nSaved to: {final_path}\n\nDetails:\n{details}"
    except Exception as error:
        return f"Error creating model bundle: {error}"


if __name__ == "__main__":
    with gr.Blocks() as demo:
        model_bundle_tab()
    demo.launch()
