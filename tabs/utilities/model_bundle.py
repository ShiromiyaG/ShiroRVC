from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from rvc.lib.model_bundle import (
    MODEL_BUNDLE_EXTENSION,
    create_model_bundle as build_model_bundle,
    extract_model_bundle,
    extraction_report,
    resolve_bundle_path,
    walk_models,
)

from rvc.lib.i18n import _


def _upload_path(uploaded_file: Any) -> Path:
    """Return a filesystem path for a Gradio upload or a plain path value."""
    return Path(getattr(uploaded_file, "name", uploaded_file)).expanduser()


def _uploaded_paths(uploaded_files: list[Any] | None) -> list[Path]:
    return [_upload_path(uploaded_file) for uploaded_file in uploaded_files or []]


def _project_logs_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "logs"


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


def list_bundles() -> list[str]:
    """``.srvc`` files under logs/, relative to the project root."""
    logs_dir = _project_logs_dir()
    return sorted(
        str((Path(root) / name).relative_to(logs_dir.parent))
        for root, _dirs, files in walk_models(logs_dir)
        for name in files
        if name.lower().endswith(MODEL_BUNDLE_EXTENSION)
    )


def run_extract_model_bundle_script(bundle_path, output_path, overwrite):
    """Unpack a bundle into Applio's ``<name>/<name>.pth`` + ``.index`` layout."""
    if not bundle_path or not str(bundle_path).strip():
        return "Error: choose a bundle."
    bundle = Path(str(bundle_path).strip()).expanduser()
    if not bundle.is_absolute():
        bundle = _project_logs_dir().parent / bundle
    if output_path and output_path.strip():
        target = Path(output_path.strip()).expanduser()
    else:
        target = _project_logs_dir() / f"{bundle.stem}_extracted"

    try:
        reports = extract_model_bundle(bundle, target, bool(overwrite))
    except Exception as error:
        return f"Error extracting model bundle: {error}"

    lines = extraction_report(reports)
    return f"Success!\nSaved to: {target}\n\nDetails:\n" + "\n".join(lines)


def model_bundle_tab():
    with gr.Column():
        gr.Markdown(
            _(
                "# Model Bundle Maker\n"
                "Combine one or more RVC `.pth` models into a compressed "
                "`{extension}` bundle.\n\n"
                "- Speaker names come from the `.pth` filenames.\n"
                "- Each `.index` goes to the model it is named after: the same "
                "name, else the model's name without `_<N>e_<N>s` -- "
                "`voz_10e_850s.pth` takes `voz.index`, or a lone `voz_spk0.index`.\n"
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

        gr.Markdown(
            _(
                "# Extract a bundle\n"
                "Writes each model back out as `<name>/<name>.pth` and "
                "`<name>.index`, the layout Applio pairs by name. Copy those "
                "folders into Applio's `logs/`."
            )
        )
        with gr.Row():
            bundle_select = gr.Dropdown(
                label=_("Bundle"),
                info=_("A .srvc under logs/, or type a path."),
                choices=list_bundles(),
                allow_custom_value=True,
                interactive=True,
            )
            refresh_bundles_button = gr.Button(_("Refresh"))
        extract_path_input = gr.Textbox(
            label=_("Output folder"),
            info=_("Empty extracts to logs/<bundle>_extracted."),
            interactive=True,
        )
        overwrite_checkbox = gr.Checkbox(
            label=_("Overwrite existing files"),
            value=False,
        )
        extract_output_info = gr.Textbox(
            label=_("Output information"),
            value="",
            max_lines=12,
            interactive=False,
        )
        extract_button = gr.Button(_("Extract bundle"), variant="primary")

        refresh_bundles_button.click(
            fn=lambda: gr.update(choices=list_bundles()),
            inputs=[],
            outputs=[bundle_select],
        )
        extract_button.click(
            fn=run_extract_model_bundle_script,
            inputs=[bundle_select, extract_path_input, overwrite_checkbox],
            outputs=[extract_output_info],
        )


def create_model_bundle(
    pth_paths: list[Path],
    index_paths: list[Path] | None = None,
    output_path: str | None = None,
    use_single_index: bool = False,
    comp_level: int = 3,
) -> str:
    """Build and save a bundle; returns the message the tab shows."""
    try:
        final_path = resolve_bundle_path(output_path, _project_logs_dir(), list(pth_paths))
        report = build_model_bundle(
            pth_paths, index_paths, final_path, bool(use_single_index), int(comp_level)
        )
    except Exception as error:
        return f"Error creating model bundle: {error}"
    details = "\n".join(report)
    return f"Success!\nSaved to: {final_path}\n\nDetails:\n{details}"


if __name__ == "__main__":
    with gr.Blocks() as demo:
        model_bundle_tab()
    demo.launch()
