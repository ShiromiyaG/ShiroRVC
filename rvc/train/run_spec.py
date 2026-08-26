"""The options one training run was launched with.

Training runs in a separate process, so the launcher has to hand these across a
process boundary.  That used to be 32 *positional* argv slots: ``core.py`` built
a list, ``train.py`` read ``sys.argv[N]`` into a global, and nothing checked
that the two agreed.  A misalignment delivered ``batch_size`` where
``sample_rate`` was expected -- both parse as ``int``, so the run simply trained
something other than what was asked for, silently.

This module is the single definition instead.  ``core.py`` builds a
``TrainRunSpec``, writes it to ``logs/<model>/run_spec.json`` and passes that
path; ``train.py`` loads it back.  Adding an option means adding a field here
and a control in the UI -- there is no ordered list left to keep in sync.

Two properties of the launcher constrain the design:

* the trainer re-launches itself per GPU with the ``spawn`` start method, which
  re-executes this module's importer from scratch in every child.  The spec has
  to be re-readable from ``sys.argv``, which a file path is and an in-memory
  object is not.
* ``core._find_trainer_processes`` identifies the trainer by the OS command
  line's ``cmdline[1]`` being the script path.  That listing includes the
  interpreter; ``sys.argv`` inside the process does not.  So the same path is
  ``cmdline[2]`` from outside and ``sys.argv[1]`` from inside.

Writing the spec to the log directory is not incidental: it makes "what was
this trained with?" answerable after the process is gone.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import get_type_hints

#: Bumped when a field is renamed or its meaning changes, so a stale spec fails
#: loudly instead of being read with the wrong semantics.
SPEC_VERSION = 1


@dataclass(frozen=True)
class TrainRunSpec:
    """Everything ``rvc/train/train.py`` needs that is not in the model config.

    The split is deliberate.  Model and data properties that must survive a
    resume (sample rate, architecture, ``lr_decay``, ``rolling_loss_steps``)
    live in ``logs/<model>/config.json``.  What lands here is per-*launch*:
    which GPU, which pretrained weights, how many epochs this time.  Folding
    these into the model config would make a resume silently inherit the
    previous launch's flags.
    """

    # -- identity and data ------------------------------------------------
    model_name: str
    sample_rate: int
    vocoder: str = "hifi"

    # -- schedule ---------------------------------------------------------
    total_epoch_count: int = 300
    epoch_save_frequency: int = 10
    batch_size: int = 8
    gpus: str = "0"

    # -- checkpointing ----------------------------------------------------
    save_only_latest_net_models: bool = False
    save_weight_models: bool = True
    cleanup: bool = False

    # -- starting weights -------------------------------------------------
    # Empty means "from scratch".  ``training_phase`` is derived from these
    # rather than passed: a pretrained source *is* what makes a run a
    # fine-tune, and two fields that must agree are one field too many.
    pretrain_g: str = ""
    pretrain_d: str = ""

    # -- optimisation -----------------------------------------------------
    optimizer_choice: str = "AdamW"
    lr_scheduler: str = "exp decay step"
    use_warmup: bool = False
    warmup_duration: int = 5
    use_custom_lr: bool = False
    custom_lr_g: float = 1e-4
    custom_lr_d: float = 1e-4

    # -- performance ------------------------------------------------------
    use_checkpointing: bool = False
    use_tf32: bool = False
    use_fp16: bool = False
    use_benchmark: bool = True
    compile_vocoder: bool = False
    torch_compile_mode: str = "default"

    # -- monitoring -------------------------------------------------------
    overtrain_detector: bool = False
    stop_on_overtrain: bool = False
    use_ema: bool = True

    @property
    def training_phase(self) -> str:
        """``"finetune"`` when any pretrained source is set, else ``"pretrain"``."""
        sources = (self.pretrain_g, self.pretrain_d)
        has_pretrain = any(str(p).strip() not in ("", "None") for p in sources)
        return "finetune" if has_pretrain else "pretrain"

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"spec_version": SPEC_VERSION, **asdict(self)}
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TrainRunSpec":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        version = raw.pop("spec_version", None)
        if version != SPEC_VERSION:
            raise ValueError(
                f"{path} was written by run spec version {version!r}, but this "
                f"build reads version {SPEC_VERSION}. Relaunch from the UI."
            )
        # ``from __future__ import annotations`` makes ``Field.type`` a string,
        # so the real types have to be resolved rather than read off the field.
        known = get_type_hints(cls)
        unknown = set(raw) - set(known)
        if unknown:
            raise ValueError(f"{path} has unknown fields: {sorted(unknown)}")
        # json gives back str/int/float/bool already; coerce anyway so a
        # hand-edited spec fails here with the field name rather than 2000
        # lines into training.
        coerced = {}
        for name, value in raw.items():
            target = known[name]
            try:
                if target is bool:
                    coerced[name] = value if isinstance(value, bool) else bool(value)
                elif target is int:
                    coerced[name] = int(value)
                elif target is float:
                    coerced[name] = float(value)
                else:
                    coerced[name] = str(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}: field {name!r} cannot be read as {target}: {value!r}"
                ) from exc
        return cls(**coerced)
