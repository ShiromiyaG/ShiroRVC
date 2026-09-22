# Style flow

A flow-matching model, separate from RVC, that generates the fine F0 detail of a
singer (vibrato, scoops, phrase-end drops, attacks) on top of the source's
coarse melody. RVC is unchanged; with the style model off, conversions are
identical to before.

Code: `rvc/lib/style_flow/`. Configs: `rvc/configs/style_flow/`.

Everything below is in the **Style** tab (name, then "1. Style data", then
"2. Training"), or on the command line. Style data comes from a folder of
whole audio files of any length (`<id>_<name>` subfolders are speakers), so
no RVC preprocessing is needed; an already preprocessed and extracted RVC
experiment can be used instead, reusing its RMVPE F0 and features.

## 1. Pretrain a base (one per embedder)

`prepare_pretrain_dataset.py` joins M4Singer, GTSinger and EARS 0-4 into
`assets/datasets/style_pretrain`. Then, with "New base (pretrain)":

```
python core.py style_extract --model_name style_base_contentvec \
    --dataset_path assets/datasets/style_pretrain --embedder_model contentvec
python core.py style_train --model_name style_base_contentvec --speech_speakers 0-4
```

The coarse melody is a step contour (`representation.coarse_mode: notes` in
`base.yaml`): each note holds its pitch, so scoops, glides and phrase-end drops
are generated rather than copied from the source. A dataset extracted with
another representation can be re-split without re-extracting:
`python tools/style_flow/redecompose.py --data <old style_data> --out <new>`.

Writes `logs/style_base_contentvec/style_base.pt`; copy it to
`rvc/models/style/` to distribute it. `tools/style_flow/pretrain.py` has the
extra knobs (`--overfit 8` for a sanity run).

## 2. Fine-tune per voice

With "Fine-tune a base" and a style base picked:

```
python core.py style_extract --model_name <name> --dataset_path assets/datasets/<singer> \
    --base_path rvc/models/style/style_base.pt
python core.py style_train --model_name <name> --base_path rvc/models/style/style_base.pt
```

Writes `logs/<name>/<name>_style.pt` with the singer's descriptors.
The fine-tune trains LoRA adapters; `--precision bf16|fp16|fp32`. Without
`--dataset_path`, the RVC experiment `logs/<name>` is used.

## 3. Inference

Inference tab, "Style Model": pick any style model (independent of the voice
model), then strength, F0 style rate, steps, CFG and descriptor offsets.
Autotune and F0 files disable the style for that conversion. From the CLI,
`infer`/`batch_infer` take `--style_model`, `--style_strength`, `--style_rate`,
`--style_steps`, `--style_cfg` and `--style_descriptors '{"vibrato_extent_cents": 1}'`.

Contours only, for evaluation:

```
python tools/style_flow/infer.py --model logs/<name>/<name>_style.pt --input src.wav --out out.npz
python tools/style_flow/evaluate.py --source src.wav --output out.npz --target logs/<name>/style_data
python tools/style_flow/visualize.py --source src.wav --generated out.npz --target logs/<name>/style_data --out plot.png
```

## Results

| Phase | Metric | Value |
|---|---|---|
| 1 pretrain | melody error (cents) | |
| 1 pretrain | descriptor distance (val) | |
| 1 pretrain | response (vibrato extent low / high) | |
| 2 fine-tune | descriptor distance, base / fine-tuned | |
| 3 inference | descriptor distance to target, source / output | |
| 3 inference | melody error vs. source (cents) | |
