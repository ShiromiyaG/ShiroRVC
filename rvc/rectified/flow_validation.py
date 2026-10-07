"""What a flow run shows of itself as it trains: the held-out loss and the
preview clips, both through the averaged weights."""

import torch

from rvc.rectified.mel import denormalize_mel, normalize_mel

#: Sampling steps of the previews.
PREVIEW_STEPS = 16
#: Where in the trained time range the validation loss is taken.
EVAL_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)


@torch.no_grad()
def preview(model, ema, previews, clips, vocoder, data: dict, device, epoch: int, step: int):
    """Every preview clip through the flow's EMA weights. With a vocoder,
    also its real mel through the vocoder alone and the aux decoder's mel,
    and for a held-out clip the flow started from the real mel instead of
    the aux decoder's, on the same noise: together they tell the vocoder's,
    the aux decoder's and the flow's errors apart. Without a vocoder, the mel
    figure only. ``clips`` pairs each clip with whether it was held out;
    ``data`` is the config's."""
    with previews.media.writer(step) as media, ema.applied(model):
        model.eval()
        for index, (clip, held_out) in enumerate(clips):
            inputs = clip.inputs.to(device)
            real = normalize_mel(clip.mel.to(device), data)
            noise = torch.randn_like(real)
            generated = model.sample(inputs, steps=PREVIEW_STEPS, noise=noise)
            if vocoder is None:
                previews.log(
                    epoch, step, clip.path, generated_mel=denormalize_mel(generated, data),
                    reference_mel=clip.mel, sample_index=index, writer=media,
                )
                continue
            mels = {"vocoder_on_real_mel": real}
            if model.aux is not None:
                mels["aux_decoder"] = model.aux_mel(inputs)
            if model.starts_from_aux and held_out:
                mels["flow_from_real_mel"] = model.sample(
                    inputs, steps=PREVIEW_STEPS, noise=noise, start_mel=real
                )
            previews.log(
                epoch, step, clip.path,
                generated_audio=vocoder(generated, inputs.f0),
                reference_audio=clip.audio.to(device),
                extra_audio={name: vocoder(mel, inputs.f0) for name, mel in mels.items()},
                sample_index=index, writer=media,
            )
        model.train()


@torch.no_grad()
def evaluate(model, ema, holdout, data: dict, device, amp_dtype, writer, step: int):
    """Held-out flow loss through the EMA weights, from the same noise and
    at the same times on every call, so the curve compares across steps."""
    totals = torch.zeros(len(EVAL_FRACTIONS), device=device)
    aux_total, items = 0.0, 0
    with ema.applied(model):
        model.eval()
        for index, (mel, inputs) in enumerate(holdout):
            inputs = inputs.to(device)
            mel = normalize_mel(mel.to(device), data) * inputs.mask
            generator = torch.Generator(device=device).manual_seed(index)
            noise = torch.randn(mel.shape, device=device, generator=generator)
            with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                losses, aux = model.validation_losses(mel, inputs, noise, EVAL_FRACTIONS)
            totals += losses.float() * mel.shape[0]
            aux_total += (aux.item() if aux is not None else 0.0) * mel.shape[0]
            items += mel.shape[0]
        model.train()
    totals /= max(1, items)
    writer.add_scalar("holdout/flow", totals.mean().item(), step)
    for fraction, value in zip(EVAL_FRACTIONS, totals.tolist()):
        writer.add_scalar(f"holdout/flow_t{fraction:g}", value, step)
    if model.aux is not None:
        writer.add_scalar("holdout/aux_mel_l1", aux_total / max(1, items), step)
