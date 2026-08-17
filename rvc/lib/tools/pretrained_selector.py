import os

from rvc.configs.vocoders import get_vocoder_sample_rates, get_vocoder_spec, normalize_vocoder


def pretrained_selector(vocoder, sample_rate):
    vocoder_id = normalize_vocoder(vocoder)
    if int(sample_rate) not in get_vocoder_sample_rates(vocoder_id):
        return "", ""

    pretrained_dir = get_vocoder_spec(vocoder_id).get("pretrained_dir", vocoder_id)
    base_path = os.path.join("rvc", "models", "pretraineds", pretrained_dir)

    path_g = os.path.join(base_path, f"f0G{str(sample_rate)[:2]}k.pth")
    path_d = os.path.join(base_path, f"f0D{str(sample_rate)[:2]}k.pth")

    return (
        path_g if os.path.exists(path_g) else "",
        path_d if os.path.exists(path_d) else "",
    )
