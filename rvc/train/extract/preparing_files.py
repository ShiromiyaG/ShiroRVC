import os
import shutil
from random import shuffle
from rvc.configs.config import Config
import json
import librosa
import soundfile as sf

config = Config()
current_directory = os.getcwd()


def ensure_mute_audio(mute_base_path: str, sample_rate: int) -> str:
    target_path = os.path.join(
        mute_base_path,
        "sliced_audios",
        f"mute{int(sample_rate)}.wav",
    )
    if os.path.isfile(target_path):
        return target_path

    source_candidates = [
        os.path.join(mute_base_path, "sliced_audios", "mute48000.wav"),
        os.path.join(mute_base_path, "sliced_audios", "mute40000.wav"),
        os.path.join(mute_base_path, "sliced_audios", "mute32000.wav"),
        os.path.join(mute_base_path, "sliced_audios", "mute24000.wav"),
        os.path.join(mute_base_path, "sliced_audios_16k", "mute.wav"),
    ]
    source_path = next(
        (path for path in source_candidates if os.path.isfile(path)),
        None,
    )
    if source_path is None:
        raise FileNotFoundError(target_path)

    audio, source_rate = sf.read(source_path, always_2d=False)
    if int(source_rate) != int(sample_rate):
        if audio.ndim == 1:
            audio = librosa.resample(
                audio,
                orig_sr=source_rate,
                target_sr=int(sample_rate),
                res_type="soxr_vhq",
            )
        else:
            audio = librosa.resample(
                audio.T,
                orig_sr=source_rate,
                target_sr=int(sample_rate),
                res_type="soxr_vhq",
            ).T

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    sf.write(target_path, audio, int(sample_rate), subtype="PCM_16")
    return target_path


def generate_config(sample_rate: int, model_path: str, vocoder_arch: str):
    from rvc.configs.vocoders import normalize_vocoder

    vocoder_arch = normalize_vocoder(vocoder_arch)
    config_path = os.path.join("rvc", "configs", vocoder_arch, f"{sample_rate}.json")
    config_save_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_save_path):
        shutil.copyfile(config_path, config_save_path)
        print(f"Config saved at {config_save_path}")
    else:
        print(f"Config file already exists at {config_save_path}")

def generate_filelist(
    model_path: str, sample_rate: int, include_mutes: int = 2, embedder_model: str = "contentvec", vocoder_arch: str = "hifi"
):
    from rvc.configs.vocoders import normalize_vocoder

    vocoder_arch = normalize_vocoder(vocoder_arch)
    gt_wavs_dir = os.path.join(model_path, "sliced_audios")
    feature_dir = os.path.join(model_path, f"extracted")

    f0_dir, f0nsf_dir = None, None
    f0_dir = os.path.join(model_path, "f0")
    f0nsf_dir = os.path.join(model_path, "f0_voiced")

    gt_wavs_files = sorted(os.listdir(gt_wavs_dir), key=lambda x: x.split(".")[0])
    feature_files = sorted(os.listdir(feature_dir), key=lambda x: x.split(".")[0])

    f0_files = sorted(os.listdir(f0_dir), key=lambda x: x.split(".")[0])
    f0nsf_files = sorted(os.listdir(f0nsf_dir), key=lambda x: x.split(".")[0])

    options = []

    if embedder_model == "contentvec":
        mute_folder = "mute"
    elif embedder_model == "spin_v1":
        mute_folder = "mute_spin_v1"
    else:
        mute_folder = "mute_spin_v2"

    mute_base_path = os.path.join(current_directory, "logs", mute_folder)

    sids = []
    
    for gt_wavs_file, feature_file, f0_file, f0nsf_file in zip(gt_wavs_files, feature_files, f0_files, f0nsf_files, strict=True):
        sid = gt_wavs_file.split("_")[0]
        if sid not in sids:
            sids.append(sid)
        options.append(
            f"{os.path.join(gt_wavs_dir, gt_wavs_file)}|{os.path.join(feature_dir, feature_file)}|{os.path.join(f0_dir, f0_file)}|{os.path.join(f0nsf_dir, f0nsf_file)}|{sid}"
        )

    if include_mutes > 0:
        mute_audio_path = ensure_mute_audio(mute_base_path, sample_rate)
        mute_feature_path = os.path.join(
            mute_base_path, f"extracted", "mute.npy"
        )
        mute_f0_path = os.path.join(mute_base_path, "f0", "mute.wav.npy")
        mute_f0nsf_path = os.path.join(mute_base_path, "f0_voiced", "mute.wav.npy")

        # adding x files per sid
        for sid in sids * include_mutes:
            options.append(
                f"{mute_audio_path}|{mute_feature_path}|{mute_f0_path}|{mute_f0nsf_path}|{sid}"
            )

    file_path = os.path.join(model_path, "model_info.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    data["speakers_id"] = len(sids)
    data["vocoder_architecture"] = vocoder_arch

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    shuffle(options)


    with open(os.path.join(model_path, "filelist.txt"), "w") as f:
        f.write("\n".join(options))
