import torch
from .mel_extractor import Wav2Mel, Wav2MelModule
import pathlib


class DotDict(dict):
    """Dict with attribute access, e.g. ``config.a.b`` instead of ``config['a']['b']``."""

    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def spawn_wav2mel(args: DotDict, device: str = None) -> Wav2MelModule:
    _type = args.mel.type
    if (str(_type).lower() == 'none') or (str(_type).lower() == 'default'):
        _type = 'default'
    elif str(_type).lower() == 'stft':
        _type = 'stft'
    else:
        raise ValueError(f'  [ERROR] torchfcpe.tools.args.spawn_wav2mel: {_type} is not a supported args.mel.type')
    wav2mel = Wav2MelModule(
        sr=catch_none_args_opti(
            args.mel.sr,
            default=16000,
            func_name='torchfcpe.tools.spawn_wav2mel',
            warning_str='args.mel.sr is None',
        ),
        n_mels=catch_none_args_opti(
            args.mel.num_mels,
            default=128,
            func_name='torchfcpe.tools.spawn_wav2mel',
            warning_str='args.mel.num_mels is None',
        ),
        n_fft=catch_none_args_opti(
            args.mel.n_fft,
            default=1024,
            func_name='torchfcpe.tools.spawn_wav2mel',
            warning_str='args.mel.n_fft is None',
        ),
        win_size=catch_none_args_opti(
            args.mel.win_size,
            default=1024,
            func_name='torchfcpe.tools.spawn_wav2mel',
            warning_str='args.mel.win_size is None',
        ),
        hop_length=catch_none_args_opti(
            args.mel.hop_size,
            default=160,
            func_name='torchfcpe.tools.spawn_wav2mel',
            warning_str='args.mel.hop_size is None',
        ),
        fmin=catch_none_args_opti(
            args.mel.fmin,
            default=0,
            func_name='torchfcpe.tools.spawn_wav2mel',
            warning_str='args.mel.fmin is None',
        ),
        fmax=catch_none_args_opti(
            args.mel.fmax,
            default=8000,
            func_name='torchfcpe.tools.spawn_wav2mel',
            warning_str='args.mel.fmax is None',
        ),
        clip_val=1e-05,
        mel_type=_type,
    )
    device = catch_none_args_opti(
        device,
        default='cpu',
        func_name='torchfcpe.tools.spawn_wav2mel',
        warning_str='.device is None',
    )
    return wav2mel.to(torch.device(device))


def catch_none_args_opti(x, default, func_name, warning_str=None, level='WARN'):
    """Return default if x is None, optionally logging a warning."""
    if x is None:
        if warning_str is not None:
            print(f'[{level}] {warning_str}; using default {default} ({func_name}).')
        return default
    else:
        return x


def catch_none_args_must(x, func_name, warning_str):
    """Raise if x is None."""
    level = "ERROR"
    if x is None:
        raise ValueError(f'  [{level}] {warning_str}')
    else:
        return x


def get_device(device: str, func_name: str) -> str:
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    else:
        device = device

    if ((device == 'cuda' and not torch.cuda.is_available()) or
            (device == 'mps' and not torch.backends.mps.is_available())):
        print(f'[WARNING] {device} is not available; switching to the CPU.')
        device = 'cpu'

    return device


def get_config_json_in_same_path(path: str) -> str:
    path = pathlib.Path(path)
    config_json = path.parent / 'config.json'
    if config_json.exists():
        return str(config_json)
    else:
        raise FileNotFoundError(f'  [ERROR] {config_json} not found.')
