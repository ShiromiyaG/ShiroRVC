"""The flow's training features computed once and kept on disk: every clip's
mel and curves, and its shifted and stretched copies, so a step only reads
them. With the lengths known ahead, batches can be made of whole clips of
similar length.
"""

import hashlib
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from rvc.lib.paths import LOGS_DIR
from rvc.lib.terminal import info, progress_task
from rvc.rectified.common import FlowItem, RectifiedDataset, to_mel_rate

TAG = "[CACHE]"
CACHE_VERSION = 1
#: Lengths this many frames apart sort as equal, so batches change by epoch.
LENGTH_GRID = 8
#: Batches are padded to a multiple of this many frames, so they come in few
#: shapes and cuDNN's benchmark settles on each. A multiple of ``LENGTH_GRID``.
PAD_GRID = 32
CURVES = ("f0", "energy", "breathiness", "tension")


def augmentation_plan(count: int, settings: dict, seed: int) -> list:
    """[clip, key shift, speed] of every item: each of ``count`` clips as it
    is, then DiffSinger's offline augmentation. ``key_shift_scale`` times the
    clips are shifted, and ``time_stretch_scale`` times the result is
    stretched, split between plain clips, copies of shifted ones and shifted
    ones stretched in place."""
    rng = random.Random(seed)
    clips = range(count)
    plan = [[clip, 0.0, 1.0] for clip in clips]
    shift_range = float(settings.get("key_shift_range", 0.0))
    shift_scale = float(settings.get("key_shift_scale", 0.0)) if shift_range > 0 else 0.0
    shifted = [
        [clip, rng.uniform(-shift_range, shift_range), 1.0]
        for clip in rng.choices(clips, k=int(shift_scale * count))
    ]
    low, high = settings.get("time_stretch_range", (1.0, 1.0))
    stretch_scale = float(settings.get("time_stretch_scale", 0.0)) if low < high else 0.0
    stretched = []
    if stretch_scale > 0:
        def speed():
            return low * (high / low) ** rng.random()

        plain = int(stretch_scale / (1 + shift_scale) * count)
        copies = int(shift_scale * stretch_scale / (1 + shift_scale) * count)
        in_place = int(shift_scale * stretch_scale / (1 + stretch_scale) * count)
        stretched = [[clip, 0.0, speed()] for clip in rng.choices(clips, k=plain)]
        if shifted:
            stretched += [[clip, shift, speed()] for clip, shift, _ in rng.choices(shifted, k=copies)]
            for item in rng.sample(shifted, k=min(in_place, len(shifted))):
                item[2] = speed()
    return plan + shifted + stretched


def cache_folder(name: str, config: dict, entries, seed: int) -> str:
    """Where the cache of these ``entries`` under this ``config`` lives; any
    change to what the features depend on gives another folder."""
    settings = config["flow"]
    recipe = {
        "version": CACHE_VERSION,
        "data": config["data"],
        "tension": bool(settings["model"].get("tension", False)),
        "augmentation": [settings.get(key) for key in (
            "key_shift_range", "key_shift_scale", "time_stretch_range", "time_stretch_scale")],
        "seed": seed,
        "clips": hashlib.sha256("\n".join(f"{entry[0]}|{entry[4]}" for entry in entries).encode()).hexdigest(),
    }
    fingerprint = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:16]
    return os.path.join(LOGS_DIR, name, "flow_cache", fingerprint)


def item_path(folder: str, number: int) -> str:
    return os.path.join(folder, f"{number // 1000:05d}", f"{number:08d}.npz")


class _Writer(Dataset):
    """Writes a clip's items on being read, and gives (number, frames) of each."""

    def __init__(self, dataset: RectifiedDataset, by_clip: dict, folder: str):
        self.dataset = dataset
        self.by_clip = by_clip
        self.clips = sorted(by_clip)
        self.folder = folder

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, position):
        clip = self.clips[position]
        source, written = None, []
        for number, key_shift, speed in self.by_clip[clip]:
            path = item_path(self.folder, number)
            if os.path.exists(path):
                with np.load(path) as saved:
                    written.append((number, int(saved["f0"].shape[0])))
                continue
            if source is None:
                source = self.dataset.source(clip)
            hop = int(round(self.dataset.hop * speed))
            item = self.dataset.features(*source, key_shift, hop)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Under another name until whole, so an interrupted run leaves no half file.
            with open(path + ".tmp", "wb") as handle:
                np.savez(
                    handle, mel=item.mel.numpy().astype(np.float16),
                    **{curve: getattr(item, curve).numpy() for curve in CURVES},
                )
            os.replace(path + ".tmp", path)
            written.append((number, item.mel.shape[-1]))
        return written


def build_cache(name: str, config: dict, entries, workers: int) -> None:
    """Compute and write what is missing of the cache for ``entries``. An
    interrupted build continues from the items already written."""
    settings = config["flow"]
    seed = int(settings.get("seed", 1234))
    folder = cache_folder(name, config, entries, seed)
    index_path = os.path.join(folder, "index.json")
    if os.path.exists(index_path):
        return
    plan = augmentation_plan(len(entries), settings, seed)
    by_clip = {}
    for number, (clip, key_shift, speed) in enumerate(plan):
        by_clip.setdefault(clip, []).append((number, key_shift, speed))
    os.makedirs(folder, exist_ok=True)
    info(f"Writing the features of {len(plan)} items ({len(entries)} clips and their augmented "
         f"copies) to {folder}; this is done once.", tag=TAG)
    dataset = RectifiedDataset(entries, config, "flow", 0, augment=False)
    loader = DataLoader(
        _Writer(dataset, by_clip, folder), batch_size=None, num_workers=workers,
    )
    frames = [0] * len(plan)
    with progress_task(len(plan), "Feature cache") as (progress, task):
        for written in loader:
            for number, count in written:
                frames[number] = int(count)
            progress.update(task, advance=len(written))
    items = [[clip, frames[number], key_shift, hop_speed(dataset.hop, speed)]
             for number, (clip, key_shift, speed) in enumerate(plan)]
    with open(index_path + ".tmp", "w", encoding="utf-8") as handle:
        json.dump({"items": items}, handle)
    os.replace(index_path + ".tmp", index_path)


def hop_speed(hop: int, speed: float) -> float:
    """``speed`` as the whole hop it is read with gives it."""
    return int(round(hop * speed)) / hop


def load_cache(name: str, config: dict, entries):
    """The cache's folder and its items, [clip, frames, key shift, speed]."""
    folder = cache_folder(name, config, entries, int(config["flow"].get("seed", 1234)))
    with open(os.path.join(folder, "index.json"), encoding="utf-8") as handle:
        return folder, json.load(handle)["items"]


class CachedFlowDataset(Dataset):
    """The cache's items as ``FlowItem``s, cut at random to ``max_frames``.
    ``source`` is the dataset of the clips they were made from, whose content
    is read as it is and not kept twice."""

    def __init__(self, source: RectifiedDataset, folder: str, items: list, max_frames: int):
        self.source = source
        self.folder = folder
        self.items = items
        self.max_frames = int(max_frames)

    def __len__(self):
        return len(self.items)

    def lengths(self) -> np.ndarray:
        """Frames each item gives."""
        return np.minimum(np.array([item[1] for item in self.items]), self.max_frames)

    def __getitem__(self, number):
        clip, frames, key_shift, speed = self.items[number]
        length = min(frames, self.max_frames)
        start = random.randint(0, frames - length)
        stop = start + length
        with np.load(item_path(self.folder, number)) as saved:
            mel = torch.from_numpy(saved["mel"][:, start:stop].astype(np.float32))
            curves = {curve: torch.from_numpy(saved[curve][start:stop]) for curve in CURVES}
        source = self.source
        hop = int(round(source.hop * speed))
        content = to_mel_rate(source.content(clip), frames, source.sample_rate, hop)[start:stop]
        return FlowItem(
            mel=mel, content=content, key_shift=key_shift, speed=speed,
            speaker=int(source.entries[clip][4]), **curves,
        )


class BucketBatchSampler(Sampler):
    """Batches of items of similar length, each within ``max_frames`` frames
    as padded to ``PAD_GRID`` and ``max_items`` items, formed anew every
    epoch, always the same number of them, and dealt between the ranks."""

    def __init__(self, lengths, max_frames: int, max_items: int, seed: int = 1234, rank: int = 0, world: int = 1):
        self.lengths = np.asarray(lengths)
        self.max_frames = int(max_frames)
        self.max_items = int(max_items)
        self.seed, self.rank, self.world = int(seed), int(rank), int(world)
        self.epoch = 0
        self._formed = None
        self._count = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _form(self, rng) -> list:
        order = rng.permutation(len(self.lengths))
        # Grouped as the padding rounds, up, so a batch's first item is its longest padded.
        order = order[np.argsort(-((self.lengths[order] - 1) // LENGTH_GRID), kind="stable")]
        batches, batch, longest = [], [], 0
        for index in order.tolist():
            frames = -(-int(self.lengths[index]) // PAD_GRID) * PAD_GRID
            if batch and (len(batch) == self.max_items
                          or (len(batch) + 1) * max(longest, frames) > self.max_frames):
                batches.append(batch)
                batch, longest = [], 0
            batch.append(index)
            longest = max(longest, frames)
        if batch:
            batches.append(batch)
        return [batches[index] for index in rng.permutation(len(batches)).tolist()]

    def _batches(self) -> list:
        if self._formed is not None and self._formed[0] == self.epoch:
            return self._formed[1]
        if self._count is None:
            self._count = len(self._form(np.random.default_rng(self.seed)))
        # The same on every rank, which then takes its own share.
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = self._form(rng)
        # Every epoch has the same number of steps: batches over it are left
        # out, and the ones missing are repeated.
        missing = max(self._count - len(batches), 0)
        repeated = rng.choice(len(batches), missing, replace=False).tolist()
        batches = (batches + [batches[index] for index in repeated])[: self._count]
        each = len(batches) // self.world
        if each == 0:
            raise ValueError(f"{len(batches)} batches is fewer than one for each of {self.world} GPUs.")
        batches = batches[self.rank : each * self.world : self.world]
        self._formed = (self.epoch, batches)
        return batches

    def __len__(self):
        return len(self._batches())

    def __iter__(self):
        return iter(self._batches())
