import os
import random
import numpy as np
import torch
import torch.utils.data

from mel_processing import spectrogram_torch
from utils import load_filepaths_and_text, load_wav_to_torch

from rvc.lib.terminal import warning

debug_shapes = False


def source_group_key(audiopath):
    """Group slices back into the recording they were cut from.

    Preprocessing names every slice ``{speaker}_{source}_{slice}``
    (``rvc/train/preprocess/preprocess.py``), so dropping the last field
    recovers the source.  Falls back to the whole stem for filelists that do
    not follow the convention, which makes every slice its own group -- a
    weaker split, but never a wrong one.
    """
    stem = os.path.splitext(os.path.basename(audiopath))[0]
    head, separator, _tail = stem.rpartition("_")
    return head if separator else stem


def holdout_split_indices(
    audiopaths_and_text,
    fraction=0.02,
    minimum=16,
    maximum=96,
    seed=1234,
    lengths=None,
    max_frames=None,
):
    """Partition filelist rows into ``(train_indices, holdout_indices)``.

    The split is by *source recording*, not by slice.  Slices cut from one
    recording share room tone, mic placement, level and usually phonetic
    context, so a slice-wise holdout is most of the way to being training data:
    a model can memorise its way to a good score on it and the curve never
    turns over.  Whole sources held out is the only version that measures
    generalisation, which is the entire point of having a holdout.

    Deterministic in ``seed``, because a split that moved between runs would
    leak the holdout into training one resume at a time.

    Returns an empty holdout when the dataset cannot afford one, so the caller
    can fall back rather than train on almost nothing.
    """
    total = len(audiopaths_and_text)
    groups = {}
    for index, row in enumerate(audiopaths_and_text):
        groups.setdefault(source_group_key(row[0]), []).append(index)

    target = min(int(maximum), max(int(minimum), int(total * float(fraction))))
    # Four groups so at least two survive in training, and a 4x margin so the
    # holdout never eats a meaningful share of a small dataset.
    if len(groups) < 4 or target * 4 > total:
        return list(range(total)), []

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    held_keys = set()
    held_count = 0
    for key in keys:
        if held_count >= target:
            break
        group = groups[key]
        # One long recording can hold more slices than the whole budget.
        # Skipping it keeps the holdout from collapsing to a single source.
        if len(group) > target and held_keys:
            continue
        held_keys.add(key)
        held_count += len(group)

    if not held_keys or len(groups) - len(held_keys) < 2:
        return list(range(total)), []

    train_indices, holdout_indices = [], []
    for index, row in enumerate(audiopaths_and_text):
        if source_group_key(row[0]) in held_keys:
            holdout_indices.append(index)
        else:
            train_indices.append(index)

    # The whole source stays out of training; only the *evaluation* list is
    # trimmed, so trimming can never leak training data back in.
    #
    # Trimmed by duration and not only by count, because the evaluation cost is
    # seconds of audio to synthesise, not slices.  On a 47 h dataset the slices
    # average six seconds, so a 96-slice cap is ten minutes of audio per
    # evaluation -- enough to cost several percent of training throughput for a
    # metric that is already stable on a fraction of it.  The set is fixed, so
    # its noise comes from the model rather than from sampling, and a couple of
    # minutes spread over every held-out recording measures the same thing.
    # Round-robin across the held recordings rather than straight down the
    # list, so a budget that runs out part way still covers every recording
    # instead of spending itself on whichever one sorts first.
    by_source = {}
    for index in holdout_indices:
        by_source.setdefault(source_group_key(audiopaths_and_text[index][0]), []).append(index)
    interleaved = []
    for position in range(max(len(group) for group in by_source.values())):
        for key in sorted(by_source):
            if position < len(by_source[key]):
                interleaved.append(by_source[key][position])

    interleaved = interleaved[: int(maximum)]
    if max_frames and lengths:
        budgeted, used = [], 0
        for index in interleaved:
            if used >= max_frames and budgeted:
                break
            budgeted.append(index)
            used += lengths[index]
        interleaved = budgeted
    return train_indices, sorted(interleaved)


class TextAudioLoaderMultiNSFsid(torch.utils.data.Dataset):
    """
    Dataset that loads text and audio pairs.

    Args:
        hparams: Hyperparameters.
    """

    def __init__(self, hparams, n_mel_bins=192):
        self.audiopaths_and_text = load_filepaths_and_text(hparams.training_files)
        self.max_wav_value = hparams.max_wav_value
        self.sample_rate = hparams.sample_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sample_rate = hparams.sample_rate
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 5000)
        self.n_mel_bins = n_mel_bins
        self.mel_fmin = hparams.mel_fmin
        self.mel_fmax = hparams.mel_fmax
        self._filter()

    def _filter(self):
        """
        Filters audio paths and text pairs based on text length.
        """
        audiopaths_and_text_new = []
        lengths = []
        for audiopath, text, pitch, pitchf, dv in self.audiopaths_and_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_and_text_new.append([audiopath, text, pitch, pitchf, dv])
                lengths.append(os.path.getsize(audiopath) // (3 * self.hop_length))
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_sid(self, sid):
        """
        Converts speaker ID to a LongTensor.

        Args:
            sid (str): Speaker ID.
        """
        try:
            sid = torch.LongTensor([int(sid)])
        except ValueError as error:
            warning(
                f"Speaker ID {sid!r} is not an integer ({error}); using 0.",
                tag="[DATA]",
            )
            sid = torch.LongTensor([0])
        return sid

    def get_audio_text_pair(self, audiopath_and_text):
        """
        Loads and processes audio and text data for a single pair.

        Args:
            audiopath_and_text (list): List containing audio path, text, pitch, pitchf, and speaker ID.
        """
        file = audiopath_and_text[0]
        phone = audiopath_and_text[1]
        pitch = audiopath_and_text[2]
        pitchf = audiopath_and_text[3]
        dv = audiopath_and_text[4]

        phone, pitch, pitchf = self.get_labels(phone, pitch, pitchf)
        spec, wav = self.get_audio(file)
        dv = self.get_sid(dv)

        if debug_shapes:
            # After getting spec, wav, phone, pitch, pitchf
            print(f" Data_Utils [DEBUG] file: {file}")
            print(f"        spec.shape  = {spec.shape}")  # (n_mels, T)
            print(f"        phone.shape = {phone.shape}") # (T, dim)
            print(f"        pitch.shape = {pitch.shape}")
            print(f"        pitchf.shape= {pitchf.shape}")


        len_phone = phone.size(0)
        len_spec = spec.size()[-1]
        if len_phone != len_spec:
            if debug_shapes:
                print(f"  └── [LEN MISMATCH] spec={len_spec}  phone={len_phone}  → trimming everything to {min(len_phone, len_spec)}")

            len_min = min(len_phone, len_spec)
            len_wav = len_min * self.hop_length

            spec = spec[:, :len_min]
            wav = wav[:, :len_wav]

            phone = phone[:len_min, :]
            pitch = pitch[:len_min]
            pitchf = pitchf[:len_min]

        return (spec, wav, phone, pitch, pitchf, dv)

    def get_labels(self, phone, pitch, pitchf):
        """
        Loads and processes phoneme, pitch, and pitchf labels.

        Args:
            phone (str): Path to phoneme label file.
            pitch (str): Path to pitch label file.
            pitchf (str): Path to pitchf label file.
        """
        phone = np.load(phone, allow_pickle=False)
        phone = torch.from_numpy(phone).float().repeat_interleave(2, dim=0)
        if debug_shapes:
            print(f"[ Data_Utils [DEBUG]:get_labels] after scaling = {phone.shape}")  # AFTER repeat

        pitch = np.load(pitch, allow_pickle=False)
        pitchf = np.load(pitchf, allow_pickle=False)
        n_num = min(phone.shape[0], 900)
        phone = phone[:n_num, :]
        pitch = pitch[:n_num]
        pitchf = pitchf[:n_num]
        pitch = torch.LongTensor(pitch)
        pitchf = torch.FloatTensor(pitchf)
        return phone, pitch, pitchf

    def get_audio(self, filename):
        """
        Loads and processes audio data.

        Args:
            filename (str): Path to audio file.
        """
        audio, sample_rate = load_wav_to_torch(filename)
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"{sample_rate} SR doesn't match target {self.sample_rate} SR"
            )
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            self.filter_length,
            self.hop_length,
            self.win_length,
            center=False,
        )
        return torch.squeeze(spec, 0), audio_norm

    def __getitem__(self, index):
        """
        Returns a single audio-text pair.

        Args:
            index (int): Index of the data sample.
        """
        return self.get_audio_text_pair(self.audiopaths_and_text[index])


    def get_file_paths(self, indices):
        """
        Returns a list of file paths for the given batch indices.

        Args:
            indices (list or tensor): Indices of the samples in the batch.
        """
        file_paths = [self.audiopaths_and_text[idx][0] for idx in indices]
        return file_paths


    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.audiopaths_and_text)


class TextAudioCollateMultiNSFsid:
    """
    Collates text and audio data for training.

    Args:
        return_ids (bool, optional): Whether to return sample IDs. Defaults to False.
    """

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """
        Collates a batch of data samples.

        Args:
            batch (list): List of data samples.
        """
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]), dim=0, descending=True
        )

        max_spec_len = max([x[0].size(1) for x in batch])
        max_wave_len = max([x[1].size(1) for x in batch])
        spec_lengths = torch.LongTensor(len(batch))
        wave_lengths = torch.LongTensor(len(batch))
        spec_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len)
        wave_padded = torch.FloatTensor(len(batch), 1, max_wave_len)
        spec_padded.zero_()
        wave_padded.zero_()

        max_phone_len = max([x[2].size(0) for x in batch])
        phone_lengths = torch.LongTensor(len(batch))
        phone_padded = torch.FloatTensor(
            len(batch), max_phone_len, batch[0][2].shape[1]
        )
        phone_padded.zero_()
        pitch_padded = torch.LongTensor(len(batch), max_phone_len)
        pitchf_padded = torch.FloatTensor(len(batch), max_phone_len)
        pitch_padded.zero_()
        pitchf_padded.zero_()
        sid = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            spec = row[0]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wave = row[1]
            wave_padded[i, :, : wave.size(1)] = wave
            wave_lengths[i] = wave.size(1)

            phone = row[2]
            phone_padded[i, : phone.size(0), :] = phone
            phone_lengths[i] = phone.size(0)

            pitch = row[3]
            pitch_padded[i, : pitch.size(0)] = pitch
            pitchf = row[4]
            pitchf_padded[i, : pitchf.size(0)] = pitchf

            sid[i] = row[5]

        return (
            phone_padded,
            phone_lengths,
            pitch_padded,
            pitchf_padded,
            spec_padded,
            spec_lengths,
            wave_padded,
            wave_lengths,
            sid,
        )


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Distributed sampler that groups data into buckets based on length.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to sample from.
        batch_size (int): Batch size.
        boundaries (list): List of length boundaries for buckets.
        num_replicas (int, optional): Number of processes participating in distributed training. Defaults to None.
        rank (int, optional): Rank of the current process. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
        self.start_index = 0

    def _create_buckets(self):
        """
        Creates buckets of data samples based on length.
        """
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, -1, -1):  #
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        """
        Iterates over batches of data samples.
        """
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]

        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        start = self.start_index
        self.start_index = 0

        return iter(self.batches[start:])

    def _bisect(self, x, lo=0, hi=None):
        """
        Performs binary search to find the bucket index for a given length.

        Args:
            x (int): Length to find the bucket for.
            lo (int, optional): Lower bound of the search range. Defaults to 0.
            hi (int, optional): Upper bound of the search range. Defaults to None.
        """
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        """
        Returns the length of the sampler.
        """
        return self.num_samples // self.batch_size
