"""Build ``assets/datasets/vocoder_pretrain``: many EARS speakers plus every
singer of ``style_pretrain`` with part of their songs.

EARS speakers are downloaded, converted to 24-bit FLAC and written in place;
singing files are hard links into ``style_pretrain``. Speakers are balanced by
gender and spread over age groups; songs are drawn at random under a per-singer
cap, so every singer keeps a share and the short ones keep all they have.

Usage: python tools/build_vocoder_dataset.py [ears_speakers] [singing_hours] [total_hours]
"""

import io
import json
import os
import random
import sys
import urllib.request
import zipfile

import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = os.path.join(ROOT, "assets", "datasets")
SOURCE = os.path.join(DATASETS, "style_pretrain")
OUTPUT = os.path.join(DATASETS, "vocoder_pretrain")
EARS_URL = "https://github.com/facebookresearch/ears_dataset/releases/download/dataset/{}.zip"
EARS_STATS = "https://raw.githubusercontent.com/facebookresearch/ears_dataset/main/speaker_statistics.json"
SEED = 0


def hours(paths):
    return sum(sf.info(path).duration for path in paths) / 3600


def pick_ears(count):
    """``count`` EARS speakers, half male, each half spread evenly over age."""
    with urllib.request.urlopen(EARS_STATS) as response:
        stats = json.load(response)
    male = sorted((s for s in stats if stats[s]["gender"] == "male"), key=lambda s: (stats[s]["age"], s))
    other = sorted((s for s in stats if stats[s]["gender"] != "male"), key=lambda s: (stats[s]["age"], s))
    n_male = min(len(male), count // 2)

    def spread(pool, n):
        return [pool[int(i * len(pool) / n)] for i in range(n)]

    return sorted(spread(male, n_male) + spread(other, count - n_male))


def fetch_ears(speaker, folder):
    """``folder`` with the speaker's recordings as 24-bit FLAC, from the
    ``style_pretrain`` copy when there is one."""
    os.makedirs(folder, exist_ok=True)
    done = os.path.join(folder, ".complete")
    if os.path.exists(done):
        return
    existing = [name for name in os.listdir(SOURCE) if name.endswith(f"_ears-{speaker}")]
    if existing:
        source = os.path.join(SOURCE, existing[0])
        for name in os.listdir(source):
            target = os.path.join(folder, name)
            if not os.path.exists(target):
                os.link(os.path.join(source, name), target)
    else:
        print(f"Downloading {speaker}...", flush=True)
        with urllib.request.urlopen(EARS_URL.format(speaker)) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        for member in archive.namelist():
            if not member.lower().endswith(".wav"):
                continue
            audio, rate = sf.read(io.BytesIO(archive.read(member)), dtype="float32")
            name = os.path.splitext(os.path.basename(member))[0] + ".flac"
            sf.write(os.path.join(folder, name), audio, rate, subtype="PCM_24")
    open(done, "w").close()


def cap_for(budgets, target):
    """The per-singer cap whose capped hours sum to ``target``."""
    low, high = 0.0, max(budgets)
    for _ in range(60):
        cap = (low + high) / 2
        if sum(min(b, cap) for b in budgets) < target:
            low = cap
        else:
            high = cap
    return high


def main(ears_count=70, singing_hours=58.0, total_hours=121.4):
    with open(os.path.join(SOURCE, "speakers.json"), encoding="utf-8") as handle:
        singers = [s for s in json.load(handle) if s["source"] != "ears"]
    os.makedirs(OUTPUT, exist_ok=True)
    rng = random.Random(SEED)
    entries = []

    for speaker in pick_ears(ears_count):
        folder = os.path.join(OUTPUT, f"{len(entries)}_ears-{speaker}")
        fetch_ears(speaker, folder)
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".flac")]
        entries.append({"sid": len(entries), "source": "ears", "singer": speaker,
                        "speech": True, "folder": os.path.basename(folder), "hours": hours(files)})
    speech = sum(e["hours"] for e in entries)
    singing_hours = min(singing_hours, total_hours - speech)

    songs = []
    for singer in singers:
        source = os.path.join(SOURCE, singer["folder"])
        files = sorted(os.path.join(source, f) for f in os.listdir(source))
        songs.append((singer, [(f, sf.info(f).duration / 3600) for f in files]))
    cap = cap_for([sum(h for _, h in files) for _, files in songs], singing_hours)

    for singer, files in songs:
        rng.shuffle(files)
        chosen, total = [], 0.0
        for path, length in files:
            if total >= cap:
                break
            chosen.append(path)
            total += length
        folder = os.path.join(OUTPUT, f"{len(entries)}_{singer['source']}-{singer['singer']}")
        os.makedirs(folder, exist_ok=True)
        for path in chosen:
            target = os.path.join(folder, os.path.basename(path))
            if not os.path.exists(target):
                os.link(path, target)
        entries.append({"sid": len(entries), "source": singer["source"], "singer": singer["singer"],
                        "songs": len(chosen), "speech": False, "folder": os.path.basename(folder),
                        "hours": total})

    with open(os.path.join(OUTPUT, "speakers.json"), "w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2, ensure_ascii=False)
    singing = sum(e["hours"] for e in entries if not e["speech"])
    print(f"{len(entries)} speakers, {speech + singing:.1f} h: "
          f"{speech:.1f} h speech, {singing:.1f} h singing (cap {cap:.2f} h per singer).")


if __name__ == "__main__":
    main(*(float(a) if "." in a else int(a) for a in sys.argv[1:]))
