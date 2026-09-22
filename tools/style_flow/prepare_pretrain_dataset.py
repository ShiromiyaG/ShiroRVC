"""Lay out GTSinger's singing audio (from its zip archives) with M4Singer
and, optionally, a few EARS speakers as one multi-speaker RVC dataset for
the style pretrain.

    python tools/style_flow/prepare_pretrain_dataset.py \\
        --gtsinger-archives assets/datasets/GTSinger/compressed --download-ears 5 \\
        --out assets/datasets/style_pretrain

Speakers come out as ``<sid>_<source>-<singer>/``: EARS first (so they are
``0..n-1``, the speech ids in ``pretrain.yaml``), then M4Singer, then
GTSinger.  GTSinger's ``Paired_Speech_Group`` is skipped.  Consecutive
segments of one song (and one technique group) are joined with a short
silence into one file, because the segments are mostly shorter than the 5 s
the style clips need.  ``speakers.json`` records the mapping.  Rerunning
skips files that already exist, so languages can be added as their archives
arrive; the speaker ids after them shift, so start a fresh ``--out`` (and
experiment) when that happens.
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys
import zipfile
from collections import defaultdict

import numpy as np
import soundfile as sf

sys.path.append(os.getcwd())

from rvc.lib.terminal import info, success, track, warning

SUBTYPE = "PCM_24"
EARS_URL = "https://github.com/facebookresearch/ears_dataset/releases/download/dataset/p{:03d}.zip"


def safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name).strip("_") or "x"


def _member_name(member: zipfile.ZipInfo) -> str:
    """The member's real name: these archives store UTF-8 names without the
    flag that says so, which ``zipfile`` then reads as cp437."""
    name = member.filename
    if not member.flag_bits & 0x800:
        try:
            name = name.encode("cp437").decode("utf-8")
        except UnicodeError:
            pass
    return name


def extract_gtsinger(archive_dir: str, dest: str) -> str:
    """Extract the singing ``.wav`` files of every GTSinger zip in
    ``archive_dir`` into ``dest``, skipping speech groups, macOS metadata and
    files already extracted."""
    archives = sorted(glob.glob(os.path.join(archive_dir, "*.zip")))
    if not archives:
        raise SystemExit(f"No GTSinger .zip in {archive_dir}.")
    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            members = []
            for member in zf.infolist():
                name = _member_name(member)
                parts = name.split("/")
                if (
                    not name.lower().endswith(".wav")
                    or parts[0] == "__MACOSX"
                    or parts[-1].startswith("._")
                    or "Paired_Speech_Group" in parts
                ):
                    continue
                target = os.path.join(dest, *parts)
                if os.path.exists(target) and os.path.getsize(target) == member.file_size:
                    continue
                members.append((member, target))
            info(f"{os.path.basename(archive)}: {len(members)} singing files to extract.", tag="[DATA]")
            for member, target in track(members, total=len(members), description=f"Extracting {os.path.basename(archive)}"):
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with zf.open(member) as src, open(target + ".part", "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
                os.replace(target + ".part", target)
    return dest


def download_ears(cache: str, count: int) -> str:
    import urllib.request

    os.makedirs(cache, exist_ok=True)
    for i in range(1, count + 1):
        speaker = f"p{i:03d}"
        if os.path.isdir(os.path.join(cache, speaker)):
            continue
        archive = os.path.join(cache, speaker + ".zip")
        info(f"Downloading EARS {speaker}.", tag="[DATA]")
        urllib.request.urlretrieve(EARS_URL.format(i), archive + ".part")
        os.replace(archive + ".part", archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(cache)
        os.remove(archive)
    return cache


def join_segments(paths, out_path, gap_s):
    """Concatenate ``paths`` (one sample rate, mono or not) with ``gap_s`` of
    silence between them, as 24-bit FLAC."""
    parts, sr = [], None
    for path in paths:
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr is None:
            sr = rate
        elif rate != sr:
            warning(f"{path} is {rate} Hz, not {sr}; skipped.", tag="[DATA]")
            continue
        if parts:
            parts.append(np.zeros(int(gap_s * sr), dtype=np.float32))
        parts.append(audio)
    if parts:
        # Written aside and renamed, so an interrupted run leaves no half file.
        sf.write(out_path + ".part", np.concatenate(parts), sr, format="FLAC", subtype=SUBTYPE)
        os.replace(out_path + ".part", out_path)


def m4singer_songs(root):
    """``{singer: {song: [segment paths]}}`` from ``<Singer>#<song>/NNNN.wav``."""
    out = defaultdict(dict)
    for folder in sorted(os.listdir(root)):
        path = os.path.join(root, folder)
        if not os.path.isdir(path) or "#" not in folder:
            continue
        singer, song = folder.split("#", 1)
        segments = sorted(glob.glob(os.path.join(path, "*.wav")))
        if segments:
            out[singer][song] = segments
    return out


def gtsinger_songs(root):
    """``{singer: {song_group: [segment paths]}}`` from
    ``<Language>/<Singer>/<Technique>/<song>/<Group>/NNNN.wav``."""
    out = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(root, "*", "*", "*", "*", "*_Group", "*.wav"))):
        parts = path.split(os.sep)
        singer, technique, song, group = parts[-5], parts[-4], parts[-3], parts[-2]
        if group == "Paired_Speech_Group":
            continue
        out[singer].setdefault(f"{technique}_{song}_{group}", []).append(path)
    return out


def ears_speakers(root, count):
    """``{speaker: {file: [path]}}`` for the first ``count`` EARS speakers,
    without the nonverbal clips (laughs, coughs...), whose F0 says nothing
    about phrasing."""
    speakers = sorted(d for d in os.listdir(root) if re.fullmatch(r"p\d{3}", d))[:count]
    if len(speakers) < count:
        warning(f"Only {len(speakers)} EARS speakers found in {root}.", tag="[DATA]")
    return {
        s: {
            os.path.splitext(os.path.basename(p))[0]: [p]
            for p in sorted(glob.glob(os.path.join(root, s, "**", "*.wav"), recursive=True))
            if not os.path.basename(p).startswith("nonverbal")
        }
        for s in speakers
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join("assets", "datasets", "style_pretrain"))
    parser.add_argument("--m4singer", default=os.path.join("assets", "datasets", "m4singer"))
    parser.add_argument(
        "--gtsinger-archives", default=os.path.join("assets", "datasets", "GTSinger", "compressed"),
        help="Folder of GTSinger .zip archives (one per language).",
    )
    parser.add_argument(
        "--gtsinger", default=os.path.join("assets", "datasets", "GTSinger", "extracted"),
        help="Where the archives are extracted (or an already extracted copy).",
    )
    parser.add_argument("--cache", default=os.path.join("assets", "datasets", "_downloads"))
    ears = parser.add_mutually_exclusive_group()
    ears.add_argument("--ears", help="Existing EARS folder (p001/, p002/, ...).")
    ears.add_argument("--download-ears", type=int, default=0, metavar="N", help="Download the first N EARS speakers.")
    parser.add_argument("--ears-speakers", type=int, default=5, help="How many EARS speakers to use from --ears.")
    parser.add_argument("--gap", type=float, default=0.5, help="Seconds of silence between joined segments.")
    args = parser.parse_args()

    sources = []
    if args.ears or args.download_ears:
        root = args.ears or download_ears(os.path.join(args.cache, "EARS"), args.download_ears)
        count = args.download_ears or args.ears_speakers
        sources.append(("ears", ears_speakers(root, count)))
    if args.m4singer and os.path.isdir(args.m4singer):
        sources.append(("m4singer", m4singer_songs(args.m4singer)))
    else:
        warning(f"No M4Singer at {args.m4singer}; continuing without it.", tag="[DATA]")
    if os.path.isdir(args.gtsinger_archives):
        extract_gtsinger(args.gtsinger_archives, args.gtsinger)
    if os.path.isdir(args.gtsinger):
        sources.append(("gtsinger", gtsinger_songs(args.gtsinger)))
    else:
        warning(f"No GTSinger at {args.gtsinger_archives} or {args.gtsinger}; continuing without it.", tag="[DATA]")

    jobs, mapping, sid = [], [], 0
    for source, singers in sources:
        for singer in sorted(singers):
            folder = os.path.join(args.out, f"{sid}_{source}-{safe(singer)}")
            os.makedirs(folder, exist_ok=True)
            songs = singers[singer]
            mapping.append({"sid": sid, "source": source, "singer": singer, "songs": len(songs),
                            "speech": source == "ears", "folder": os.path.basename(folder)})
            for song, paths in songs.items():
                jobs.append((paths, os.path.join(folder, safe(song) + ".flac")))
            sid += 1

    expected = {m["folder"] for m in mapping}
    stale = [d for d in os.listdir(args.out) if os.path.isdir(os.path.join(args.out, d)) and d not in expected]
    if stale:
        warning(f"{args.out} has speaker folders from an earlier layout ({', '.join(sorted(stale)[:3])}...); "
                "delete them, or RVC will read them as extra speakers.", tag="[DATA]")

    def done(path):
        # Files from an earlier run at another bit depth are redone.
        return os.path.exists(path) and sf.info(path).subtype == SUBTYPE

    todo = [(p, o) for p, o in jobs if not done(o)]
    info(f"{sid} speakers, {len(jobs)} files, {len(todo)} to write.", tag="[DATA]")
    for paths, out_path in track(todo, total=len(todo), description="Joining segments"):
        join_segments(paths, out_path, args.gap)

    with open(os.path.join(args.out, "speakers.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    speech = [m["sid"] for m in mapping if m["speech"]]
    success(f"{sid} speakers in {args.out}.", tag="[DATA]")
    if speech:
        info(f'Speech speakers: "{speech[0]}-{speech[-1]}" (data.speech_speakers in pretrain.yaml).', tag="[DATA]")
    else:
        warning('No speech speakers: set data.speech_speakers to [] in pretrain.yaml.', tag="[DATA]")


if __name__ == "__main__":
    main()
