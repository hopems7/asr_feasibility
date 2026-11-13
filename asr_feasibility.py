#!/usr/bin/env python3
"""
ASR data feasibility check
==========================

Scans a folder of audio and video files and produces an AGGREGATE report that
helps judge whether the recordings could be useful for fine-tuning a speech
recognition model on children's speech.

Privacy design
--------------
* Nothing is uploaded or sent anywhere. The tool only reads files and writes a
  report to a local folder.
* The shareable report contains only totals and distributions. It contains no
  file names, folder names, transcripts, audio, or per-file details.
* An optional per-file table (--per-file) is written to a separate file marked
  LOCAL ONLY, for the data owner's own use.
* The pitch statistics are coarse frequency ranges across the whole collection.
  They do not identify anyone and are not used to recognize individuals.
  The tool deliberately does not count speakers or estimate ages, since both
  would require biometric processing of children's voices.
* Caption files (.srt, .vtt and similar) are opened only to read their cue
  timestamps. The caption text is discarded immediately and never stored.
* Dataset manifests (JSON, JSON Lines, CSV, TSV) are read to find which
  recordings have a transcription and how long it is. The text, speaker IDs and
  locations are never stored or reported; speakers and locations are only
  counted, and age and gender groups under 5 recordings are combined.

Interruptions
-------------
Progress is saved after every file. If the run is interrupted (power cut,
computer shutdown, closing the window, Ctrl+C), run the tool again on the same
folder with the same settings and it continues where it left off. The progress
file stores only hashed file IDs and statistics, never file names, and is
deleted automatically when the run finishes.

Usage
-----
    python asr_feasibility.py                    # opens a folder picker
    python asr_feasibility.py "D:/Recordings"    # scan a specific folder
    python asr_feasibility.py "D:/Recordings" --max-minutes-per-file 10
    python asr_feasibility.py "D:/Recordings" --fresh   # ignore saved progress
    python asr_feasibility.py --help

Requirements: Python 3.9+, numpy, and ffmpeg (either installed on the system
or via the imageio-ffmpeg package). Optional: webrtcvad-wheels (better speech
detection), faster-whisper (language identification).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

TOOL_VERSION = "1.6.0"

# ---------------------------------------------------------------------------
# Analysis constants
# ---------------------------------------------------------------------------
SR = 16000                      # Whisper's input sample rate
FRAME = 480                     # 30 ms analysis frames (also valid for webrtcvad)
CHUNK_SECONDS = 30              # decode in 30 s chunks (Whisper's window size)
CHUNK = SR * CHUNK_SECONDS      # divisible by FRAME
CLIP_LEVEL = 32700              # int16 sample magnitude treated as clipped

PITCH_FMIN, PITCH_FMAX = 70.0, 700.0
MIN_LAG = int(SR // PITCH_FMAX)
MAX_LAG = int(math.ceil(SR / PITCH_FMIN))
YIN_THRESHOLD = 0.15            # standard YIN absolute threshold
VOICED_MAX_CMNDF = 0.35         # above this, a frame is treated as unvoiced
PITCH_EVERY = 2                 # estimate pitch on every 2nd frame (speed)
F0_BIN_EDGES = np.arange(50, 750, 25, dtype=np.float64)
CHILD_F0 = 250.0                # voiced frames at or above this count as child-range

BAND_HZ = 250                   # spectrum summarized in 250 Hz bands up to 8 kHz
N_BANDS = (SR // 2) // BAND_HZ
SPEC_N = 512
_HANN = np.hanning(FRAME).astype(np.float32)
_BAND_MATRIX = np.zeros((SPEC_N // 2 + 1, N_BANDS), dtype=np.float32)
_BAND_MATRIX[np.arange(SPEC_N // 2 + 1),
             np.minimum((np.fft.rfftfreq(SPEC_N, 1 / SR) // BAND_HZ).astype(int), N_BANDS - 1)] = 1
BANDWIDTH_DROP_DB = 50.0        # bandwidth = highest band within 50 dB of the 250-3000 Hz level

GAP_MERGE_FRAMES = 10           # pauses shorter than 300 ms don't split a speech run
MIN_RUN_FRAMES = 7              # speech runs shorter than ~200 ms are ignored

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus",
             ".wma", ".amr", ".aiff", ".aif", ".caf", ".webm_audio"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".m4v", ".webm", ".mpg",
             ".mpeg", ".3gp", ".3g2", ".flv", ".mts", ".m2ts", ".ts", ".vob",
             ".dv", ".ogv"}
CAPTION_EXT = {".srt", ".vtt", ".ass", ".ssa", ".sbv", ".ttml", ".dfxp"}   # timed captions
DOC_EXT = {".sub", ".txt", ".doc", ".docx", ".rtf", ".pdf", ".odt",
           ".eaf", ".textgrid", ".trs", ".cha", ".gdoc"}                   # .gdoc: Google Docs                            # possible transcripts
TRANSCRIPT_EXT = CAPTION_EXT | DOC_EXT
MANIFEST_EXT = {".json", ".jsonl", ".csv", ".tsv"}                       # dataset manifests
ARCHIVE_EXT = {".zip", ".tar", ".gz", ".tgz", ".xz", ".bz2", ".7z", ".rar", ".zst"}

NO_WINDOW = 0x08000000 if os.name == "nt" else 0   # hide console pop-ups on Windows

# ---------------------------------------------------------------------------
# Report buckets (upper bound, label)
# ---------------------------------------------------------------------------
INF = float("inf")
DURATION_BUCKETS = [(30, "under 30 s"), (120, "30 s - 2 min"), (600, "2 - 10 min"),
                    (3600, "10 - 60 min"), (INF, "over 60 min")]
SNR_BUCKETS = [(5, "below 5 dB (very noisy)"), (10, "5 - 10 dB (noisy)"),
               (20, "10 - 20 dB (moderate)"), (30, "20 - 30 dB (good)"),
               (INF, "30 dB and above (clean)")]
SPEECH_RATIO_BUCKETS = [(0.10, "under 10% speech"), (0.30, "10 - 30% speech"),
                        (0.60, "30 - 60% speech"), (INF, "over 60% speech")]
LEVEL_BUCKETS = [(-45, "very quiet (below -45 dBFS)"), (-35, "quiet (-45 to -35 dBFS)"),
                 (-15, "typical (-35 to -15 dBFS)"), (INF, "loud (above -15 dBFS)")]
CLIP_BUCKETS = [(0.0, "none"), (1e-4, "trace (under 0.01% of samples)"),
                (1e-3, "some (0.01 - 0.1%)"), (INF, "frequent (over 0.1%)")]
F0_BANDS = [(165, "below 165 Hz (typical adult male range)"),
            (250, "165 - 250 Hz (typical adult female range; overlaps older children)"),
            (400, "250 - 400 Hz (typical child range)"),
            (INF, "above 400 Hz (very young children, shouting/singing, or noise)")]
RUN_BUCKETS = [(2, "under 2 s"), (5, "2 - 5 s"), (10, "5 - 10 s"),
               (30, "10 - 30 s"), (INF, "over 30 s (needs alignment or finer splitting)")]
BANDWIDTH_BUCKETS = [(5000, "narrowband, under 5 kHz (phone-call or voice-note quality)"),
                     (6500, "reduced, 5 - 6.5 kHz"),
                     (INF, "full, 6.5 kHz and above")]
CHILD_MIX_BUCKETS = [(0.30, "mostly adult-range pitch (under 30% child-range)"),
                     (0.70, "mixed (30 - 70% child-range)"),
                     (INF, "mostly child-range pitch (70% or more)")]
BITRATE_BUCKETS = [(64, "under 64 kb/s"), (128, "64 - 128 kb/s"),
                   (256, "128 - 256 kb/s"), (INF, "256 kb/s and above")]


def bucket_label(value, buckets):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown / not measurable"
    for upper, label in buckets:
        if value <= upper if upper == 0.0 else value < upper:
            return label
    return buckets[-1][1]


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------
def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # bundles a static ffmpeg binary
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def find_ffprobe(ffmpeg):
    exe = shutil.which("ffprobe")
    if exe:
        return exe
    if ffmpeg:
        cand = Path(ffmpeg).with_name("ffprobe" + (".exe" if os.name == "nt" else ""))
        if cand.exists():
            return str(cand)
    return None


def _empty_probe():
    return {"ok": False, "has_audio": False, "has_video": False, "has_subtitles": False,
            "audio_codec": None, "sample_rate": None, "channels": None,
            "bitrate_kbps": None, "duration_s": None, "meta_year": None}


def _plausible_year(text):
    """Year from a creation_time tag, ignoring unset camera clocks and epoch defaults."""
    m = re.match(r"\s*(\d{4})-", text or "")
    if not m:
        return None
    y = int(m.group(1))
    return y if 1995 <= y <= dt.date.today().year + 1 else None


def probe_ffprobe(path, ffprobe):
    info = _empty_probe()
    out = subprocess.run([ffprobe, "-v", "error", "-print_format", "json",
                          "-show_format", "-show_streams", str(path)],
                         capture_output=True, timeout=180, creationflags=NO_WINDOW)
    data = json.loads(out.stdout.decode("utf-8", "replace") or "{}")
    if not data.get("streams") and not data.get("format"):
        return info
    info["ok"] = True
    fmt = data.get("format", {})
    tags = {k.lower(): v for k, v in (fmt.get("tags") or {}).items()}
    info["meta_year"] = _plausible_year(tags.get("creation_time") or tags.get("date"))
    try:
        info["duration_s"] = float(fmt.get("duration"))
    except (TypeError, ValueError):
        pass
    for s in data.get("streams", []):
        ctype = s.get("codec_type")
        if ctype == "audio" and not info["has_audio"]:
            info["has_audio"] = True
            info["audio_codec"] = s.get("codec_name")
            try:
                info["sample_rate"] = int(s.get("sample_rate"))
            except (TypeError, ValueError):
                pass
            info["channels"] = s.get("channels")
            try:
                info["bitrate_kbps"] = int(s.get("bit_rate")) // 1000
            except (TypeError, ValueError):
                pass
            if info["duration_s"] is None:
                try:
                    info["duration_s"] = float(s.get("duration"))
                except (TypeError, ValueError):
                    pass
        elif ctype == "video" and not s.get("disposition", {}).get("attached_pic", 0):
            info["has_video"] = True
        elif ctype == "subtitle":
            info["has_subtitles"] = True
    return info


_RE_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_RE_AUDIO = re.compile(r"Stream #\S+.*?:\s*Audio:\s*([A-Za-z0-9_]+)(.*)")
_RE_HZ = re.compile(r"(\d+)\s*Hz")
_RE_KBS = re.compile(r"(\d+)\s*kb/s")


def probe_ffmpeg_stderr(path, ffmpeg):
    """Fallback when ffprobe is unavailable: parse `ffmpeg -i` output."""
    info = _empty_probe()
    out = subprocess.run([ffmpeg, "-hide_banner", "-nostdin", "-i", str(path)],
                         capture_output=True, timeout=180, creationflags=NO_WINDOW)
    text = out.stderr.decode("utf-8", "replace")
    if "Input #0" not in text:
        return info
    info["ok"] = True
    cm = re.search(r"creation_time\s*:\s*(\S+)", text)
    if cm:
        info["meta_year"] = _plausible_year(cm.group(1))
    m = _RE_DURATION.search(text)
    if m:
        h, mi, s = m.groups()
        info["duration_s"] = int(h) * 3600 + int(mi) * 60 + float(s)
    for line in text.splitlines():
        if "Stream #" not in line:
            continue
        if "Audio:" in line and not info["has_audio"]:
            am = _RE_AUDIO.search(line)
            info["has_audio"] = True
            if am:
                info["audio_codec"] = am.group(1)
                rest = am.group(2)
                hz = _RE_HZ.search(rest)
                if hz:
                    info["sample_rate"] = int(hz.group(1))
                if "mono" in rest:
                    info["channels"] = 1
                elif "stereo" in rest:
                    info["channels"] = 2
                else:
                    ch = re.search(r"(\d+)\s*channels", rest)
                    if ch:
                        info["channels"] = int(ch.group(1))
                kb = _RE_KBS.search(rest)
                if kb:
                    info["bitrate_kbps"] = int(kb.group(1))
        elif "Video:" in line and "attached pic" not in line:
            info["has_video"] = True
        elif "Subtitle:" in line:
            info["has_subtitles"] = True
    return info


def decode_chunks(path, ffmpeg, start=None, max_seconds=None, status=None):
    """Yield int16 mono 16 kHz chunks of the first audio stream."""
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path), "-map", "0:a:0", "-vn", "-sn", "-dn",
            "-ac", "1", "-ar", str(SR), "-f", "s16le"]
    if max_seconds:
        cmd += ["-t", f"{max_seconds:.3f}"]
    cmd += ["-"]
    errf = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf,
                            stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
    try:
        while True:
            data = proc.stdout.read(CHUNK * 2)
            if not data:
                break
            if len(data) % 2:
                data = data[:-1]
            yield np.frombuffer(data, dtype=np.int16)
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        try:
            rc = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait()
        if status is not None:
            status["returncode"] = rc
            errf.seek(0)
            status["stderr"] = errf.read()[-400:].decode("utf-8", "replace")
        errf.close()


# ---------------------------------------------------------------------------
# Signal analysis
# ---------------------------------------------------------------------------
def yin_f0(windows):
    """Vectorized YIN pitch estimate. windows: (n, FRAME + MAX_LAG) float32.
    Returns f0 in Hz, NaN where the frame looks unvoiced."""
    n = windows.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float32)
    W = FRAME
    x = windows.astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    L = x.shape[1]
    nfft = 1 << int(math.ceil(math.log2(L + W)))
    X = np.fft.rfft(x, nfft, axis=1)
    X0 = np.fft.rfft(x[:, :W], nfft, axis=1)
    cross = np.fft.irfft(np.conj(X0) * X, nfft, axis=1)[:, :MAX_LAG + 1]
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(x * x, axis=1)], axis=1)
    taus = np.arange(MAX_LAG + 1)
    e_tau = cs[:, taus + W] - cs[:, taus]
    d = np.maximum(e_tau[:, :1] + e_tau - 2.0 * cross, 0.0)
    d[:, 0] = 0.0
    cmndf = np.ones_like(d)
    cum = np.cumsum(d[:, 1:], axis=1)
    cmndf[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(cum, 1e-12)

    seg = cmndf[:, MIN_LAG:MAX_LAG]
    rows = np.arange(n)
    # Adaptive threshold: in noise the true period may not dip below the
    # standard YIN threshold, and taking the global minimum then tends to pick
    # a subharmonic (octave-down error). Accept the first dip that comes
    # close to the global minimum instead.
    gmin = seg.min(axis=1, keepdims=True)
    thr = np.minimum(np.maximum(YIN_THRESHOLD, gmin + 0.12), VOICED_MAX_CMNDF)
    below = seg < thr
    has = below.any(axis=1)
    idx = np.where(has, below.argmax(axis=1), seg.argmin(axis=1))
    for _ in range(30):  # walk down to the local minimum
        nxt = np.minimum(idx + 1, seg.shape[1] - 1)
        better = seg[rows, nxt] < seg[rows, idx]
        if not better.any():
            break
        idx = np.where(better, nxt, idx)
    val = seg[rows, idx]

    # parabolic interpolation for sub-sample lag precision
    tau = idx.astype(np.float64)
    inner = (idx > 0) & (idx < seg.shape[1] - 1)
    a = seg[rows, np.maximum(idx - 1, 0)]
    b = val
    c = seg[rows, np.minimum(idx + 1, seg.shape[1] - 1)]
    denom = a - 2 * b + c
    ok = inner & (np.abs(denom) > 1e-12)
    tau[ok] += 0.5 * (a[ok] - c[ok]) / denom[ok]
    tau += MIN_LAG

    f0 = (SR / tau).astype(np.float32)
    voiced = (val < VOICED_MAX_CMNDF) & (e_tau[:, 0] / W > 1e-7)
    f0[~voiced] = np.nan
    return f0


def analyze_file(path, ffmpeg, vad_method, max_seconds, do_pitch, keep_frames=False):
    """Decode a file and compute frame-level statistics. Returns a dict."""
    vad = None
    if vad_method == "webrtc":
        import webrtcvad
        vad = webrtcvad.Vad(2)

    dbs, f0s, vads, chunk_frames = [], [], [], []
    band_rows, band_frames, frame_pos = [], [], 0
    total_samples = 0
    clipped = 0
    status = {}
    for x in decode_chunks(path, ffmpeg, max_seconds=max_seconds, status=status):
        if STOP.is_set():
            raise _Stopped()
        n = len(x) // FRAME
        if n == 0:
            continue
        xi = x[:n * FRAME]
        total_samples += len(xi)
        clipped += int(np.count_nonzero(np.abs(xi.astype(np.int32)) >= CLIP_LEVEL))
        xf = xi.astype(np.float32) / 32768.0
        fr = xf.reshape(n, FRAME)
        dbs.append(10.0 * np.log10(np.mean(fr * fr, axis=1) + 1e-10))
        if vad is not None:
            vads.append(np.fromiter(
                (vad.is_speech(xi[i * FRAME:(i + 1) * FRAME].tobytes(), SR) for i in range(n)),
                dtype=bool, count=n))
        f0 = np.full(n, np.nan, dtype=np.float32)
        if do_pitch:
            starts = np.arange(0, n, PITCH_EVERY) * FRAME
            starts = starts[starts + FRAME + MAX_LAG <= len(xf)]
            if len(starts):
                win = sliding_window_view(xf, FRAME + MAX_LAG)[starts]
                f0[starts // FRAME] = yin_f0(win)
        f0s.append(f0)
        # spectrum in 250 Hz bands on every 2nd frame, for bandwidth estimation
        sidx = np.arange(0, n, PITCH_EVERY)
        spec = np.abs(np.fft.rfft(fr[sidx] * _HANN, SPEC_N, axis=1)) ** 2
        band_rows.append((spec @ _BAND_MATRIX).astype(np.float32))
        band_frames.append(sidx + frame_pos)
        frame_pos += n
        chunk_frames.append(n)

    if total_samples == 0:
        failed = status.get("returncode", 0) != 0
        return {"status": "decode_error" if failed else "empty",
                "error_detail": status.get("stderr", "")}

    db = np.concatenate(dbs)
    f0 = np.concatenate(f0s)
    if vad is not None:
        # webrtcvad readily labels steady background noise as speech, so also
        # require frames to sit a little above the file's noise floor.
        floor = float(np.percentile(db, 10))
        speech = np.concatenate(vads) & (db > -60.0) & (db > floor + 3.0)
    else:
        # Adaptive energy threshold: frames clearly above the file's noise floor.
        # If the file's level barely varies, treat it as steady noise/silence.
        p10, p90 = float(np.percentile(db, 10)), float(np.percentile(db, 90))
        if p90 - p10 < 4.0:
            speech = np.zeros(len(db), dtype=bool)
        else:
            thresh = max(p10 + min(9.0, 0.5 * (p90 - p10)), -55.0)
            speech = db > thresh
        if len(speech) > 4:  # short hangover to bridge brief dips inside words
            speech = np.convolve(speech.astype(np.float32), np.ones(5), "same") > 0

    power = 10.0 ** (db / 10.0)
    n_speech = int(speech.sum())
    n_noise = int((~speech).sum())
    snr = None
    if n_speech >= 10 and n_noise >= 10:
        snr = 10.0 * math.log10(float(np.median(power[speech])) /
                                max(float(np.median(power[~speech])), 1e-10))
    speech_level = float(10 * math.log10(np.mean(power[speech]))) if n_speech else None

    voiced = speech & ~np.isnan(f0)
    f0_vals = f0[voiced]
    hist, _ = np.histogram(f0_vals, bins=F0_BIN_EDGES)

    # effective audio bandwidth from the long-term spectrum of speech frames
    bandwidth = None
    bidx = np.concatenate(band_frames)
    sel = speech[bidx] & (db[bidx] > -60.0)
    if sel.sum() >= 50:
        ltas = np.concatenate(band_rows)[sel].mean(axis=0)
        ref = 10 * math.log10(float(ltas[1:12].mean()) + 1e-20)      # 250 - 3000 Hz
        ltas_db = 10 * np.log10(ltas + 1e-20)
        within = np.nonzero(ltas_db >= ref - BANDWIDTH_DROP_DB)[0]
        if len(within):
            bandwidth = int((within.max() + 1) * BAND_HZ)

    run_s, pauses = speech_runs(speech)
    run_counts = [0] * len(RUN_BUCKETS)
    run_time = [0.0] * len(RUN_BUCKETS)
    for length in run_s:
        i = [b[1] for b in RUN_BUCKETS].index(bucket_label(float(length), RUN_BUCKETS))
        run_counts[i] += 1
        run_time[i] += float(length)

    # per-chunk speech ratio, used to pick windows for optional language ID
    chunk_ratios, pos = [], 0
    for n in chunk_frames:
        chunk_ratios.append(float(speech[pos:pos + n].mean()))
        pos += n

    frames = None
    if keep_frames:   # for drive_explore.py: frame-level arrays, used in memory only
        frames = {"speech": speech, "f0": f0, "db": db,
                  "bands": np.concatenate(band_rows), "band_frames": bidx}
    return {
        "_frames": frames,
        "status": "ok",
        "analyzed_s": total_samples / SR,
        "speech_ratio": n_speech / len(speech),
        "snr_db": snr,
        "speech_level_dbfs": speech_level,
        "clip_fraction": clipped / total_samples,
        "f0_hist": hist.tolist(),
        "voiced_frames": int(voiced.sum()),
        "median_f0": float(np.median(f0_vals)) if len(f0_vals) >= 20 else None,
        "chunk_ratios": chunk_ratios,
        "bandwidth_hz": bandwidth,
        "run_counts": run_counts,
        "run_time_s": [round(t, 2) for t in run_time],
        "pauses": int(pauses),
        "decode_warning": status.get("returncode", 0) != 0,
    }


def speech_runs(mask):
    """Lengths (s) of continuous speech runs, bridging pauses under 300 ms,
    and the number of pauses of 300 ms or more between them."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    if len(starts) == 0:
        return np.empty(0), 0
    keep = (starts[1:] - ends[:-1]) >= GAP_MERGE_FRAMES
    run_starts = np.concatenate([[starts[0]], starts[1:][keep]])
    run_ends = np.concatenate([ends[:-1][keep], [ends[-1]]])
    lengths = run_ends - run_starts
    lengths = lengths[lengths >= MIN_RUN_FRAMES]
    return lengths * FRAME / SR, max(len(lengths) - 1, 0)


# ---------------------------------------------------------------------------
# Interruption and resume
# ---------------------------------------------------------------------------
STOP = threading.Event()        # set on Ctrl+C so workers stop promptly


class _Stopped(Exception):
    pass


def file_key(path, root):
    """Stable ID for a file across runs: a hash of its relative path, size and
    modification time. If a file is changed, it gets a new key and is redone.
    Only the hash is stored, so no file or folder names reach the checkpoint."""
    st = path.stat()
    raw = f"{path.relative_to(root).as_posix()}|{st.st_size}|{st.st_mtime_ns}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]


class Checkpoint:
    """Append-only progress log so an interrupted run can pick up where it left off.

    Each finished step is written as one JSON line and forced to disk at once,
    so a power cut loses at most the files being processed at that moment.
    A run resumes only if the folder and the analysis settings are unchanged;
    otherwise it starts fresh. The log is deleted after a successful run."""

    FILENAME = "resume_checkpoint__LOCAL_ONLY.jsonl"

    def __init__(self, out_dir, settings, fresh=False):
        self.path = out_dir / self.FILENAME
        self.settings = settings
        self.probes, self.analyses, self.lids, self.extra = {}, {}, {}, {}
        self.resumed, self.reason = False, None
        if self.path.exists() and not fresh:
            self._load()
        elif self.path.exists():
            self.reason = "previous progress discarded (--fresh)"
        if self.resumed:
            self._fh = open(self.path, "a", encoding="utf-8")
        else:
            self.probes, self.analyses, self.lids, self.extra = {}, {}, {}, {}
            self._fh = open(self.path, "w", encoding="utf-8")
            self._write({"type": "header", "settings": settings})

    def _load(self):
        data = self.path.read_bytes()
        cut = data.rfind(b"\n") + 1
        if cut < len(data):  # last line was cut off mid-write; drop it
            with open(self.path, "r+b") as fh:
                fh.truncate(cut)
            data = data[:cut]
        header = None
        for line in data.decode("utf-8", "replace").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if header is None:
                if rec.get("type") != "header":
                    break
                header = rec
                continue
            if rec.get("type") == "probe":
                self.probes[rec["k"]] = rec["v"]
            elif rec.get("type") == "analysis":
                self.analyses[rec["k"]] = rec["v"]
            elif rec.get("type") == "lid":
                self.lids[rec["k"]] = rec["v"]
            elif rec.get("type"):
                self.extra.setdefault(rec["type"], {})[rec["k"]] = rec["v"]
        if header is None:
            self.reason = "previous progress file was unreadable; starting fresh"
        elif header.get("settings") != self.settings:
            self.reason = "folder or settings changed since the last run; starting fresh"
        else:
            self.resumed = True

    def _write(self, obj):
        self._fh.write(json.dumps(obj, separators=(",", ":")) + "\n")
        self._fh.flush()
        try:
            os.fsync(self._fh.fileno())
        except OSError:
            pass

    def save_probe(self, key, info):
        self.probes[key] = info
        self._write({"type": "probe", "k": key, "v": info})

    def save_analysis(self, key, result):
        result = {k: v for k, v in result.items() if k not in ("error_detail", "_frames")}
        self.analyses[key] = result
        self._write({"type": "analysis", "k": key, "v": result})

    def save(self, kind, key, value):
        self.extra.setdefault(kind, {})[key] = value
        self._write({"type": kind, "k": key, "v": value})

    def save_lid(self, key, result):
        self.lids[key] = result
        self._write({"type": "lid", "k": key, "v": result})

    def close(self, delete=False):
        try:
            self._fh.close()
        except Exception:
            pass
        if delete:
            try:
                self.path.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------
def scan_folder(root, exclude):
    media, labels, manifests, other_ext = [], [], [], Counter()
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
        dpath = Path(dirpath)
        dirnames[:] = [d for d in dirnames
                       if not d.startswith(".") and (dpath / d).resolve() != exclude]
        for fn in filenames:
            if fn.startswith("."):
                continue
            p = dpath / fn
            ext = p.suffix.lower()
            if ext in AUDIO_EXT or ext in VIDEO_EXT:
                media.append(p)
            else:
                other_ext[ext or "(no extension)"] += 1
                if ext in TRANSCRIPT_EXT:
                    labels.append(p)
                elif ext in MANIFEST_EXT:
                    manifests.append(p)
    return sorted(media), sorted(labels), sorted(manifests), other_ext


def file_year(path):
    """Earliest of the file's modification and (where available) creation time."""
    st = path.stat()
    t = min(st.st_mtime, getattr(st, "st_birthtime", st.st_mtime))
    y = dt.datetime.fromtimestamp(t).year
    return y if 1995 <= y <= dt.date.today().year + 1 else None


# ---------------------------------------------------------------------------
# Existing labels: caption timestamps only (caption text is never kept)
# ---------------------------------------------------------------------------
_TS = r"(?:(\d+):)?(\d{1,2}):(\d{2})[,.](\d{1,3})"
_RE_CUE = re.compile(_TS + r"\s*-->\s*" + _TS)                         # srt, vtt
_RE_ASS = re.compile(r"^Dialogue:\s*[^,]*,(\d+):(\d{2}):(\d{2})\.(\d{1,3}),"
                     r"(\d+):(\d{2}):(\d{2})\.(\d{1,3}),", re.M)      # ass, ssa
_RE_SBV = re.compile(r"^(\d+):(\d{2}):(\d{2})\.(\d{1,3}),(\d+):(\d{2}):(\d{2})\.(\d{1,3})\s*$",
                     re.M)                                                # sbv
_RE_TTML = re.compile(r'begin="([^"]+)"[^>]*?end="([^"]+)"')              # ttml, dfxp


def _hmsf(h, m, sec, frac):
    return int(h or 0) * 3600 + int(m) * 60 + int(sec) + int(frac) / (10 ** len(frac))


def _ttml_time(t):
    t = t.strip()
    try:
        if t.endswith("ms"):
            return float(t[:-2]) / 1000
        if t.endswith("s"):
            return float(t[:-1])
        parts = t.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        pass
    return None


def caption_seconds(path):
    """Seconds of a recording covered by a caption file, from cue timestamps only.
    Returns None if the file has no readable timestamps."""
    raw = path.read_bytes()[:20_000_000]
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16", "replace")
    else:
        text = raw.decode("utf-8", "replace")
    ext = path.suffix.lower()
    iv = []
    if ext in (".srt", ".vtt"):
        for g in _RE_CUE.findall(text):
            iv.append((_hmsf(*g[:4]), _hmsf(*g[4:])))
    elif ext in (".ass", ".ssa"):
        for g in _RE_ASS.findall(text):
            iv.append((_hmsf(*g[:4]), _hmsf(*g[4:])))
    elif ext == ".sbv":
        for g in _RE_SBV.findall(text):
            iv.append((_hmsf(*g[:4]), _hmsf(*g[4:])))
    elif ext in (".ttml", ".dfxp"):
        for b, e in _RE_TTML.findall(text):
            tb, te = _ttml_time(b), _ttml_time(e)
            if tb is not None and te is not None:
                iv.append((tb, te))
    del text
    iv = sorted((a, b) for a, b in iv if b > a)
    if not iv:
        return None
    total, cur_a, cur_b = 0.0, iv[0][0], iv[0][1]
    for a, b in iv[1:]:
        if a > cur_b:
            total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    return total + (cur_b - cur_a)


def match_labels(labels, index):
    """Pair caption/transcript files with media files of the same name in the
    same folder ("lesson1.mp4" with "lesson1.srt", "lesson1.en.srt" or
    "lesson1.mp4.srt"). Updates the matched records in place."""
    stats = Counter()
    for p in labels:
        ext = p.suffix.lower()
        stem = p.stem.lower()
        rec = None
        while True:
            rec = index.get((str(p.parent), stem))
            if rec is not None or "." not in stem:
                break
            stem = stem.rsplit(".", 1)[0]
        is_caption = ext in CAPTION_EXT
        if rec is None:
            stats["caption files not matching any recording" if is_caption
                  else "document files not matching any recording"] += 1
            continue
        if not is_caption:
            rec["has_doc"] = True
            continue
        try:
            sec = caption_seconds(p)
        except OSError:
            sec = None
        if sec is None:
            stats["matched caption files without readable timestamps"] += 1
            rec["has_caption"] = True
            continue
        rec["has_caption"] = True
        # several caption files for one recording (e.g. two languages) describe the
        # same audio, so keep the largest coverage rather than adding them up
        rec["caption_s"] = max(rec.get("caption_s", 0.0), sec)
    return dict(stats)


def fingerprint(path, size):
    h = hashlib.sha1(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        if size > (2 << 20):
            f.seek(-(1 << 20), os.SEEK_END)
            h.update(f.read(1 << 20))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Existing labels: dataset manifests (JSON, JSON Lines, CSV, TSV)
#
# Many datasets (AfriVoice, NeMo-style corpora, Common Voice, Hugging Face
# exports) keep transcriptions in a manifest that links each entry to an audio
# file by path or ID. Only the *presence and length* of each transcription is
# measured; the text itself is discarded as soon as it has been counted.
# ---------------------------------------------------------------------------
MANIFEST_MAX_BYTES = 1_000_000_000      # skip larger manifests (memory)
MANIFEST_SUPPRESS_BELOW = 5             # categories with fewer entries are not shown

# Field names are matched case-insensitively, in priority order.
PATH_FIELDS = ["audio_filepath", "audio_path", "audio_file", "audio_filename", "filepath",
               "file_path", "path", "audio", "wav", "wav_path", "file", "filename",
               "file_name", "recording", "media_path"]
KEY_FIELDS = ["audio_id", "utt_id", "utterance_id", "id", "key", "__key__", "uid",
              "segment_id", "clip_id", "sample_id"]
TEXT_FIELDS = ["transcription", "transcript", "text", "raw_text", "raw_transcription",
               "normalized_text", "normalized_transcription", "label", "labels",
               "utterance", "words", "orthographic", "verbatim"]
PROMPT_FIELDS = ["script", "prompt", "prompt_text", "sentence", "read_text", "source_text"]
SPEAKER_FIELDS = ["speaker_id", "speaker", "spk_id", "spk", "voice_creator_id", "creator_id",
                  "client_id", "user_id", "participant_id", "contributor_id"]
AGE_FIELDS = ["age_group", "age_range", "age_band", "age"]
GENDER_FIELDS = ["gender", "sex"]
LOCATION_FIELDS = ["location", "region", "county", "district", "province", "city", "locale"]
# Recording-date fields, most specific first. Fields about transcription or
# upload dates are deliberately not listed.
DATE_FIELDS = ["recording_date", "recorded_at", "recorded_on", "date_recorded", "recording_time",
               "recording_datetime", "recording_year", "record_date", "session_date",
               "capture_date", "year", "date", "datetime", "timestamp", "created_at",
               "creation_date", "creation_time", "created", "time"]
DIR_FIELDS = ["dir", "directory", "audio_dir", "root"]

AGE_NUM_BUCKETS = [(6, "0 - 5"), (10, "6 - 9"), (13, "10 - 12"), (18, "13 - 17"), (INF, "18 and over")]
CPS_BUCKETS = [(4, "under 4 characters per second of speech (sparse: partial or condensed?)"),
               (25, "4 - 25 (typical)"),
               (INF, "over 25 (dense: possibly mismatched with the audio)")]


def _ci(record):
    return {str(k).lower(): v for k, v in record.items()}


def _first_str(rec, fields):
    for f in fields:
        v = rec.get(f)
        if isinstance(v, dict):                     # e.g. {"path": "...", "bytes": ...}
            v = v.get("path") or v.get("filename")
        if isinstance(v, list):
            v = " ".join(str(x) for x in v if isinstance(x, (str, int, float)))
        if isinstance(v, (str, int)) and str(v).strip() and str(v).strip().lower() not in ("null", "none", "nan"):
            return f, str(v).strip()
    return None, None


def iter_manifest_records(path):
    """Yield dict records from a JSON, JSON Lines, CSV or TSV manifest."""
    ext = path.suffix.lower()
    if ext in (".csv", ".tsv"):
        with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as fh:
            sample = fh.read(65536)
            fh.seek(0)
            delim = "\t" if ext == ".tsv" or sample.count("\t") > sample.count(",") else ","
            for row in csv.DictReader(fh, delimiter=delim):
                yield row
        return
    if ext == ".jsonl":
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(obj, dict):
                        yield obj
        return
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        text = fh.read()
    try:
        data = json.loads(text)
    except ValueError:
        # a .json file that is really JSON Lines
        del text
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            for line in fh:
                line = line.strip().rstrip(",")
                if line.startswith("{"):
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(obj, dict):
                        yield obj
        return
    del text
    yield from _walk_json(data)


def _walk_json(data, depth=0):
    if depth > 3:
        return
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return
    if not isinstance(data, dict):
        return
    lists = [v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
    if lists:                                       # {"data": [ {...}, ... ]}
        for lst in lists:
            yield from _walk_json(lst, depth + 1)
        return
    dict_vals = [(k, v) for k, v in data.items() if isinstance(v, dict)]
    if dict_vals and len(dict_vals) >= 0.8 * len(data):   # {"utt001": {...}, ...}
        for k, v in dict_vals:
            rec = dict(v)
            rec.setdefault("__key__", k)
            yield rec
        return
    yield data                                      # a single record


class MediaIndex:
    """Look up recordings by relative path, file name, or name without extension."""

    def __init__(self, root, records, aliases):
        self.root = root.resolve()
        self.by_rel, self.by_base, self.by_stem = {}, {}, {}
        for p, rec in [(r["path"], r) for r in records] + list(aliases):
            try:
                rel = p.resolve().relative_to(self.root).as_posix().lower()
            except ValueError:
                continue
            self.by_rel.setdefault(rel, rec)
            self.by_base.setdefault(p.name.lower(), []).append((rel, rec))
            self.by_stem.setdefault(p.stem.lower(), []).append((rel, rec))

    def _unique(self, entries, near):
        recs = {id(r): r for _, r in entries}
        if len(recs) == 1:
            return next(iter(recs.values())), False
        close = {id(r): r for rel, r in entries if near and rel.startswith(near)}
        if len(close) == 1:
            return next(iter(close.values())), False
        return None, len(recs) > 1

    def find(self, value, manifest_dir, dir_hint=None, key_only=False):
        """Return (record or None, ambiguous). manifest_dir is an absolute, resolved
        path string; joins use pure path arithmetic so large manifests stay fast."""
        v = value.replace("\\", "/").strip()
        if v.lower().startswith("file://"):
            v = v[7:]
        root = str(self.root)
        near = os.path.relpath(manifest_dir, root).replace(os.sep, "/").lower()
        near = "" if near in (".", "") or near.startswith("..") else near + "/"
        if not key_only:
            bases = [manifest_dir, root]
            if dir_hint:
                bases += [os.path.join(manifest_dir, dir_hint), os.path.join(root, dir_hint)]
            for b in bases:
                rel = os.path.relpath(os.path.normpath(os.path.join(b, v)), root)
                rel = rel.replace(os.sep, "/").lower()
                if not rel.startswith("..") and rel in self.by_rel:
                    return self.by_rel[rel], False
        name = v.rsplit("/", 1)[-1].lower()
        if name in self.by_base:
            return self._unique(self.by_base[name], near)
        stem = name.rsplit(".", 1)[0] if "." in name else name
        if stem in self.by_stem:
            return self._unique(self.by_stem[stem], near)
        return None, False


def _norm_age(v):
    try:
        n = float(v)
        if 0 <= n < 120:
            return bucket_label(n, AGE_NUM_BUCKETS)
    except (TypeError, ValueError):
        pass
    return str(v).strip()[:40] or None


def _year_from_value(v):
    """Plausible recording year (1995 to next year) from a date-like value:
    a year, an ISO or day/month/year date, or a Unix timestamp in s or ms."""
    hi = dt.date.today().year + 1
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, str):
        t = v.strip()
        if not t:
            return None
        m8 = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", t)  # 20230501
        if m8 and 1 <= int(m8.group(2)) <= 12 and 1 <= int(m8.group(3)) <= 31:
            y = int(m8.group(1))
            return y if 1995 <= y <= hi else None
        if re.fullmatch(r"\d+(\.\d+)?", t):
            v = float(t)                              # "2023", "1688000000", "1.688e12" style
        else:
            m = re.match(r"(\d{4})(?:[-/.T\s]|$)", t)  # 2023-05-01, 2023/05/01, 2023-05-01T...
            if not m:
                m = re.search(r"\b\d{1,2}[-/.]\d{1,2}[-/.](\d{4})\b", t)   # 01/05/2023
            if not m:
                m = re.search(r"(?<!\d)(\d{4})(?!\d)", t)                   # "May 2023"
            if m:
                y = int(m.group(1))
                return y if 1995 <= y <= hi else None
            return None
    if isinstance(v, (int, float)):
        n = float(v)
        if 1995 <= n <= hi:
            return int(n)
        for scale in (1, 1000):                       # Unix seconds or milliseconds
            secs = n / scale
            if 788918400 <= secs <= (dt.datetime(hi, 12, 31).timestamp()):
                return dt.datetime.fromtimestamp(secs, dt.timezone.utc).year
    return None


def _manifest_year(rec_ci):
    for f in DATE_FIELDS:
        if f in rec_ci:
            v = rec_ci[f]
            if isinstance(v, dict):                   # e.g. {"$date": "..."} or {"seconds": ...}
                v = v.get("$date") or v.get("seconds") or v.get("_seconds")
            y = _year_from_value(v)
            if y:
                return f, y
    return None, None


def _hash(v):
    return hashlib.sha1(str(v).encode("utf-8")).hexdigest()[:16]


def match_manifests(paths, media_index, keep_text=False):
    """Scan candidate manifest files and attach transcription facts to recordings.
    Returns a dict of manifest-level statistics (counts and field names only)."""
    st = {"files_examined": 0, "manifests_used": 0, "files_skipped_too_large": 0,
          "files_unreadable": 0, "entries": 0, "entries_linked": 0,
          "entries_audio_not_found": 0, "entries_ambiguous": 0,
          "entries_without_audio_reference": 0}
    path_fields, text_fields, prompt_fields, meta_fields = Counter(), Counter(), Counter(), Counter()
    for mp in paths:
        st["files_examined"] += 1
        try:
            if mp.stat().st_size > MANIFEST_MAX_BYTES:
                st["files_skipped_too_large"] += 1
                continue
        except OSError:
            continue
        linked_here, entries_here = 0, Counter()
        mdir = str(mp.parent.resolve())
        try:
            for raw in iter_manifest_records(mp):
                if STOP.is_set():
                    raise _Stopped()
                rec_ci = _ci(raw)
                pf, pval = _first_str(rec_ci, PATH_FIELDS)
                key_only = False
                if pval is None:
                    # any other string field naming a media file
                    for k, v in rec_ci.items():
                        if isinstance(v, str) and Path(v.strip()).suffix.lower() in AUDIO_EXT | VIDEO_EXT:
                            pf, pval = k, v
                            break
                if pval is None:
                    pf, pval = _first_str(rec_ci, KEY_FIELDS)
                    key_only = True
                entries_here["entries"] += 1
                if pval is None:
                    entries_here["no_ref"] += 1
                    continue
                _, dir_hint = _first_str(rec_ci, DIR_FIELDS)
                media, ambiguous = media_index.find(pval, mdir, dir_hint, key_only)
                if media is None:
                    entries_here["ambiguous" if ambiguous else "not_found"] += 1
                    continue
                linked_here += 1
                path_fields[pf] += 1
                tf, tval = _first_str(rec_ci, TEXT_FIELDS)
                qf, qval = _first_str(rec_ci, PROMPT_FIELDS)
                if tval:
                    text_fields[tf] += 1
                    chars = len(" ".join(tval.split()))
                    media["mf_text"] = True
                    media["mf_chars"] = max(media.get("mf_chars", 0), chars)
                if qval:
                    prompt_fields[qf] += 1
                    media["mf_prompt"] = True
                    if not tval:
                        media["mf_prompt_chars"] = max(media.get("mf_prompt_chars", 0),
                                                       len(" ".join(qval.split())))
                if keep_text:                       # drive_explore.py only
                    if tval:
                        media.setdefault("mf_texts", []).append(tval)
                    if qval:
                        media.setdefault("mf_prompts", []).append(qval)
                tval = qval = None                  # discard the text
                media["mf_entry"] = True
                for fields, key, norm in ((SPEAKER_FIELDS, "mf_speaker", _hash),
                                          (AGE_FIELDS, "mf_age", _norm_age),
                                          (GENDER_FIELDS, "mf_gender", lambda v: str(v).strip().lower()[:20]),
                                          (LOCATION_FIELDS, "mf_location", _hash)):
                    f, v = _first_str(rec_ci, fields)
                    if v is not None and key not in media:
                        media[key] = norm(v)
                        meta_fields[f] += 1
                if "mf_year" not in media:
                    f, y = _manifest_year(rec_ci)
                    if y:
                        media["mf_year"] = y
                        meta_fields[f] += 1
        except _Stopped:
            raise
        except (OSError, csv.Error, UnicodeError, RecursionError, MemoryError):
            st["files_unreadable"] += 1
            continue
        if linked_here or entries_here["ambiguous"]:
            st["manifests_used"] += 1
            st["entries"] += entries_here["entries"]
            st["entries_linked"] += linked_here
            st["entries_audio_not_found"] += entries_here["not_found"]
            st["entries_ambiguous"] += entries_here["ambiguous"]
            st["entries_without_audio_reference"] += entries_here["no_ref"]
    st["audio_reference_fields"] = dict(path_fields.most_common())
    st["transcription_fields"] = dict(text_fields.most_common())
    st["prompt_fields"] = dict(prompt_fields.most_common())
    st["metadata_fields"] = dict(meta_fields.most_common())
    return st


# ---------------------------------------------------------------------------
# Optional language identification
# ---------------------------------------------------------------------------
# Language identification: estimated speech hours per language
#
# For each selected file, Whisper's language detector runs on a few 30-second
# windows spread across the file's speech. Each window is labelled English,
# non-English (with its most likely language), or mixed/uncertain, and the
# file's speech hours are divided between those labels. No transcription is
# produced: only the language probabilities are read.
LID_EN_MIN = 0.60          # English if English is the top language at 60% or more
LID_NON_EN_MAX = 0.25      # non-English if English scores below 25%
LANG_NAMES = {"en": "English", "sw": "Swahili", "so": "Somali", "ar": "Arabic", "fr": "French",
              "am": "Amharic", "hi": "Hindi", "ur": "Urdu", "yo": "Yoruba", "ha": "Hausa",
              "ln": "Lingala", "sn": "Shona", "af": "Afrikaans", "pt": "Portuguese",
              "es": "Spanish", "de": "German", "zh": "Chinese", "nl": "Dutch", "mg": "Malagasy",
              "it": "Italian", "ru": "Russian", "tr": "Turkish", "id": "Indonesian",
              "ms": "Malay", "jw": "Javanese", "cy": "Welsh", "la": "Latin", "haw": "Hawaiian",
              "mi": "Maori", "tl": "Tagalog", "ja": "Japanese", "ko": "Korean"}
_LID_LOCK = threading.Lock()


def lang_name(code):
    return f"{LANG_NAMES.get(code, code)} ({code})"


LID_MODEL_MB = {"tiny": 75, "base": 145, "small": 485, "medium": 1530}


def app_dir():
    """Folder the user sees: next to the .exe when frozen, else next to this script."""
    return Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else \
        Path(__file__).resolve().parent


def local_lid_model(size):
    """A model folder shipped with the tool, if any: the given path itself, or
    models/faster-whisper-<size> next to the program."""
    p = Path(size)
    candidates = [p] if p.is_absolute() or os.sep in size or "/" in size else []
    candidates += [app_dir() / "models" / f"faster-whisper-{size}", app_dir() / f"faster-whisper-{size}"]
    for c in candidates:
        if (c / "model.bin").exists():
            return str(c)
    return None


def load_lid_model(size):
    """Load a faster-whisper model, on a GPU if one is available. Uses a model
    folder shipped next to the program if there is one; otherwise downloads the
    model once (it is then cached). Returns (model, error)."""
    try:
        import av  # noqa: F401
    except ImportError:
        # faster-whisper imports PyAV on startup but only uses it to decode audio
        # files itself; this tool always passes decoded audio, so a placeholder
        # module lets the executable leave PyAV (and its own ffmpeg copy) out.
        import types
        sys.modules.setdefault("av", types.ModuleType("av"))
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None, "skipped: the faster-whisper package is not installed"
    device, ctype = "cpu", "int8"
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            device, ctype = "cuda", "float16"
    except Exception:
        pass
    local = local_lid_model(size)
    if local:
        print(f"  Using the language model in {local}")
    else:
        mb = LID_MODEL_MB.get(size)
        print(f"  Downloading the Whisper '{size}' model"
              + (f" (about {mb} MB)" if mb else "") + "; this happens once and needs internet...")
    try:
        return WhisperModel(local or size, device=device, compute_type=ctype), None
    except Exception as e:  # model download or load failure
        where = "the shipped model folder" if local else "the model (no internet?)"
        return None, f"skipped: could not load {where} ({type(e).__name__})"


def lid_windows(analysis, k):
    """Indices of up to k 30 s chunks with at least 30% speech, spread across the file."""
    ratios = analysis.get("chunk_ratios") or []
    eligible = [i for i, r in enumerate(ratios) if r >= 0.3]
    if len(eligible) <= k:
        return eligible
    if k == 1:
        return [eligible[len(eligible) // 2]]
    return sorted({eligible[round(j * (len(eligible) - 1) / (k - 1))] for j in range(k)})


def classify_window(probs):
    """probs: [(language, probability)] sorted high to low."""
    p_en = dict(probs).get("en", 0.0)
    top, p_top = probs[0]
    if top == "en" and p_top >= LID_EN_MIN:
        return "en"
    if p_en < LID_NON_EN_MAX:
        return top
    return "mixed"


def lid_for_file(path, ffmpeg, analysis, model, k):
    windows = []
    for ci in lid_windows(analysis, k):
        if STOP.is_set():
            raise _Stopped()
        chunks = list(decode_chunks(path, ffmpeg, start=ci * CHUNK_SECONDS,
                                    max_seconds=CHUNK_SECONDS))
        if not chunks:
            continue
        audio = np.concatenate(chunks).astype(np.float32) / 32768.0
        with _LID_LOCK:
            segments, info = model.transcribe(audio, beam_size=1, vad_filter=False,
                                              without_timestamps=True)
            del segments  # never iterated: no transcript is produced
        probs = sorted(info.all_language_probs or [(info.language, info.language_probability)],
                       key=lambda t: -t[1])
        windows.append({"label": classify_window(probs), "top": probs[0][0],
                        "p_top": round(float(probs[0][1]), 3),
                        "p_en": round(float(dict(probs).get("en", 0.0)), 3)})
    return {"status": "ok" if windows else "no_windows", "windows": windows}


def lid_signature(model_size, windows_per_file):
    raw = f"{model_size}|{windows_per_file}|{LID_EN_MIN}|{LID_NON_EN_MAX}"
    return hashlib.sha1(raw.encode()).hexdigest()[:8]


def plan_lid(records, k, max_windows):
    """Files to check, within the overall window budget (a random sample if needed)."""
    cands = [r for r in records if (r.get("analysis") or {}).get("status") == "ok"
             and lid_windows(r["analysis"], k)]
    if sum(len(lid_windows(r["analysis"], k)) for r in cands) <= max_windows:
        return cands
    order = cands[:]
    random.Random(0).shuffle(order)
    chosen, used = [], 0
    for r in order:
        n = len(lid_windows(r["analysis"], k))
        if used + n <= max_windows:
            chosen.append(r)
            used += n
    return chosen


def run_language_id(records, ffmpeg, model_size, windows_per_file, max_windows, ckpt):
    model, err = load_lid_model(model_size)
    if model is None:
        return {"status": err}
    plan = plan_lid(records, windows_per_file, max_windows)
    sig = lid_signature(model_size, windows_per_file)
    todo = []
    for r in plan:
        saved = ckpt.lids.get(f"{r['key']}|{sig}") if r["key"] else None
        if saved is not None:
            r["lid"] = saved
        else:
            todo.append(r)
    print(f"  {len(plan)} files selected"
          + (f", {len(plan) - len(todo)} already done in the previous run." if len(todo) < len(plan) else "."))
    for i, r in enumerate(todo, 1):
        print(f"  Language check: {i}/{len(todo)} files", end="\r", flush=True)
        try:
            res = lid_for_file(r["path"], ffmpeg, r["analysis"], model, windows_per_file)
        except _Stopped:
            raise
        except Exception:
            res = {"status": "error", "windows": []}
        r["lid"] = res
        if r["key"]:
            ckpt.save_lid(f"{r['key']}|{sig}", res)
    print()
    return lid_meta(model_size, windows_per_file, max_windows, len(plan))


def lid_meta(model_size, windows_per_file, max_windows, files_selected):
    return {"status": "ok", "model": model_size, "windows_per_file": windows_per_file,
            "max_windows": max_windows, "files_selected": files_selected, "note": LID_NOTE}


LID_NOTE = ("Whisper only recognizes the languages it was trained on, and cannot recognize "
            "most Kenyan languages other than Swahili and Somali (for example Kikuyu, Dholuo, "
            "Luhya, Kalenjin or Kamba). For the English vs. non-English split this matters "
            "less: speech in an unrecognized language is usually labelled as some other "
            "non-English language, so it still counts as non-English, but the specific "
            "language label is unreliable. Sheng and code-switched speech tend to land in "
            "'mixed or uncertain' or be labelled Swahili. Children's Kenyan-accented English "
            "can occasionally be labelled Swahili, which would overstate non-English hours.")


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------
def tally(records, key_fn, weight_fn=None):
    counts, hours = Counter(), Counter()
    for r in records:
        k = key_fn(r)
        counts[k] += 1
        if weight_fn:
            hours[k] += weight_fn(r)
    return counts, hours


def ordered(counts, hours, buckets):
    labels = [b[1] for b in buckets] + ["unknown / not measurable"]
    return [{"range": lab, "files": counts.get(lab, 0), "hours": round(hours.get(lab, 0.0), 2)}
            for lab in labels if counts.get(lab, 0)]


def child_share(a):
    """Share of a file's voiced speech frames in the child pitch range."""
    hist = np.asarray(a.get("f0_hist") or [], dtype=float)
    if hist.sum() < 20:
        return None
    centers = (F0_BIN_EDGES[:-1] + F0_BIN_EDGES[1:]) / 2
    return float(hist[centers >= CHILD_F0].sum() / hist.sum())


def build_report(n_media, other_ext, records, dup_count, settings, lid, label_stats,
                 manifest_stats=None):
    def dur_h(r):
        d = r["probe"].get("duration_s") or (r.get("analysis") or {}).get("analyzed_s") or 0.0
        return d / 3600.0

    audio_recs = [r for r in records if r["probe"].get("has_audio")]
    analyzed = [r for r in audio_recs if (r.get("analysis") or {}).get("status") == "ok"]
    pitch_on = bool(settings.get("pitch_analysis"))

    status = Counter()
    for r in records:
        if not r["probe"].get("ok"):
            status["could not be opened"] += 1
        elif not r["probe"].get("has_audio"):
            status["no audio track"] += 1
        elif r.get("analysis") is None:
            status["has audio (not in analysis sample)"] += 1
        else:
            st = r["analysis"]["status"]
            status[{"ok": "analyzed", "empty": "audio track empty",
                    "decode_error": "audio could not be decoded",
                    "download_error": "could not be downloaded"}.get(st, st)] += 1

    total_audio_h = sum(dur_h(r) for r in audio_recs)
    analyzed_h = sum(dur_h(r) for r in analyzed)
    scale = (total_audio_h / analyzed_h) if analyzed_h > 0 else 0.0

    def speech_h(r):
        return dur_h(r) * r["analysis"]["speech_ratio"]

    def share_h(r):
        return child_share(r["analysis"])

    est_speech_h = sum(speech_h(r) for r in analyzed) * scale

    # --- by source type
    by_source = []
    for kind, label in (("video", "audio inside video files"), ("audio", "audio-only files")):
        recs = [r for r in audio_recs if r["kind"] == kind]
        an = [r for r in analyzed if r["kind"] == kind]
        row = {"source": label, "files": len(recs),
               "audio_hours": round(sum(dur_h(r) for r in recs), 2),
               "est_speech_hours": round(sum(speech_h(r) for r in an) * scale, 2)}
        if pitch_on:
            row["est_child_range_speech_hours"] = round(
                sum(speech_h(r) * (share_h(r) or 0) for r in an) * scale, 2)
        by_source.append(row)

    # --- pitch and child/adult mix
    hist = np.zeros(len(F0_BIN_EDGES) - 1)
    for r in analyzed:
        hist += np.array(r["analysis"]["f0_hist"])
    centers = (F0_BIN_EDGES[:-1] + F0_BIN_EDGES[1:]) / 2
    band_share = Counter()
    if hist.sum() > 0:
        for c, v in zip(centers, hist):
            band_share[bucket_label(float(c), F0_BANDS)] += v / hist.sum()
    pitch_frames = [{"range": b[1], "share_of_voiced_speech": round(band_share.get(b[1], 0.0), 3)}
                    for b in F0_BANDS]
    mc, mh = tally(analyzed, lambda r: bucket_label(share_h(r), CHILD_MIX_BUCKETS), speech_h)
    mix_rows = [{"range": d["range"], "files": d["files"], "speech_hours": round(d["hours"] * scale, 2)}
                for d in ordered(mc, mh, CHILD_MIX_BUCKETS)] if pitch_on else []
    child_speech_h = mostly_child_h = None
    if pitch_on:
        child_speech_h = round(sum(speech_h(r) * (share_h(r) or 0) for r in analyzed) * scale, 2)
        mostly_child_h = round(mh.get(CHILD_MIX_BUCKETS[-1][1], 0.0) * scale, 2)

    # --- quality
    snr_c, snr_h = tally(analyzed, lambda r: bucket_label(r["analysis"]["snr_db"], SNR_BUCKETS), dur_h)
    low_snr_h = sum(snr_h.get(b[1], 0.0) for b in SNR_BUCKETS[:2])
    lvl_c, lvl_h = tally(analyzed, lambda r: bucket_label(r["analysis"]["speech_level_dbfs"], LEVEL_BUCKETS), dur_h)
    clip_c, clip_h = tally(analyzed, lambda r: bucket_label(r["analysis"]["clip_fraction"], CLIP_BUCKETS), dur_h)
    bw_c, bw_h = tally(analyzed, lambda r: bucket_label(r["analysis"].get("bandwidth_hz"), BANDWIDTH_BUCKETS), dur_h)
    narrow_h = bw_h.get(BANDWIDTH_BUCKETS[0][1], 0.0)
    sr_c, sr_h = tally(analyzed, lambda r: bucket_label(r["analysis"]["speech_ratio"], SPEECH_RATIO_BUCKETS), dur_h)
    du_c, du_h = tally(audio_recs, lambda r: bucket_label(dur_h(r) * 3600 if dur_h(r) else None, DURATION_BUCKETS), dur_h)
    br_c, br_h = tally(audio_recs, lambda r: bucket_label(r["probe"].get("bitrate_kbps"), BITRATE_BUCKETS), dur_h)

    # --- speech segments
    run_counts = np.zeros(len(RUN_BUCKETS))
    run_time = np.zeros(len(RUN_BUCKETS))
    pauses = 0
    for r in analyzed:
        a = r["analysis"]
        run_counts += np.array(a.get("run_counts") or [0] * len(RUN_BUCKETS))
        run_time += np.array(a.get("run_time_s") or [0] * len(RUN_BUCKETS))
        pauses += a.get("pauses", 0)
    total_run = float(run_time.sum())
    seg_rows = [{"range": b[1], "speech_runs": int(run_counts[i]),
                 "share_of_speech_time": round(float(run_time[i]) / total_run, 3) if total_run else 0.0}
                for i, b in enumerate(RUN_BUCKETS)]
    long_share = round(float(run_time[-1]) / total_run, 3) if total_run else None
    pause_rate = round(pauses / (total_run / 60), 1) if total_run else None

    # --- existing labels
    cap_recs = [r for r in audio_recs if r.get("has_caption")]
    doc_recs = [r for r in audio_recs if r.get("has_doc")]
    caption_h = sum(min(r.get("caption_s", 0.0) / 3600, dur_h(r) or INF) for r in cap_recs)
    caption_files = {e: n for e, n in other_ext.items() if e in CAPTION_EXT}
    doc_files = {e: n for e, n in other_ext.items() if e in DOC_EXT}

    # --- manifests
    ms = manifest_stats or {}
    mf_text = [r for r in audio_recs if r.get("mf_text")]
    mf_prompt_only = [r for r in audio_recs if r.get("mf_prompt") and not r.get("mf_text")]
    mf_empty = [r for r in audio_recs if r.get("mf_entry") and not r.get("mf_text")
                and not r.get("mf_prompt")]
    mf_any = [r for r in audio_recs if r.get("mf_entry")]
    labeled = [r for r in audio_recs if r.get("mf_text") or r.get("caption_s")]

    def speech_s(r):
        a = r.get("analysis") or {}
        if a.get("status") == "ok":
            return dur_h(r) * 3600 * a["speech_ratio"]
        return dur_h(r) * 3600

    cps_c, cps_h = Counter(), Counter()
    for r in mf_text:
        sp = speech_s(r)
        lab = bucket_label(r["mf_chars"] / sp if sp > 0.5 else None, CPS_BUCKETS)
        cps_c[lab] += 1
        cps_h[lab] += dur_h(r)

    def suppressed(key):
        c, h = Counter(), Counter()
        for r in mf_any:
            v = r.get(key)
            k = str(v) if v not in (None, "") else "not given"
            c[k] += 1
            h[k] += dur_h(r)
        rows, small = [], 0
        for k in sorted(c, key=lambda k: (k == "not given", k)):
            if c[k] < MANIFEST_SUPPRESS_BELOW:
                small += c[k]
                continue
            rows.append({"range": k, "files": c[k], "hours": round(h[k], 2)})
        if small:
            rows.append({"range": f"(groups with fewer than {MANIFEST_SUPPRESS_BELOW} recordings, combined)",
                         "files": small, "hours": None})
        return rows

    spk_h = Counter()
    for r in mf_any:
        if r.get("mf_speaker"):
            spk_h[r["mf_speaker"]] += dur_h(r)
    speakers = None
    if spk_h:
        vals = sorted(spk_h.values(), reverse=True)
        tot = sum(vals) or 1
        speakers = {
            "distinct_speakers": len(vals),
            "recordings_with_speaker_id": sum(1 for r in mf_any if r.get("mf_speaker")),
            "median_audio_minutes_per_speaker": round(float(np.median(vals)) * 60, 1),
            "share_of_hours_from_top_speaker": round(vals[0] / tot, 3),
            "share_of_hours_from_top_10_percent_of_speakers":
                round(sum(vals[:max(1, len(vals) // 10)]) / tot, 3),
        }
    locations = len({r["mf_location"] for r in mf_any if r.get("mf_location")})
    manifests = {
        "files_examined": ms.get("files_examined", 0),
        "manifests_used": ms.get("manifests_used", 0),
        "entries_in_manifests": ms.get("entries", 0),
        "entries_linked_to_a_recording": ms.get("entries_linked", 0),
        "entries_whose_audio_was_not_found": ms.get("entries_audio_not_found", 0),
        "entries_matching_several_recordings": ms.get("entries_ambiguous", 0),
        "entries_without_an_audio_reference": ms.get("entries_without_audio_reference", 0),
        "files_skipped_too_large": ms.get("files_skipped_too_large", 0),
        "files_unreadable": ms.get("files_unreadable", 0),
        "fields_used_for_audio_reference": ms.get("audio_reference_fields", {}),
        "fields_used_for_transcription": ms.get("transcription_fields", {}),
        "fields_used_for_prompt_or_script": ms.get("prompt_fields", {}),
        "metadata_fields_found": ms.get("metadata_fields", {}),
        "recordings_with_transcription": len(mf_text),
        "hours_with_transcription": round(sum(dur_h(r) for r in mf_text), 2),
        "recordings_with_prompt_or_script_only": len(mf_prompt_only),
        "hours_with_prompt_or_script_only": round(sum(dur_h(r) for r in mf_prompt_only), 2),
        "recordings_listed_without_any_text": len(mf_empty),
        "transcription_density": ordered(cps_c, cps_h, CPS_BUCKETS),
        "speakers": speakers,
        "age_groups": suppressed("mf_age") if any(r.get("mf_age") for r in mf_any) else [],
        "gender": suppressed("mf_gender") if any(r.get("mf_gender") for r in mf_any) else [],
        "distinct_locations": locations or None,
    }

    # --- languages
    language = dict(lid)
    lang_summary = {}
    if lid.get("status") == "ok":
        lang_h, lang_child_h, top_windows = Counter(), Counter(), Counter()
        assessed_h = assessed_child_h = 0.0
        files_assessed = n_windows = 0
        for r in analyzed:
            res = r.get("lid") or {}
            wins = res.get("windows") or []
            if res.get("status") != "ok" or not wins:
                continue
            files_assessed += 1
            sh, cs = speech_h(r), (share_h(r) or 0.0)
            assessed_h += sh
            assessed_child_h += sh * cs
            for w in wins:
                lang_h[w["label"]] += sh / len(wins)
                lang_child_h[w["label"]] += sh * cs / len(wins)
                top_windows[w["top"]] += 1
                n_windows += 1
        f = (est_speech_h / assessed_h) if assessed_h else 0.0
        en_h, mixed_h = lang_h.get("en", 0.0), lang_h.get("mixed", 0.0)
        non_h = max(assessed_h - en_h - mixed_h, 0.0)
        non_child_h = max(assessed_child_h - lang_child_h.get("en", 0.0)
                          - lang_child_h.get("mixed", 0.0), 0.0)

        def row(label, h, ch):
            r = {"range": label, "speech_hours_assessed": round(h, 2),
                 "share": round(h / assessed_h, 3) if assessed_h else 0.0,
                 "est_speech_hours_all": round(h * f, 2)}
            if pitch_on:
                r["est_child_range_speech_hours_all"] = round(ch * f, 2)
            return r

        rows = [row("English", en_h, lang_child_h.get("en", 0.0)),
                row("non-English (all languages)", non_h, non_child_h),
                row("mixed or uncertain", mixed_h, lang_child_h.get("mixed", 0.0))]
        by_lang = [row("  " + lang_name(code), h, lang_child_h.get(code, 0.0))
                   for code, h in sorted(lang_h.items(), key=lambda t: -t[1])
                   if code not in ("en", "mixed")]
        language.update({
            "files_assessed": files_assessed,
            "windows_checked": n_windows,
            "speech_hours_assessed": round(assessed_h, 2),
            "share_of_estimated_speech_assessed": round(assessed_h / est_speech_h, 3) if est_speech_h else None,
            "by_language_group": rows,
            "non_english_by_language": by_lang,
            "most_likely_language_per_window": {lang_name(k): v for k, v in top_windows.most_common()},
        })
        lang_summary = {
            "est_non_english_speech_hours": round(non_h * f, 2),
            "est_non_english_child_range_speech_hours": round(non_child_h * f, 2) if pitch_on else None,
            "share_of_speech_non_english": round(non_h / assessed_h, 3) if assessed_h else None,
            "share_of_speech_mixed_or_uncertain": round(mixed_h / assessed_h, 3) if assessed_h else None,
        }

    # --- recording dates
    def year_of(r):
        # manifest metadata first, then the recording's embedded date, then file dates
        return r.get("mf_year") or r["probe"].get("meta_year") or r.get("file_year")
    yc, yh = Counter(), Counter()
    for r in audio_recs:
        y = year_of(r)
        k = str(y) if y else "unknown"
        yc[k] += 1
        yh[k] += dur_h(r)
    year_rows = [{"range": k, "files": yc[k], "hours": round(yh[k], 2)}
                 for k in sorted(yc, key=lambda k: (k == "unknown", k))]
    date_sources = Counter("manifest metadata" if r.get("mf_year")
                           else "embedded recording date" if r["probe"].get("meta_year")
                           else "file date" if r.get("file_year") else "unknown"
                           for r in audio_recs)
    date_conflicts = sum(1 for r in audio_recs if r.get("mf_year") and r["probe"].get("meta_year")
                         and r["mf_year"] != r["probe"]["meta_year"])

    def simple_counter(fn):
        c = Counter(fn(r) for r in audio_recs)
        return {str(k): v for k, v in c.most_common()}

    other_top = dict(Counter({e: n for e, n in other_ext.items()
                              if e not in TRANSCRIPT_EXT and e not in MANIFEST_EXT}).most_common(15))

    return {
        "tool": {"name": "asr_feasibility", "version": TOOL_VERSION,
                 "run_date": dt.date.today().isoformat(),
                 "os": platform.system(), "python": platform.python_version()},
        "settings": settings,
        "summary": {
            "media_files_found": n_media,
            "duplicate_files_skipped": dup_count,
            "files_with_audio": len(audio_recs),
            "total_audio_hours": round(total_audio_h, 2),
            "files_acoustically_analyzed": len(analyzed),
            "hours_in_analyzed_files": round(analyzed_h, 2),
            "estimated_speech_hours": round(est_speech_h, 2),
            "estimated_child_range_speech_hours": child_speech_h,
            "speech_hours_in_mostly_child_files": mostly_child_h,
            "share_of_audio_hours_with_snr_below_10db":
                round(low_snr_h / analyzed_h, 3) if analyzed_h else None,
            "share_of_audio_hours_narrowband":
                round(narrow_h / analyzed_h, 3) if analyzed_h else None,
            "files_with_frequent_clipping": clip_c.get(CLIP_BUCKETS[-1][1], 0),
            "share_of_speech_in_runs_over_30s": long_share,
            "recordings_with_matching_caption_files": len(cap_recs),
            "caption_covered_hours": round(caption_h, 2),
            "recordings_with_matching_transcript_documents": len(doc_recs),
            "media_files_with_embedded_subtitle_tracks":
                sum(1 for r in records if r["probe"].get("has_subtitles")),
            "recordings_with_manifest_transcription": len(mf_text),
            "hours_with_manifest_transcription": round(sum(dur_h(r) for r in mf_text), 2),
            "recordings_with_any_caption_or_manifest_text": len(labeled),
            "hours_with_any_caption_or_manifest_text": round(sum(dur_h(r) for r in labeled), 2),
            "distinct_speakers_from_manifest": speakers["distinct_speakers"] if speakers else None,
            "archive_files_not_scanned": sum(n for e, n in other_ext.items() if e in ARCHIVE_EXT),
            "est_non_english_speech_hours": lang_summary.get("est_non_english_speech_hours"),
            "est_non_english_child_range_speech_hours":
                lang_summary.get("est_non_english_child_range_speech_hours"),
            "share_of_speech_non_english": lang_summary.get("share_of_speech_non_english"),
            "share_of_speech_mixed_or_uncertain": lang_summary.get("share_of_speech_mixed_or_uncertain"),
        },
        "inventory": {
            "file_status": dict(status.most_common()),
            "by_source": by_source,
            "total_size_gb": round(sum(r["size"] for r in records) / 1e9, 2),
            "extensions": simple_counter(lambda r: r["ext"]),
            "other_file_types_top15": other_top,
        },
        "formats": {
            "audio_codecs": simple_counter(lambda r: r["probe"].get("audio_codec") or "unknown"),
            "original_sample_rates_hz": simple_counter(lambda r: r["probe"].get("sample_rate") or "unknown"),
            "channels": simple_counter(lambda r: r["probe"].get("channels") or "unknown"),
            "audio_bitrate": ordered(br_c, br_h, BITRATE_BUCKETS),
        },
        "durations": ordered(du_c, du_h, DURATION_BUCKETS),
        "speech_activity": ordered(sr_c, sr_h, SPEECH_RATIO_BUCKETS),
        "speech_segments": {
            "runs_by_length": seg_rows,
            "pauses_per_minute_of_speech": pause_rate,
            "share_of_speech_in_runs_over_30s": long_share,
        },
        "quality": {
            "estimated_snr": ordered(snr_c, snr_h, SNR_BUCKETS),
            "speech_level": ordered(lvl_c, lvl_h, LEVEL_BUCKETS),
            "clipping": ordered(clip_c, clip_h, CLIP_BUCKETS),
            "audio_bandwidth": ordered(bw_c, bw_h, BANDWIDTH_BUCKETS),
        },
        "pitch": {
            "voiced_speech_by_pitch_range": pitch_frames,
            "files_by_child_adult_mix": mix_rows,
            "histogram_25hz_bins": {f"{int(F0_BIN_EDGES[i])}-{int(F0_BIN_EDGES[i + 1])}":
                                    int(v) for i, v in enumerate(hist) if v},
        },
        "existing_labels": {
            "caption_files_by_type": caption_files,
            "recordings_with_matching_caption_files": len(cap_recs),
            "audio_hours_of_those_recordings": round(sum(dur_h(r) for r in cap_recs), 2),
            "hours_covered_by_caption_timestamps": round(caption_h, 2),
            "document_files_by_type": doc_files,
            "recordings_with_same_name_document": len(doc_recs),
            "audio_hours_of_those_recordings_doc": round(sum(dur_h(r) for r in doc_recs), 2),
            "media_files_with_embedded_subtitle_tracks":
                sum(1 for r in records if r["probe"].get("has_subtitles")),
            "notes": label_stats,
            "manifests": manifests,
        },
        "recording_dates": {
            "by_year": year_rows,
            "date_source": dict(date_sources.most_common()),
            "manifest_year_differs_from_embedded_date": date_conflicts,
        },
        "language_id": language,
    }


def fmt_rows(rows, value_keys):
    if not rows:
        return "    (no data)\n"
    width = max(len(str(r["range"])) for r in rows) + 2
    out = []
    for r in rows:
        vals = "   ".join(f"{k.replace('_', ' ')}: {'-' if r[k] is None else r[k]}"
                           for k in value_keys)
        out.append(f"    {str(r['range']):<{width}}{vals}")
    return "\n".join(out) + "\n"


def fmt_dict(d):
    if not d:
        return "    (none)\n"
    return "\n".join(f"    {k}: {v}" for k, v in d.items()) + "\n"


def _pct(v):
    return "n/a" if v is None else f"{v * 100:.0f}%"


def _hours(v, off_text="n/a (pitch analysis off)"):
    return off_text if v is None else f"{v} hours"


def render_text(rep):
    s = rep["summary"]
    L = []
    L.append("ASR DATA FEASIBILITY REPORT")
    L.append("=" * 64)
    L.append(f"Tool version {rep['tool']['version']}, run on {rep['tool']['run_date']}\n")
    L.append("WHAT THIS REPORT CONTAINS")
    L.append("  Only totals and distributions across the whole collection.")
    L.append("  It contains no file names, folder names, transcripts, caption text,")
    L.append("  audio, or details about any individual recording. Please read it")
    L.append("  through before sharing it.\n")
    if s["archive_files_not_scanned"]:
        L.append(f"NOTE: {s['archive_files_not_scanned']} archive file(s) (.zip, .tar, .xz and similar) were")
        L.append("  found. Recordings inside archives are not scanned; extract them and run")
        L.append("  the tool again to include them.\n")
    L.append("QUICK SUMMARY")
    rows = [
        ("Media files found", s["media_files_found"]),
        ("Duplicate copies skipped", s["duplicate_files_skipped"]),
        ("Files with an audio track", s["files_with_audio"]),
        ("Total audio", f"{s['total_audio_hours']} hours"),
        ("Estimated speech", f"{s['estimated_speech_hours']} hours"),
        ("Est. speech in child pitch range", _hours(s["estimated_child_range_speech_hours"])),
        ("Speech in mostly-child files", _hours(s["speech_hours_in_mostly_child_files"])),
        ("Audio with SNR below 10 dB", _pct(s["share_of_audio_hours_with_snr_below_10db"])),
        ("Audio that is narrowband", _pct(s["share_of_audio_hours_narrowband"])),
        ("Files with frequent clipping", s["files_with_frequent_clipping"]),
        ("Speech in runs over 30 s", _pct(s["share_of_speech_in_runs_over_30s"])),
        ("Recordings with caption files", s["recordings_with_matching_caption_files"]),
        ("Hours covered by captions", f"{s['caption_covered_hours']} hours"),
        ("Recordings with same-name documents", s["recordings_with_matching_transcript_documents"]),
        ("Media with embedded subtitles", s["media_files_with_embedded_subtitle_tracks"]),
        ("Recordings with manifest transcriptions", s["recordings_with_manifest_transcription"]),
        ("Hours with manifest transcriptions", f"{s['hours_with_manifest_transcription']} hours"),
        ("Hours with any caption/manifest text", f"{s['hours_with_any_caption_or_manifest_text']} hours"),
        ("Est. non-English speech",
         "n/a (language check not run; see section 10)" if s["est_non_english_speech_hours"] is None
         else f"{s['est_non_english_speech_hours']} hours ({_pct(s['share_of_speech_non_english'])} of speech)"),
        ("Est. non-English child-range speech",
         "n/a" if s["est_non_english_child_range_speech_hours"] is None
         else f"{s['est_non_english_child_range_speech_hours']} hours"),
        ("Speech with mixed/uncertain language",
         "n/a" if s["share_of_speech_mixed_or_uncertain"] is None
         else _pct(s["share_of_speech_mixed_or_uncertain"])),
        ("Distinct speakers (manifest metadata)",
         "n/a" if s["distinct_speakers_from_manifest"] is None else s["distinct_speakers_from_manifest"]),
    ]
    w = max(len(k) for k, _ in rows) + 3
    for k, v in rows:
        L.append(f"  {k + ':':<{w}}{v}")
    L.append("")

    inv = rep["inventory"]
    L.append("1. INVENTORY")
    L.append(f"  Total size: {inv['total_size_gb']} GB")
    L.append("  File status:")
    L.append(fmt_dict(inv["file_status"]))
    L.append("  Where the audio comes from:")
    keys = ["files", "audio_hours", "est_speech_hours"]
    if inv["by_source"] and "est_child_range_speech_hours" in inv["by_source"][0]:
        keys.append("est_child_range_speech_hours")
    L.append(fmt_rows([dict(r, range=r["source"]) for r in inv["by_source"]], keys))
    L.append("  File types (media with audio):")
    L.append(fmt_dict(inv["extensions"]))
    L.append("  Other file types present (most common):")
    L.append(fmt_dict(inv["other_file_types_top15"]))

    f = rep["formats"]
    L.append("2. AUDIO FORMATS")
    L.append("  Codecs:")
    L.append(fmt_dict(f["audio_codecs"]))
    L.append("  Original sample rates (Hz):")
    L.append(fmt_dict(f["original_sample_rates_hz"]))
    L.append("  Channels:")
    L.append(fmt_dict(f["channels"]))
    L.append("  Audio bitrate:")
    L.append(fmt_rows(f["audio_bitrate"], ["files", "hours"]))

    L.append("3. RECORDING LENGTHS")
    L.append(fmt_rows(rep["durations"], ["files", "hours"]))

    L.append("4. SPEECH ACTIVITY (share of each file that contains speech)")
    L.append(fmt_rows(rep["speech_activity"], ["files", "hours"]))

    sg = rep["speech_segments"]
    L.append("5. SPEECH SEGMENTS (continuous speech between pauses of 0.3 s or more)")
    L.append(fmt_rows(sg["runs_by_length"], ["speech_runs", "share_of_speech_time"]))
    L.append(f"  Pauses per minute of speech: "
             f"{'n/a' if sg['pauses_per_minute_of_speech'] is None else sg['pauses_per_minute_of_speech']}\n")

    q = rep["quality"]
    L.append("6. AUDIO QUALITY")
    L.append("  Estimated signal-to-noise ratio:")
    L.append(fmt_rows(q["estimated_snr"], ["files", "hours"]))
    L.append("  Speech loudness:")
    L.append(fmt_rows(q["speech_level"], ["files", "hours"]))
    L.append("  Clipping (distortion from recording too loud):")
    L.append(fmt_rows(q["clipping"], ["files", "hours"]))
    L.append("  Effective audio bandwidth (highest frequency actually present):")
    L.append(fmt_rows(q["audio_bandwidth"], ["files", "hours"]))

    p = rep["pitch"]
    L.append("7. VOICE PITCH AND CHILD / ADULT MIX (rough indicators)")
    if not rep["settings"].get("pitch_analysis"):
        L.append("    pitch analysis was turned off for this run\n")
    L.append("  Share of all voiced speech by pitch range:")
    L.append(fmt_rows(p["voiced_speech_by_pitch_range"], ["share_of_voiced_speech"]))
    L.append("  Files by how much of their voiced speech is in the child range (250 Hz+):")
    L.append(fmt_rows(p["files_by_child_adult_mix"], ["files", "speech_hours"]))

    el = rep["existing_labels"]
    L.append("8. EXISTING LABELS (caption timestamps only; caption text is never read into the report)")
    L.append("  Caption files by type:")
    L.append(fmt_dict(el["caption_files_by_type"]))
    L.append(f"  Recordings with a matching caption file: {el['recordings_with_matching_caption_files']} "
             f"({el['audio_hours_of_those_recordings']} hours of audio)")
    L.append(f"  Hours covered by caption timestamps: {el['hours_covered_by_caption_timestamps']}")
    L.append("  Possible transcript documents by type:")
    L.append(fmt_dict(el["document_files_by_type"]))
    L.append(f"  Recordings with a same-name document: {el['recordings_with_same_name_document']} "
             f"({el['audio_hours_of_those_recordings_doc']} hours of audio)")
    L.append(f"  Media files with embedded subtitle tracks: {el['media_files_with_embedded_subtitle_tracks']}")
    if el["notes"]:
        L.append("  Notes:")
        L.append(fmt_dict(el["notes"]))
    m = el["manifests"]
    L.append("")
    L.append("  DATASET MANIFESTS (JSON, JSON Lines, CSV, TSV; only the presence and length of")
    L.append("  text is measured, the text itself is never stored or reported)")
    if not m["manifests_used"]:
        L.append(f"    {m['files_examined']} JSON/CSV/TSV files examined; none linked entries to recordings.\n")
    else:
        L.append(f"  Files examined: {m['files_examined']}, used as manifests: {m['manifests_used']}")
        L.append(f"  Entries: {m['entries_in_manifests']}, linked to a recording: "
                 f"{m['entries_linked_to_a_recording']}")
        L.append(f"  Entries whose audio file was not found: {m['entries_whose_audio_was_not_found']}")
        L.append(f"  Entries matching several recordings (not counted): "
                 f"{m['entries_matching_several_recordings']}")
        L.append(f"  Entries with no audio reference: {m['entries_without_an_audio_reference']}")
        if m["files_skipped_too_large"] or m["files_unreadable"]:
            L.append(f"  Files skipped (too large / unreadable): {m['files_skipped_too_large']} / "
                     f"{m['files_unreadable']}")
        L.append("  Fields used:")
        for label, key in (("audio reference", "fields_used_for_audio_reference"),
                           ("transcription", "fields_used_for_transcription"),
                           ("prompt or script", "fields_used_for_prompt_or_script"),
                           ("metadata", "metadata_fields_found")):
            L.append(f"    {label}: {', '.join(m[key]) if m[key] else '(none)'}")
        L.append(f"  Recordings with a transcription: {m['recordings_with_transcription']} "
                 f"({m['hours_with_transcription']} hours)")
        L.append(f"  Recordings with only a prompt/script (read speech): "
                 f"{m['recordings_with_prompt_or_script_only']} ({m['hours_with_prompt_or_script_only']} hours)")
        L.append(f"  Recordings listed without any text: {m['recordings_listed_without_any_text']}")
        L.append("  Transcription length relative to speech:")
        L.append(fmt_rows(m["transcription_density"], ["files", "hours"]))
        if m["speakers"]:
            sp = m["speakers"]
            L.append(f"  Speakers (from manifest IDs): {sp['distinct_speakers']} distinct, "
                     f"median {sp['median_audio_minutes_per_speaker']} min of audio each")
            L.append(f"    share of hours from the top speaker: {_pct(sp['share_of_hours_from_top_speaker'])}; "
                     f"from the top 10% of speakers: {_pct(sp['share_of_hours_from_top_10_percent_of_speakers'])}")
        if m["age_groups"]:
            L.append("  Age groups (from manifest):")
            L.append(fmt_rows(m["age_groups"], ["files", "hours"]))
        if m["gender"]:
            L.append("  Gender (from manifest):")
            L.append(fmt_rows(m["gender"], ["files", "hours"]))
        if m["distinct_locations"]:
            L.append(f"  Distinct locations listed: {m['distinct_locations']} (names not shown)\n")
    L.append("")

    rd = rep["recording_dates"]
    L.append("9. RECORDING DATES (by year)")
    L.append(fmt_rows(rd["by_year"], ["files", "hours"]))
    L.append("  Date taken from (in order of preference):")
    L.append(fmt_dict(rd["date_source"]))
    if rd["manifest_year_differs_from_embedded_date"]:
        L.append(f"  Recordings where the manifest year differs from the embedded date: "
                 f"{rd['manifest_year_differs_from_embedded_date']} (manifest year used)\n")

    L.append("10. LANGUAGES (estimated speech hours; optional, run with --language-id)")
    lid = rep["language_id"]
    if lid.get("status") != "ok":
        L.append(f"    {lid.get('status')}\n")
    else:
        L.append(f"  Model: whisper-{lid['model']}; up to {lid['windows_per_file']} windows of 30 s "
                 f"per file, spread across its speech")
        L.append(f"  Files assessed: {lid['files_assessed']}, windows checked: {lid['windows_checked']}, "
                 f"speech assessed: {lid['speech_hours_assessed']} hours "
                 f"({_pct(lid['share_of_estimated_speech_assessed'])} of estimated speech)")
        L.append(f"  A window is English if English is the most likely language at "
                 f"{int(LID_EN_MIN * 100)}%+, non-English if English scores under "
                 f"{int(LID_NON_EN_MAX * 100)}%, otherwise mixed or uncertain.")
        L.append("  'All' columns extrapolate the assessed share to all estimated speech.")
        keys = ["speech_hours_assessed", "share", "est_speech_hours_all"]
        if lid["by_language_group"] and "est_child_range_speech_hours_all" in lid["by_language_group"][0]:
            keys.append("est_child_range_speech_hours_all")
        L.append(fmt_rows(lid["by_language_group"], keys))
        if lid["non_english_by_language"]:
            L.append("  Non-English speech by most likely language (labels unreliable, see below):")
            L.append(fmt_rows(lid["non_english_by_language"], keys))
        L.append("  IMPORTANT: " + lid["note"] + "\n")

    L.append("CAVEATS")
    L.append("  * All figures are automatic estimates, not measurements of transcribed data.")
    L.append("  * Speech detection can count music, TV, or other voices as speech.")
    L.append("  * Pitch overlaps between children and adult women, and shouting or singing")
    L.append("    raises pitch. Child-range figures are rough upper bounds; confirm with a")
    L.append("    listened-to sample.")
    L.append("  * Signal-to-noise is estimated from quiet vs. active parts of each file and")
    L.append("    is unreliable for files that are almost all speech or almost all noise.")
    L.append("  * Bandwidth is estimated from the average spectrum of speech; very quiet or")
    L.append("    muffled recordings can look narrower than they are.")
    L.append("  * Captions are matched to recordings by file name only. Caption coverage")
    L.append("    shows how much audio has some timed text, not whether the text is accurate")
    L.append("    or verbatim; subtitles are often condensed.")
    if rep["existing_labels"]["manifests"]["manifests_used"]:
        L.append("  * Manifest entries are linked to recordings by path, file name, or ID. A")
        L.append("    prompt or script shows what the speaker was asked to read, which may differ")
        L.append("    from what a child actually said. Speaker, age and gender figures come from")
        L.append("    the manifest as provided; groups under 5 recordings are combined.")
    L.append("  * Recording years come from manifest metadata where present, then from the")
    L.append("    recording's embedded date, and otherwise from file dates, which change")
    L.append("    when files are copied.")
    if rep["language_id"].get("status") == "ok":
        L.append("  * Language estimates come from a sample of 30 s windows. Language changes")
        L.append("    within a window, and short code-switches, are not resolved; see section 10")
        L.append("    for what Whisper can and cannot recognize.")
    if rep["settings"].get("sample_files") or rep["settings"].get("max_minutes_per_file"):
        L.append("  * Only part of the collection was analyzed (see settings); speech hours")
        L.append("    are extrapolated to the full collection.")
    L.append("\nSETTINGS")
    L.append(fmt_dict(rep["settings"]))
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def choose_folder_interactively():
    try:
        import tkinter as tk
        from tkinter import filedialog
        tkroot = tk.Tk()
        tkroot.withdraw()
        folder = filedialog.askdirectory(title="Choose the folder containing the recordings")
        tkroot.destroy()
        if folder:
            return folder
    except Exception:
        pass
    try:
        return input("Paste the path of the folder containing the recordings: ").strip().strip('"')
    except EOFError:
        return ""



def main():
    ap = argparse.ArgumentParser(description="Aggregate feasibility check of audio/video "
                                             "recordings for ASR fine-tuning.")
    ap.add_argument("folder", nargs="?", help="Folder to scan (opens a picker if omitted)")
    ap.add_argument("--output", help="Where to write the report (default: ./asr_feasibility_output)")
    ap.add_argument("--max-minutes-per-file", type=float, default=None,
                    help="Analyze at most this many minutes of each file (faster)")
    ap.add_argument("--sample-files", type=int, default=None,
                    help="Acoustically analyze a random sample of this many files (faster)")
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--vad", choices=["auto", "webrtc", "energy"], default="auto",
                    help="Speech detector (auto uses webrtcvad if installed)")
    ap.add_argument("--no-pitch", action="store_true", help="Skip pitch analysis")
    ap.add_argument("--language-id", action="store_true",
                    help="Estimate spoken languages with Whisper (needs faster-whisper)")
    ap.add_argument("--lid-model", default="small", help="Whisper model size for --language-id")
    ap.add_argument("--lid-windows-per-file", type=int, default=4,
                    help="30 s windows checked per file, spread across its speech (default 4)")
    ap.add_argument("--lid-max-windows", type=int, default=1500,
                    help="Overall window budget; files are sampled if needed (default 1500)")
    ap.add_argument("--fresh", action="store_true",
                    help="Ignore saved progress from an interrupted run and start over")
    ap.add_argument("--keep-checkpoint", action="store_true",
                    help="Keep the progress file after a successful run")
    ap.add_argument("--per-file", action="store_true",
                    help="Also write a LOCAL ONLY per-file table (not for sharing)")
    args = ap.parse_args()
    interactive = args.folder is None

    print(f"ASR data feasibility check v{TOOL_VERSION}\n")
    folder = args.folder or choose_folder_interactively()
    root = Path(folder).expanduser() if folder else None
    if root is None or not root.is_dir():
        print(f"Folder not found: {root}")
        return finish(interactive, 1)

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("ffmpeg was not found. Install it, or run: pip install imageio-ffmpeg")
        return finish(interactive, 1)
    ffprobe = find_ffprobe(ffmpeg)

    vad_method = args.vad
    if vad_method in ("auto", "webrtc"):
        try:
            import webrtcvad  # noqa: F401
            vad_method = "webrtc"
        except ImportError:
            if args.vad == "webrtc":
                print("webrtcvad is not installed; falling back to energy-based detection.")
            vad_method = "energy"

    out_dir = Path(args.output or (Path.cwd() / "asr_feasibility_output")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("Scanning folder...")
    media, labels, manifest_paths, other_ext = scan_folder(root, out_dir)
    print(f"  Found {len(media)} audio/video files.")
    if not media:
        print("No audio or video files found.")
        return finish(interactive, 1)

    # De-duplicate by content fingerprint
    records, seen, dup_count, aliases = [], {}, 0, []
    for p in media:
        try:
            size = p.stat().st_size
            fp = fingerprint(p, size)
            key = file_key(p, root)
        except OSError:
            size, fp, key = 0, None, None
        if fp and fp in seen:
            dup_count += 1
            aliases.append((p, seen[fp]))   # captions next to a copy still count
            continue
        ext = p.suffix.lower()
        try:
            fyear = file_year(p)
        except (OSError, ValueError, OverflowError):
            fyear = None
        rec = {"path": p, "ext": ext, "size": size, "key": key, "file_year": fyear,
               "kind": "video" if ext in VIDEO_EXT else "audio"}
        records.append(rec)
        if fp:
            seen[fp] = rec
    if dup_count:
        print(f"  Skipping {dup_count} duplicate copies.")

    # Saved progress from an interrupted run
    max_s = args.max_minutes_per_file * 60 if args.max_minutes_per_file else None
    ckpt_settings = {
        "tool_version": TOOL_VERSION,
        "folder_id": hashlib.sha1(str(root.resolve()).encode("utf-8")).hexdigest()[:16],
        "max_minutes_per_file": args.max_minutes_per_file,
        "speech_detector": vad_method,
        "pitch_analysis": not args.no_pitch,
        "format_reader": "ffprobe" if ffprobe else "ffmpeg",
    }
    ckpt = Checkpoint(out_dir, ckpt_settings, fresh=args.fresh)
    if ckpt.reason:
        print(f"  Note: {ckpt.reason}.")
    if ckpt.resumed:
        print(f"  Resuming from saved progress: {len(ckpt.analyses)} files already analyzed.")

    def interrupted():
        ckpt.close()
        print("\n\nStopped. Progress has been saved. Run the tool again on the same folder")
        print("with the same settings and it will continue where it left off.")
        return finish(interactive, 130)

    # Probe formats (all files; reuse saved results)
    print("Reading file formats...")
    def do_probe(r):
        if STOP.is_set():
            return None
        try:
            return probe_ffprobe(r["path"], ffprobe) if ffprobe else probe_ffmpeg_stderr(r["path"], ffmpeg)
        except Exception:
            return _empty_probe()

    todo = []
    for r in records:
        saved = ckpt.probes.get(r["key"]) if r["key"] else None
        if saved is not None:
            r["probe"] = saved
        else:
            todo.append(r)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        try:
            futs = {ex.submit(do_probe, r): r for r in todo}
            for fut in as_completed(futs):
                r, info = futs[fut], fut.result()
                if info is None:
                    continue
                r["probe"] = info
                if r["key"]:
                    ckpt.save_probe(r["key"], info)
        except KeyboardInterrupt:
            STOP.set()
            ex.shutdown(wait=True, cancel_futures=True)
            return interrupted()
    for r in records:
        if r["probe"]["has_video"]:
            r["kind"] = "video"

    # Acoustic analysis (all files, or a sample; reuse saved results)
    with_audio = [r for r in records if r["probe"]["has_audio"]]
    to_analyze = with_audio
    if args.sample_files and args.sample_files < len(with_audio):
        to_analyze = random.Random(0).sample(with_audio, args.sample_files)

    pending, done, hours = [], 0, 0.0
    for r in to_analyze:
        saved = ckpt.analyses.get(r["key"]) if r["key"] else None
        if saved is not None:
            r["analysis"] = saved
            done += 1
            hours += saved.get("analyzed_s", 0) / 3600
        else:
            pending.append(r)
    print(f"Analyzing audio in {len(to_analyze)} files (speech detector: {vad_method})...")
    if done:
        print(f"  {done} already done in the previous run, {len(pending)} remaining.")

    def do_analyze(r):
        try:
            res = analyze_file(r["path"], ffmpeg, vad_method, max_s, not args.no_pitch)
            res.pop("_frames", None)
            return res
        except _Stopped:
            return None
        except Exception as e:
            return {"status": "decode_error", "error_detail": type(e).__name__}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        try:
            futs = {ex.submit(do_analyze, r): r for r in pending}
            for fut in as_completed(futs):
                r, result = futs[fut], fut.result()
                if result is None:
                    continue
                r["analysis"] = result
                if r["key"]:
                    ckpt.save_analysis(r["key"], result)
                done += 1
                hours += result.get("analyzed_s", 0) / 3600
                elapsed = time.time() - t0
                print(f"  {done}/{len(to_analyze)} files, {hours:.1f} h of audio analyzed, "
                      f"{elapsed / 60:.1f} min elapsed", end="\r", flush=True)
        except KeyboardInterrupt:
            STOP.set()
            ex.shutdown(wait=True, cancel_futures=True)
            return interrupted()
    print()

    lid = {"status": "not run (enable with --language-id)"}
    if args.language_id:
        print("Estimating spoken languages (this can take a while)...")
        try:
            lid = run_language_id(to_analyze, ffmpeg, args.lid_model, args.lid_windows_per_file,
                                  args.lid_max_windows, ckpt)
        except (KeyboardInterrupt, _Stopped):
            STOP.set()
            return interrupted()

    settings = {
        "max_minutes_per_file": args.max_minutes_per_file,
        "sample_files": args.sample_files,
        "speech_detector": vad_method,
        "pitch_analysis": not args.no_pitch,
        "format_reader": "ffprobe" if ffprobe else "ffmpeg",
        "language_id": args.language_id,
    }
    index = {(str(r["path"].parent), r["path"].stem.lower()): r for r in records}
    for p, rec in aliases:
        index.setdefault((str(p.parent), p.stem.lower()), rec)
    label_stats = match_labels(labels, index) if labels else {}
    manifest_stats = {}
    if manifest_paths:
        print(f"Checking {len(manifest_paths)} JSON/CSV/TSV files for transcription manifests...")
        try:
            manifest_stats = match_manifests(manifest_paths, MediaIndex(root, records, aliases))
        except KeyboardInterrupt:
            return interrupted()
        if manifest_stats.get("manifests_used"):
            print(f"  {manifest_stats['manifests_used']} manifest(s) linked to "
                  f"{manifest_stats['entries_linked']} entries.")
    report = build_report(len(media), other_ext, records, dup_count, settings, lid,
                          label_stats, manifest_stats)

    json_path = out_dir / "feasibility_report.json"
    txt_path = out_dir / "feasibility_report.txt"
    write_atomic(json_path, json.dumps(report, indent=2))
    write_atomic(txt_path, render_text(report))
    ckpt.close(delete=not args.keep_checkpoint)

    if args.per_file:
        local = write_per_file_csv(out_dir, records, root)
        print(f"  Local-only per-file table: {local}")

    print(f"\nDone in {(time.time() - t0) / 60:.1f} minutes.")
    print(f"Report written to:\n  {txt_path}\n  {json_path}")
    print("\nPlease open the .txt report and read it before sharing the two report files.")
    return finish(interactive, 0)


def write_per_file_csv(out_dir, records, root):
    """LOCAL ONLY table with one row per recording, for the data owner's own use."""
    local = out_dir / "LOCAL_ONLY_per_file_details__DO_NOT_SHARE.csv"
    with open(local, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["relative_path", "type", "codec", "sample_rate", "duration_min",
                    "status", "speech_ratio", "snr_db", "speech_level_dbfs",
                    "clip_fraction", "median_pitch_hz", "child_range_share",
                    "bandwidth_hz", "has_caption_file", "caption_minutes",
                    "has_same_name_document", "year", "manifest_transcription",
                    "manifest_prompt_only", "manifest_text_chars", "lid_windows",
                    "lid_share_english", "lid_share_non_english", "lid_top_non_english"])
        for r in records:
            a = r.get("analysis") or {}
            pr = r["probe"]
            def rnd(v, n=2):
                return "" if v is None else round(v, n)
            w.writerow([str(r["path"].relative_to(root)), r["kind"], pr.get("audio_codec") or "",
                        pr.get("sample_rate") or "",
                        rnd((pr.get("duration_s") or 0) / 60), a.get("status") or ("could not be opened" if not pr.get("ok") else
                                             "no audio track" if not pr.get("has_audio")
                                             else "not in sample"),
                        rnd(a.get("speech_ratio"), 3), rnd(a.get("snr_db"), 1),
                        rnd(a.get("speech_level_dbfs"), 1), rnd(a.get("clip_fraction"), 5),
                        rnd(a.get("median_f0"), 0),
                        rnd(child_share(a) if a.get("status") == "ok" else None, 2),
                        a.get("bandwidth_hz") or "", bool(r.get("has_caption")),
                        rnd(r.get("caption_s", 0) / 60 if r.get("has_caption") else None, 1),
                        bool(r.get("has_doc")),
                        r.get("mf_year") or pr.get("meta_year") or r.get("file_year") or "",
                        bool(r.get("mf_text")),
                        bool(r.get("mf_prompt") and not r.get("mf_text")),
                        r.get("mf_chars", ""), *lid_columns(r)])
    return local


def lid_columns(r):
    wins = (r.get("lid") or {}).get("windows") or []
    if not wins:
        return ["", "", "", ""]
    labels = [w["label"] for w in wins]
    non = [l for l in labels if l not in ("en", "mixed")]
    return [len(wins), round(labels.count("en") / len(wins), 2), round(len(non) / len(wins), 2),
            Counter(non).most_common(1)[0][0] if non else ""]


def write_atomic(path, text):
    """Write via a temporary file so a power cut never leaves a half-written report."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def finish(interactive, code):
    if interactive or getattr(sys, "frozen", False):
        try:
            input("\nPress Enter to close.")
        except EOFError:
            pass
    return code


if __name__ == "__main__":
    sys.exit(main())
