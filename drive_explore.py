#!/usr/bin/env python3
"""
In-depth exploration of recordings in a Google Drive folder
===========================================================

For use when you are permitted to see the data itself. Like drive_feasibility.py,
it reads straight from Google Drive: each recording is downloaded to a temporary
folder, analyzed, and deleted. No audio is kept. Progress is saved after every
recording, so a crash or lost connection resumes where it stopped.

On top of the standard feasibility report it adds:

  * Turn-level child/adult segmentation: speech is split into turns, and each
    turn is classed as child-range, adult-range, mixed or unclear by pitch.
    No AI model is used.
  * Whisper-ready segments: proposed cuts under 30 seconds at natural pauses,
    exported as a NeMo-style JSON Lines manifest that refers to the Drive files
    (offset and duration), so no audio needs to be stored.
  * Transcript analysis: word counts, vocabulary, and language of each
    transcription or caption by word lists (English, Swahili, Sheng markers,
    plus any lists you supply), including the code-switching rate.
  * Near-duplicate and overlap detection: an audio fingerprint finds re-encoded
    copies and excerpts cut from longer recordings.
  * A folder map: hours, speech and child/adult turn hours per folder.
  * Optional language checks (--language-id): estimated English, non-English and
    mixed speech hours, also split by child-range and adult-range turns.

THE OUTPUTS CONTAIN PERSONAL DATA (file names, Drive links, transcript text in
the segment manifest, and per-recording measurements). Keep the output folder
on a machine and in a location your data agreement allows, never commit it to
GitHub, and delete it when the agreement says. --report-only writes only the
aggregate reports and the folder map.

Usage
-----
    python drive_explore.py "https://drive.google.com/drive/folders/XXXX"
    python drive_explore.py FOLDER_LINK --max-download-gb 20 --wordlists kenyan_words.json
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import random
import re
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
sys.path.insert(0, str(HERE))
import asr_feasibility as af       # noqa: E402  analysis and standard report
import drive_feasibility as dfz    # noqa: E402  Google Drive access and folder walk

EXPLORE_VERSION = "1.1.0"
FORCE_LID = False   # set by drive_explore_lid.py: language checks always on, no option to change it
LID_SKIP_REASON = None   # set by drive_explore_lid.py when the user chose to continue without them
FRAME_S = af.FRAME / af.SR                      # 30 ms

# --- turns
TURN_SMOOTH_FRAMES = 10         # +/- 300 ms window for the child-range share
TURN_MIN_FRAMES = 20            # turns shorter than 0.6 s merge into a neighbour
TURN_CHILD_MIN, TURN_ADULT_MAX = 0.70, 0.30
TURN_MIN_VOICED = 5
CLASS_NAMES = {"c": "child-range", "a": "adult-range", "m": "mixed", "u": "unclear (little voicing)"}

# --- segments
SEG_MAX_S, SEG_MIN_S, SEG_PAD_S, SEG_JOIN_GAP_S = 30.0, 1.0, 0.15, 0.5
SEG_SPLIT_FROM_S = 20.0         # long turns are split at the quietest point after 20 s

# --- fingerprints (speech-rhythm landmarks; robust to re-encoding, volume and
#     phone-quality copies, and they locate excerpts within longer recordings)
FP_PROMINENCE_DB = 4.0          # a landmark is a loudness peak at least 4 dB above its surroundings
FP_MIN_SEP = 4                  # landmarks at least 120 ms apart
FP_QUANTUM = 3                  # intervals between landmarks quantized to 90 ms
FP_TOLERANCE = 2                # +/- 60 ms alignment tolerance
FP_MIN_MATCHES = 12
FP_MIN_DENSITY = 0.25           # share of landmarks in the overlapping span that match
FP_STOP_FILES = 25              # hashes found in more files than this are ignored

# --- transcript word lists (lower case). Deliberately short lists of very common
# words; add other languages with --wordlists (see README).
WORDLISTS = {
    "English": """the and is are was were to of in it that this you he she they we my your
        what where when why how with for on at have has do does did not can will be an there
        here because but so very yes please teacher school mother father look see go come
        one two three four five""".split(),
    "Swahili": """na ya wa kwa ni za katika hii huu hiyo yake wake lakini kama sana pia hapa
        kuna yeye sisi wao mimi wewe nini nani wapi lini kwamba au bado tena sasa leo kesho
        jana ndiyo hapana asante habari nzuri kitu watu mtu watoto mtoto shule mwalimu kila
        hivyo ili hadi mpaka baada kabla kwenye moja mbili tatu nne tano""".split(),
    "Sheng (markers)": """manze msee wasee buda fiti mbogi ganji niaje rada mabeshte beshte
        ndai noma mathe chali doo mresh""".split(),
}
WORD_RE = re.compile(r"[a-z\u00c0-\u024f']+")


# ---------------------------------------------------------------------------
# Turns, segments and fingerprints from the frame-level analysis
# ---------------------------------------------------------------------------
def _runs(mask):
    """(start, end) frame pairs of speech, bridging pauses under 300 ms."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    if len(starts) == 0:
        return []
    keep = (starts[1:] - ends[:-1]) >= af.GAP_MERGE_FRAMES
    rs = np.concatenate([[starts[0]], starts[1:][keep]])
    re_ = np.concatenate([ends[:-1][keep], [ends[-1]]])
    return [(int(a), int(b)) for a, b in zip(rs, re_) if b - a >= af.MIN_RUN_FRAMES]


def turns_from_frames(fr):
    speech, f0 = fr["speech"], fr["f0"]
    turns = []
    for a, b in _runs(speech):
        seg = f0[a:b]
        voiced = ~np.isnan(seg)
        child = voiced & (np.nan_to_num(seg) >= af.CHILD_F0)
        cv = np.concatenate([[0], np.cumsum(voiced)])
        cc = np.concatenate([[0], np.cumsum(child)])
        n = b - a
        lo = np.maximum(np.arange(n) - TURN_SMOOTH_FRAMES, 0)
        hi = np.minimum(np.arange(n) + TURN_SMOOTH_FRAMES + 1, n)
        nv, nc = cv[hi] - cv[lo], cc[hi] - cc[lo]
        share = np.where(nv >= 3, nc / np.maximum(nv, 1), np.nan)
        lab = np.full(n, "", dtype="<U1")
        lab[share >= TURN_CHILD_MIN] = "c"
        lab[share <= TURN_ADULT_MAX] = "a"
        lab[(share > TURN_ADULT_MAX) & (share < TURN_CHILD_MIN)] = "m"
        last = ""                                   # fill unvoiced stretches
        for i in range(n):
            if lab[i]:
                last = lab[i]
            elif last:
                lab[i] = last
        first = next((x for x in lab if x), "u")
        lab[lab == ""] = first
        # constant-label pieces; merge pieces under 0.6 s into the longer neighbour
        pieces = []
        i = 0
        while i < n:
            j = i
            while j < n and lab[j] == lab[i]:
                j += 1
            pieces.append([i, j, lab[i]])
            i = j
        changed = True
        while changed and len(pieces) > 1:
            changed = False
            for k, (s0, e0, _) in enumerate(pieces):
                if e0 - s0 < TURN_MIN_FRAMES:
                    if k == 0:
                        nb = 1
                    elif k == len(pieces) - 1:
                        nb = k - 1
                    else:
                        left, right = pieces[k - 1], pieces[k + 1]
                        nb = k - 1 if left[1] - left[0] >= right[1] - right[0] else k + 1
                    lo_, hi_ = sorted((k, nb))
                    pieces[lo_] = [pieces[lo_][0], pieces[hi_][1], pieces[nb][2]]
                    del pieces[hi_]
                    changed = True
                    break
            merged = []                             # join equal neighbours
            for pc in pieces:
                if merged and merged[-1][2] == pc[2]:
                    merged[-1][1] = pc[1]
                else:
                    merged.append(pc)
            pieces = merged
        for s0, e0, _ in pieces:
            v = voiced[s0:e0]
            nvoiced = int(v.sum())
            if nvoiced < TURN_MIN_VOICED:
                cls = "u"
            else:
                sh = float(child[s0:e0].sum() / nvoiced)
                cls = "c" if sh >= TURN_CHILD_MIN else "a" if sh <= TURN_ADULT_MAX else "m"
            med = float(np.nanmedian(seg[s0:e0])) if nvoiced else None
            turns.append({"s": round((a + s0) * FRAME_S, 2), "e": round((a + e0) * FRAME_S, 2),
                          "c": cls, "f0": round(med) if med else None})
    return turns


def segments_from_turns(turns, db, duration_s):
    """Group same-class turns into segments of at most 30 s, splitting long turns
    at the quietest point after 20 s, then pad and drop very short ones."""
    groups = []
    for t in turns:
        g = groups[-1] if groups else None
        if g and g["c"] == t["c"] and t["s"] - g["e"] < SEG_JOIN_GAP_S and t["e"] - g["s"] <= SEG_MAX_S:
            g["e"] = t["e"]
            g["f0s"].append(t["f0"])
        else:
            groups.append({"s": t["s"], "e": t["e"], "c": t["c"], "f0s": [t["f0"]]})
    out = []
    for g in groups:
        s, e = g["s"], g["e"]
        while e - s > SEG_MAX_S:
            a = int((s + SEG_SPLIT_FROM_S) / FRAME_S)
            b = min(int((s + SEG_MAX_S) / FRAME_S), len(db))
            cut = (a + int(np.argmin(db[a:b]))) * FRAME_S if b > a else s + SEG_MAX_S
            out.append((s, cut, g))
            s = cut
        out.append((s, e, g))
    segs = []
    for s, e, g in out:
        s, e = max(0.0, s - SEG_PAD_S), min(duration_s, e + SEG_PAD_S)
        if e - s >= SEG_MIN_S:
            f0s = [x for x in g["f0s"] if x]
            segs.append({"offset": round(s, 2), "duration": round(e - s, 2), "class": g["c"],
                         "median_f0": int(np.median(f0s)) if f0s else None})
    return segs


def fingerprint(fr):
    """Hashes of speech rhythm: for each loudness peak (roughly a syllable), the
    quantized intervals to the next three peaks plus whether each is louder than
    the last. Returns (hashes, frame_times) as numpy arrays."""
    db = np.convolve(fr["db"], np.ones(3) / 3, "same")
    sp = fr["speech"]
    n = len(db)
    if n < 10:
        return np.empty(0, np.uint32), np.empty(0, np.int32)
    is_peak = np.zeros(n, bool)
    is_peak[1:-1] = (db[1:-1] >= db[:-2]) & (db[1:-1] >= db[2:]) & sp[1:-1]
    pad = np.pad(db, 5, mode="edge")
    local_min = np.min(np.stack([pad[k:k + n] for k in range(11)]), axis=0)
    cand = np.nonzero(is_peak & (db - local_min >= FP_PROMINENCE_DB))[0]
    ev = []
    for i in cand:
        if ev and i - ev[-1] < FP_MIN_SEP:
            if db[i] > db[ev[-1]]:
                ev[-1] = i
            continue
        ev.append(int(i))
    ev = np.array(ev, dtype=np.int64)
    if len(ev) < 4:
        return np.empty(0, np.uint32), np.empty(0, np.int32)
    win = np.stack([ev[k:len(ev) - 3 + k] for k in range(4)], axis=1)
    q = np.clip(np.round(np.diff(win, axis=1) / FP_QUANTUM).astype(np.int64), 1, 63)
    up = (np.diff(db[win], axis=1) > 0).astype(np.int64)
    h = (((q[:, 0] << 12) | (q[:, 1] << 6) | q[:, 2]) << 3) | (up[:, 0] << 2) | (up[:, 1] << 1) | up[:, 2]
    return h.astype(np.uint32), win[:, 0].astype(np.int32)


def _b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _unb64(text, dtype):
    return np.frombuffer(base64.b64decode(text), dtype=dtype) if text else np.empty(0, dtype)


def find_near_duplicates(fps, durations):
    """fps: {file index: (hashes, frame_times)}. Returns matched pairs."""
    if len(fps) < 2:
        return []
    H = np.concatenate([h for h, _ in fps.values()])
    F = np.concatenate([np.full(len(h), i, np.int32) for i, (h, _) in fps.items()])
    T = np.concatenate([t for _, t in fps.values()])
    order = np.argsort(H, kind="stable")
    H, F, T = H[order], F[order], T[order]
    votes = Counter()
    for g in np.split(np.arange(len(H)), np.flatnonzero(np.diff(H)) + 1):
        if len(g) < 2:
            continue
        files = F[g]
        nfiles = len(np.unique(files))
        if nfiles < 2 or nfiles > FP_STOP_FILES:
            continue
        for x in range(len(g)):
            for y in range(x + 1, len(g)):
                fa, fb = int(files[x]), int(files[y])
                if fa == fb:
                    continue
                ta, tb = int(T[g[x]]), int(T[g[y]])
                if fa > fb:
                    fa, fb, ta, tb = fb, fa, tb, ta
                votes[(fa, fb, ta - tb)] += 1
    by_pair = defaultdict(Counter)
    for (fa, fb, off), n in votes.items():
        by_pair[(fa, fb)][off] += n
    pairs = []
    for (fa, fb), offs in by_pair.items():
        def near(o):
            return sum(offs.get(o + k, 0) for k in range(-FP_TOLERANCE, FP_TOLERANCE + 1))
        best = max(offs, key=near)
        n = near(best)
        if n < FP_MIN_MATCHES:
            continue
        ha, ta = fps[fa]
        hb, tb = fps[fb]
        index = defaultdict(list)
        for h, t in zip(ha.tolist(), ta.tolist()):
            index[h].append(t)
        matched = [t for h, t in zip(hb.tolist(), tb.tolist())
                   if any(abs((t1 - t) - best) <= FP_TOLERANCE for t1 in index.get(h, ()))]
        if not matched:
            continue
        lo, hi = min(matched), max(matched)
        in_span = int(((tb >= lo) & (tb <= hi)).sum())
        density = len(matched) / max(in_span, 1)
        if density < FP_MIN_DENSITY:
            continue
        span = (hi - lo) * FRAME_S + 1.0
        da, dbb = durations.get(fa) or 0, durations.get(fb) or 0
        shorter = max(min(da, dbb), 1.0)
        similar = abs(da - dbb) <= 0.05 * shorter
        kind = "near-identical copy" if similar and span >= 0.6 * shorter else "partial overlap"
        if da < dbb:          # report the longer recording as A, so the offset is where B starts in A
            fa, fb, best = fb, fa, -best
        pairs.append({"a": fa, "b": fb, "matches": int(n), "density": round(density, 2),
                      "overlap_s": round(min(span, shorter), 1), "offset_s": round(best * FRAME_S, 2),
                      "kind": kind})
    return pairs


# ---------------------------------------------------------------------------
# Transcripts
# ---------------------------------------------------------------------------
_CUE_TS = re.compile(af._TS + r"\s*-->\s*" + af._TS)


def caption_cues(path):
    """(start, end, text) cues from SRT, VTT, ASS/SSA or SBV captions."""
    raw = path.read_bytes()[:20_000_000]
    text = raw.decode("utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8", "replace")
    ext, cues = path.suffix.lower(), []
    if ext in (".srt", ".vtt"):
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            m = _CUE_TS.search(lines[i])
            if m:
                g = m.groups()
                body = []
                i += 1
                while i < len(lines) and lines[i].strip():
                    body.append(re.sub(r"<[^>]+>", "", lines[i]))
                    i += 1
                cues.append((af._hmsf(*g[:4]), af._hmsf(*g[4:]), " ".join(body)))
            i += 1
    elif ext in (".ass", ".ssa"):
        for line in text.splitlines():
            if line.startswith("Dialogue:"):
                parts = line.split(",", 9)
                if len(parts) == 10:
                    t0 = re.match(r"(\d+):(\d{2}):(\d{2})\.(\d{1,3})", parts[1].strip())
                    t1 = re.match(r"(\d+):(\d{2}):(\d{2})\.(\d{1,3})", parts[2].strip())
                    if t0 and t1:
                        cues.append((af._hmsf(*t0.groups()), af._hmsf(*t1.groups()),
                                     re.sub(r"\{[^}]*\}", "", parts[9]).replace("\\N", " ")))
    elif ext == ".sbv":
        lines = text.splitlines()
        for i, line in enumerate(lines):
            m = af._RE_SBV.match(line.strip())
            if m:
                g = m.groups()
                cues.append((af._hmsf(*g[:4]), af._hmsf(*g[4:]), lines[i + 1] if i + 1 < len(lines) else ""))
    return [(a, b, t) for a, b, t in cues if b > a and t.strip()]


def classify_text(tokens, lists):
    counts = {lang: sum(1 for w in tokens if w in words) for lang, words in lists.items()}
    en, sw = counts.get("English", 0), counts.get("Swahili", 0)
    others = {k: v for k, v in counts.items() if k not in ("English", "Swahili") and v}
    if en and sw and min(en, sw) / (en + sw) >= 0.2:
        label = "English-Swahili mixed"
    elif en > sw:
        label = "English"
    elif sw > en:
        label = "Swahili"
    elif others:
        label = max(others, key=others.get)
    else:
        label = "unrecognized"
    return label, others


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="In-depth exploration of a Google Drive folder of recordings.")
    ap.add_argument("folder", nargs="?", help="Drive folder link or ID (asks if omitted)")
    ap.add_argument("--client-secret", default=str(dfz.APP_DIR / "client_secret.json"))
    ap.add_argument("--output", help="Output folder (default: ./drive_explore_output)")
    ap.add_argument("--work-dir", help="Where temporary downloads go (default: system temp)")
    ap.add_argument("--sample-files", type=int, default=None)
    ap.add_argument("--max-download-gb", type=float, default=None)
    ap.add_argument("--max-minutes-per-file", type=float, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--vad", choices=["auto", "webrtc", "energy"], default="auto")
    if not FORCE_LID:
        ap.add_argument("--language-id", action="store_true",
                        help="Estimate English vs. non-English speech hours (needs faster-whisper)")
    ap.add_argument("--lid-model", default="small")
    ap.add_argument("--lid-windows-per-file", type=int, default=4)
    ap.add_argument("--lid-max-windows", type=int, default=1500)
    ap.add_argument("--wordlists", help="JSON file of extra word lists: {\"Kikuyu\": [\"word\", ...], ...}")
    ap.add_argument("--no-fingerprints", action="store_true", help="Skip near-duplicate detection")
    ap.add_argument("--report-only", action="store_true",
                    help="Write only the aggregate reports and folder map (no per-recording files)")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--keep-checkpoint", action="store_true")
    ap.add_argument("--keep-login", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="Check that everything needed is installed, then exit (no sign-in)")
    args = ap.parse_args()
    if FORCE_LID:
        args.language_id = LID_SKIP_REASON is None
    if args.check:
        return dfz.self_check(args, name="drive_explore")
    interactive = args.folder is None

    print(f"In-depth Drive exploration v{EXPLORE_VERSION} (analysis v{af.TOOL_VERSION})\n")
    folder_text = args.folder
    if not folder_text:
        try:
            folder_text = input("Paste the link to the Google Drive folder: ")
        except EOFError:
            folder_text = ""
    root_id = dfz.parse_folder_id(folder_text)
    if not root_id:
        print("That doesn't look like a Google Drive folder link.")
        return af.finish(interactive, 1)

    lists = {k: set(v) for k, v in WORDLISTS.items()}
    if args.wordlists:
        try:
            extra = json.loads(Path(args.wordlists).read_text(encoding="utf-8"))
            for lang, words in extra.items():
                lists[str(lang)] = {str(w).lower() for w in words}
        except Exception as e:
            print(f"Could not read the word lists ({type(e).__name__}).")
            return af.finish(interactive, 1)

    ffmpeg = af.find_ffmpeg()
    if not ffmpeg:
        print("ffmpeg was not found. Install it, or run: pip install imageio-ffmpeg")
        return af.finish(interactive, 1)
    ffprobe = af.find_ffprobe(ffmpeg)
    vad = args.vad
    if vad in ("auto", "webrtc"):
        try:
            import webrtcvad  # noqa: F401
            vad = "webrtc"
        except ImportError:
            vad = "energy"

    out_dir = Path(args.output or (Path.cwd() / "drive_explore_output")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="asr_explore_", dir=args.work_dir)).resolve()
    shadow, dl_dir = work / "tree", work / "downloads"
    shadow.mkdir()
    dl_dir.mkdir()
    try:
        return run(args, interactive, root_id, lists, ffmpeg, ffprobe, vad, out_dir, shadow, dl_dir)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run(args, interactive, root_id, lists, ffmpeg, ffprobe, vad, out_dir, shadow, dl_dir):
    t0 = time.time()
    try:
        drive = dfz.make_client(args)
    except SystemExit as e:
        print(e)
        return af.finish(interactive, 1)
    except KeyboardInterrupt:
        return af.finish(interactive, 130)
    except Exception as e:
        print(f"Sign-in did not complete ({type(e).__name__}).")
        return af.finish(interactive, 1)

    print("Listing files in the Drive folder...")
    try:
        items, n_folders = dfz.walk(drive, root_id)
    except af._Stopped:
        return af.finish(interactive, 130)
    except Exception as e:
        print(f"Could not list the folder ({type(e).__name__}).")
        return af.finish(interactive, 1)
    print(f"  {len(items)} files in {n_folders} folders.")

    # --- sort items; fetch small label files
    media, captions, docs, manifests_src, other_ext = [], [], [], [], Counter()
    for it in items:
        mime = it.get("mimeType", "")
        if mime == dfz.GDOC_MIME:
            other_ext[".gdoc"] += 1
            docs.append(shadow / (it["rel"] + ".gdoc"))
            continue
        if mime == dfz.GSHEET_MIME:
            other_ext[".gsheet"] += 1
            manifests_src.append((it, shadow / (it["rel"] + ".csv"), "sheet"))
            continue
        ext = dfz.media_ext(it)
        if ext:
            media.append((it, ext))
            continue
        fext = Path(it["name"]).suffix.lower() or "(no extension)"
        other_ext[fext] += 1
        if fext in af.CAPTION_EXT:
            captions.append((it, shadow / it["rel"]))
        elif fext in af.DOC_EXT:
            docs.append(shadow / it["rel"])
        elif fext in af.MANIFEST_EXT:
            manifests_src.append((it, shadow / it["rel"], "file"))

    def fetch(it, p, kind, limit):
        if int(it.get("size") or 0) > limit:
            return False
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            (drive.export_csv if kind == "sheet" else drive.download)(it["id"], p)
            return True
        except af._Stopped:
            raise
        except Exception:
            return False

    caption_paths, manifest_paths = [], []
    if captions or manifests_src:
        print(f"Downloading {len(captions) + len(manifests_src)} caption/manifest files...")
    try:
        caption_paths = [p for it, p in captions if fetch(it, p, "file", dfz.CAPTION_MAX_BYTES)]
        manifest_paths = [p for it, p, k in manifests_src if fetch(it, p, k, af.MANIFEST_MAX_BYTES)]
    except af._Stopped:
        return af.finish(interactive, 130)

    # --- recordings (deduplicated by Drive checksum)
    records, seen, aliases, dup_count = [], {}, [], 0
    for it, ext in sorted(media, key=lambda t: t[0]["rel"]):
        md5, path = it.get("md5Checksum"), shadow / it["rel"]
        if md5 and md5 in seen:
            dup_count += 1
            aliases.append((path, seen[md5]))
            continue
        mime = (it.get("mimeType") or "").lower()
        dur_ms = (it.get("videoMediaMetadata") or {}).get("durationMillis")
        rec = {"path": path, "display_path": it["rel"], "ext": ext, "size": int(it.get("size") or 0),
               "key": hashlib.sha1(f"{it['id']}|{md5}|{it.get('modifiedTime')}".encode()).hexdigest()[:24],
               "file_year": dfz.drive_year(it), "drive_id": it["id"],
               "kind": "video" if ext in af.VIDEO_EXT and not mime.startswith("audio/") else "audio",
               "drive_duration_s": int(dur_ms) / 1000 if dur_ms else None,
               "can_download": (it.get("capabilities") or {}).get("canDownload", True)}
        records.append(rec)
        if md5:
            seen[md5] = rec
    if not records:
        print("No audio or video files found in that folder.")
        return af.finish(interactive, 1)

    order = [r for r in records if r["can_download"]]
    random.Random(0).shuffle(order)
    if args.sample_files:
        order = order[:args.sample_files]
    if args.max_download_gb:
        budget, chosen, used = args.max_download_gb * 1e9, [], 0
        for r in order:
            if used + r["size"] <= budget:
                chosen.append(r)
                used += r["size"]
        order = chosen
    print(f"Recordings: {len(records)} ({sum(r['size'] for r in records) / 1e9:.1f} GB); "
          f"selected for analysis: {len(order)} ({sum(r['size'] for r in order) / 1e9:.1f} GB).")

    max_s = args.max_minutes_per_file * 60 if args.max_minutes_per_file else None
    ckpt = af.Checkpoint(out_dir, {
        "tool_version": af.TOOL_VERSION, "explore_version": EXPLORE_VERSION, "source": "drive-explore",
        "folder_id": hashlib.sha1(root_id.encode()).hexdigest()[:16],
        "max_minutes_per_file": args.max_minutes_per_file, "speech_detector": vad,
        "format_reader": "ffprobe" if ffprobe else "ffmpeg",
        "fingerprints": not args.no_fingerprints,
        "language_id": af.lid_signature(args.lid_model, args.lid_windows_per_file)
        if args.language_id else None}, fresh=args.fresh)
    if ckpt.reason:
        print(f"  Note: {ckpt.reason}.")
    saved_x = ckpt.extra.get("explore", {})

    import threading
    lid_sig = af.lid_signature(args.lid_model, args.lid_windows_per_file) if args.language_id else None
    model, lid_status = None, {"status": LID_SKIP_REASON or "not run (enable with --language-id)"}
    lid_lock, budget = threading.Lock(), {"used": 0}
    if args.language_id:
        print("Loading the language model...")
        model, err = af.load_lid_model(args.lid_model)
        lid_status = {"status": err} if model is None else af.lid_meta(
            args.lid_model, args.lid_windows_per_file, args.lid_max_windows, 0)

    pending, done = [], 0
    for r in order:
        if r["key"] in ckpt.analyses and r["key"] in saved_x:
            r["probe"] = ckpt.probes.get(r["key"]) or af._empty_probe()
            r["analysis"], r["x"] = ckpt.analyses[r["key"]], saved_x[r["key"]]
            lid = ckpt.lids.get(f"{r['key']}|{lid_sig}") if lid_sig else None
            if lid is not None:
                r["lid"] = lid
                budget["used"] += len(lid.get("windows") or [])
            done += 1
        else:
            pending.append(r)
    if done:
        print(f"  Resuming: {done} recordings already analyzed, {len(pending)} remaining.")

    def process(r):
        if af.STOP.is_set():
            return None
        tmp = dl_dir / f"{r['key']}{r['ext'] if r['ext'] not in ('.audio', '.video') else ''}"
        try:
            try:
                drive.download(r["drive_id"], tmp)
            except af._Stopped:
                return None
            except Exception as e:
                return {"probe": dict(af._empty_probe(), ok=True, has_audio=True),
                        "analysis": {"status": "download_error", "error_detail": type(e).__name__}, "x": {}}
            try:
                probe = af.probe_ffprobe(tmp, ffprobe) if ffprobe else af.probe_ffmpeg_stderr(tmp, ffmpeg)
            except Exception:
                probe = af._empty_probe()
            if not probe["has_audio"]:
                return {"probe": probe, "analysis": {"status": "no_audio"}, "x": {}}
            a = af.analyze_file(tmp, ffmpeg, vad, max_s, True, keep_frames=True)
            fr = a.pop("_frames", None)
            x = {}
            if a.get("status") == "ok" and fr is not None:
                dur = a["analyzed_s"]
                turns = turns_from_frames(fr)
                x["turns"] = [[t["s"], t["e"], t["c"]] for t in turns]
                x["segments"] = segments_from_turns(turns, fr["db"], dur)
                if not args.no_fingerprints:
                    h, t = fingerprint(fr)
                    x["fp_h"], x["fp_t"] = _b64(h.astype(np.uint32)), _b64(t.astype(np.int32))
            lid = None
            if model is not None and a.get("status") == "ok":
                n = len(af.lid_windows(a, args.lid_windows_per_file))
                take = False
                with lid_lock:   # files arrive in random order, so the budget gives a random sample
                    if n and budget["used"] + n <= args.lid_max_windows:
                        budget["used"] += n
                        take = True
                lid = (af.lid_for_file(tmp, ffmpeg, a, model, args.lid_windows_per_file)
                       if take else {"status": "not_selected", "windows": []})
            return {"probe": probe, "analysis": a, "x": x, "lid": lid}
        except af._Stopped:
            return None
        except Exception as e:
            return {"probe": dict(af._empty_probe(), ok=True, has_audio=True),
                    "analysis": {"status": "decode_error", "error_detail": type(e).__name__}, "x": {}}
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

    if pending:
        print(f"Downloading and analyzing {len(pending)} recordings (each deleted after analysis)...")
    gb = 0.0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        try:
            futs = {ex.submit(process, r): r for r in pending}
            for fut in as_completed(futs):
                r, res = futs[fut], fut.result()
                if res is None:
                    continue
                r["probe"], r["analysis"], r["x"] = res["probe"], res["analysis"], res["x"]
                if res.get("lid") is not None:
                    r["lid"] = res["lid"]
                if r["analysis"].get("status") != "download_error":
                    ckpt.save_probe(r["key"], r["probe"])
                    ckpt.save_analysis(r["key"], r["analysis"])
                    if lid_sig and res.get("lid") is not None:     # saved before "explore",
                        ckpt.save_lid(f"{r['key']}|{lid_sig}", res["lid"])   # which marks a file done
                    ckpt.save("explore", r["key"], r["x"])
                done += 1
                gb += r["size"] / 1e9
                print(f"  {done}/{len(order)} recordings, {gb:.1f} GB downloaded, "
                      f"{(time.time() - t0) / 60:.1f} min", end="\r", flush=True)
        except KeyboardInterrupt:
            af.STOP.set()
            ex.shutdown(wait=True, cancel_futures=True)
            ckpt.close()
            print("\n\nStopped. Progress has been saved; run again with the same folder and settings.")
            return af.finish(interactive, 130)
    print()

    # --- recordings not analyzed: totals only, with the same duration estimates
    #     as drive_feasibility.py (Drive's video durations, or from file size)
    estimated = dfz.fill_unanalyzed(records)

    # --- labels: standard matching, plus text for transcript analysis
    index = {(str(r["path"].parent), r["path"].stem.lower()): r for r in records}
    for p, rec in aliases:
        index.setdefault((str(p.parent), p.stem.lower()), rec)
    label_stats = af.match_labels(caption_paths + docs, index) if (caption_paths or docs) else {}
    manifest_stats = af.match_manifests(manifest_paths, af.MediaIndex(shadow, records, aliases),
                                        keep_text=True) if manifest_paths else {}
    for p in caption_paths:
        stem = p.stem.lower()
        rec = None
        while True:
            rec = index.get((str(p.parent), stem))
            if rec is not None or "." not in stem:
                break
            stem = stem.rsplit(".", 1)[0]
        if rec is not None:
            try:
                rec.setdefault("cues", []).extend(caption_cues(p))
            except OSError:
                pass

    settings = {"source": "Google Drive (in-depth exploration)",
                "max_minutes_per_file": args.max_minutes_per_file,
                "sample_files": args.sample_files or (len(order) if len(order) < len(records) else None),
                "max_download_gb": args.max_download_gb, "speech_detector": vad,
                "pitch_analysis": True, "format_reader": "ffprobe" if ffprobe else "ffmpeg",
                "language_id": args.language_id}
    if model is not None:
        lid_status["files_selected"] = sum(1 for r in records if (r.get("lid") or {}).get("status") == "ok")
    std = af.build_report(len(media), other_ext, records, dup_count, settings,
                          lid_status, label_stats, manifest_stats)
    std["google_drive"] = {"folders_scanned": n_folders, "files_listed": len(items),
                           "recordings_selected_for_analysis": len(order),
                           "recordings_not_downloadable": sum(1 for r in records if not r["can_download"]),
                           "durations_estimated_from_file_size": estimated}
    explore = build_explore(records, lists, args)

    af.write_atomic(out_dir / "feasibility_report.json", json.dumps(std, indent=2))
    af.write_atomic(out_dir / "feasibility_report.txt", af.render_text(std))
    af.write_atomic(out_dir / "exploration_report.json", json.dumps(explore["report"], indent=2))
    af.write_atomic(out_dir / "exploration_report.txt", render_explore(explore["report"]))
    write_folder_map(out_dir / "folder_summary.csv", explore["folders"])
    written = ["exploration_report.txt", "exploration_report.json", "feasibility_report.txt",
               "feasibility_report.json", "folder_summary.csv"]
    if not args.report_only:
        written += write_detail_files(out_dir, records, explore)
    (out_dir / "README_FIRST.txt").write_text(
        "This folder contains PERSONAL DATA about the recordings (file names, Drive links,\n"
        "transcript text and per-recording measurements). Keep it only where your data\n"
        "agreement allows, never upload it to GitHub, and delete it when required.\n", encoding="utf-8")
    ckpt.close(delete=not args.keep_checkpoint)
    if not args.keep_login:
        try:
            dfz.TOKEN_PATH.unlink()
        except OSError:
            pass
    print(f"\nDone in {(time.time() - t0) / 60:.1f} minutes. Written to {out_dir}:")
    for w in written:
        print(f"  {w}")
    return af.finish(interactive, 0)


# ---------------------------------------------------------------------------
# Aggregation and outputs
# ---------------------------------------------------------------------------
def _h(sec):
    return round(sec / 3600, 2)


def lid_shares(r):
    """(English, non-English, mixed) shares of a file's checked windows, the most
    likely non-English language, and the file's majority label; None if unchecked."""
    res = r.get("lid") or {}
    wins = res.get("windows") or []
    if res.get("status") != "ok" or not wins:
        return None
    labels = [w["label"] for w in wins]
    non = [l for l in labels if l not in ("en", "mixed")]
    n = len(labels)
    majority = Counter(labels).most_common(1)[0][0]
    return (labels.count("en") / n, len(non) / n, labels.count("mixed") / n,
            Counter(non).most_common(1)[0][0] if non else None, majority)


def build_explore(records, lists, args):
    analyzed = [r for r in records if (r.get("analysis") or {}).get("status") == "ok" and r.get("x")]
    an_h = sum(r["analysis"]["analyzed_s"] for r in analyzed) / 3600
    tot_h = sum((r["probe"].get("duration_s") or 0) for r in records if r["probe"].get("has_audio")) / 3600
    scale = tot_h / an_h if an_h else 0.0

    # turns
    cls_s, cls_n, turn_len = Counter(), Counter(), defaultdict(list)
    switches, speech_min = 0, 0.0
    for r in analyzed:
        prev = None
        for s, e, c in r["x"].get("turns", []):
            cls_s[c] += e - s
            cls_n[c] += 1
            turn_len[c].append(e - s)
            if prev and c in "ca" and prev in "ca" and c != prev:
                switches += 1
            if c in "ca":
                prev = c
            speech_min += (e - s) / 60
    turn_rows = []
    for c in "camu":
        lens = sorted(turn_len[c])
        turn_rows.append({"class": CLASS_NAMES[c], "turns": cls_n[c], "hours_analyzed": _h(cls_s[c]),
                          "est_hours_all": round(_h(cls_s[c]) * scale, 2),
                          "median_turn_s": round(lens[len(lens) // 2], 1) if lens else None})
    child_len_buckets = Counter()
    for L in turn_len["c"]:
        child_len_buckets[af.bucket_label(L, [(1, "under 1 s"), (3, "1 - 3 s"), (10, "3 - 10 s"),
                                              (af.INF, "10 s and over")])] += L

    # segments
    seg_s, seg_n, seg_len = Counter(), Counter(), Counter()
    for r in analyzed:
        for sg in r["x"].get("segments", []):
            seg_s[sg["class"]] += sg["duration"]
            seg_n[sg["class"]] += 1
            seg_len[af.bucket_label(sg["duration"], [(5, "1 - 5 s"), (15, "5 - 15 s"),
                                                       (af.INF, "15 - 30 s")])] += 1
    seg_rows = [{"class": CLASS_NAMES[c], "segments": seg_n[c], "hours": _h(seg_s[c])} for c in "camu"]

    # transcripts
    lang_n, lang_s, other_markers = Counter(), Counter(), Counter()
    words, vocab, utter, words_total = Counter(), set(), 0, 0
    for r in records:
        units = []
        dur = r["probe"].get("duration_s") or 0
        for t in r.get("mf_texts", []):
            units.append((t, dur))
        for a, b, t in r.get("cues", []):
            units.append((t, b - a))
        for text, d in units:
            toks = WORD_RE.findall(text.lower())
            if not toks:
                continue
            label, others = classify_text(toks, lists)
            lang_n[label] += 1
            lang_s[label] += d
            for k in others:
                other_markers[k] += 1
            words.update(toks)
            vocab.update(toks)
            utter += 1
            words_total += len(toks)
    lang_rows = [{"language": k, "utterances": n, "hours": _h(lang_s[k]),
                  "share": round(n / utter, 3) if utter else 0} for k, n in lang_n.most_common()]

    # near-duplicates
    pairs = []
    if not args.no_fingerprints:
        fps, durs = {}, {}
        for i, r in enumerate(analyzed):
            h = _unb64(r["x"].get("fp_h", ""), np.uint32)
            t = _unb64(r["x"].get("fp_t", ""), np.int32)
            if len(h):
                fps[i] = (h, t)
                durs[i] = r["analysis"]["analyzed_s"]
        pairs = find_near_duplicates(fps, durs)
        for p in pairs:
            p["a_rec"], p["b_rec"] = analyzed[p["a"]], analyzed[p["b"]]
    near = [p for p in pairs if p["kind"] == "near-identical copy"]
    part = [p for p in pairs if p["kind"] == "partial overlap"]

    # folders
    folders = defaultdict(lambda: Counter())
    for r in records:
        if not r["probe"].get("has_audio"):
            continue
        parts = r["display_path"].split("/")
        keys = [("top", parts[0] if len(parts) > 1 else "(top level)"),
                ("leaf", "/".join(parts[:-1]) or "(top level)")]
        a = r.get("analysis") or {}
        for k in keys:
            f = folders[k]
            f["recordings"] += 1
            f["audio_s"] += r["probe"].get("duration_s") or 0
            if a.get("status") == "ok":
                f["analyzed"] += 1
                f["speech_s"] += a["analyzed_s"] * a["speech_ratio"]
                for s, e, c in (r.get("x") or {}).get("turns", []):
                    f[f"turn_{c}"] += e - s
                if a.get("snr_db") is not None:
                    f["snr_sum"] += a["snr_db"]
                    f["snr_n"] += 1
                if (a.get("bandwidth_hz") or 99999) < 5000:
                    f["narrow"] += 1
            if r.get("mf_text") or r.get("caption_s"):
                f["labeled"] += 1
            sh = lid_shares(r) if a.get("status") == "ok" else None
            if sh is not None:
                f["lid_files"] += 1
                f["lid_speech_s"] += a["analyzed_s"] * a["speech_ratio"]
                f["non_en_s"] += a["analyzed_s"] * a["speech_ratio"] * sh[1]

    # language by turn class (files with language checks only)
    lang_cls = defaultdict(Counter)       # class -> {English, non-English, mixed}
    child_by_lang = Counter()
    checked = 0
    for r in analyzed:
        sh = lid_shares(r)
        if sh is None:
            continue
        checked += 1
        en, non, mixed, top_non, _ = sh
        tc = Counter()
        for s0, e0, c in r["x"].get("turns", []):
            tc[c] += e0 - s0
        for c in "ca":
            lang_cls[c]["English"] += tc[c] * en
            lang_cls[c]["non-English"] += tc[c] * non
            lang_cls[c]["mixed or uncertain"] += tc[c] * mixed
        if top_non:
            child_by_lang[top_non] += tc["c"] * non
    language = None
    if checked:
        language = {
            "files_checked": checked,
            "by_turn_class": [{"class": CLASS_NAMES[c], **{k: _h(v) for k, v in lang_cls[c].items()}}
                              for c in "ca"],
            "child_range_non_english_by_language": {af.lang_name(k): _h(v)
                                                    for k, v in child_by_lang.most_common()},
        }

    report = {
        "tool": {"explore_version": EXPLORE_VERSION, "analysis_version": af.TOOL_VERSION},
        "language": language,
        "language_skipped": LID_SKIP_REASON,
        "coverage": {"recordings_analyzed": len(analyzed), "hours_analyzed": round(an_h, 2),
                     "total_audio_hours": round(tot_h, 2), "extrapolation_factor": round(scale, 3)},
        "turns": {"by_class": turn_rows, "child_adult_switches_per_minute":
                  round(switches / speech_min, 2) if speech_min else None,
                  "child_speech_by_turn_length_hours": {k: _h(v) for k, v in child_len_buckets.items()}},
        "segments": {"by_class": seg_rows, "by_length": dict(seg_len),
                     "rules": f"same-class turns joined across pauses under {SEG_JOIN_GAP_S} s, "
                              f"max {SEG_MAX_S:.0f} s, min {SEG_MIN_S:.0f} s, {SEG_PAD_S} s padding"},
        "transcripts": {"utterances": utter, "words": words_total, "vocabulary_size": len(vocab),
                        "by_language": lang_rows, "utterances_with_other_language_markers": dict(other_markers),
                        "code_switching_rate": round(lang_n["English-Swahili mixed"] / utter, 3) if utter else None,
                        "word_lists": {k: len(v) for k, v in lists.items()}},
        "near_duplicates": {"near_identical_pairs": len(near), "partial_overlap_pairs": len(part),
                            "hours_in_near_identical_copies": _h(sum(min(p["a_rec"]["analysis"]["analyzed_s"],
                                                                         p["b_rec"]["analysis"]["analyzed_s"]) for p in near)),
                            "enabled": not args.no_fingerprints},
    }
    return {"report": report, "folders": folders, "pairs": pairs, "words": words}


def render_explore(rep):
    L = ["IN-DEPTH EXPLORATION REPORT", "=" * 64,
         f"Explorer v{rep['tool']['explore_version']}, analysis v{rep['tool']['analysis_version']}", ""]
    c = rep["coverage"]
    L.append(f"Recordings analyzed: {c['recordings_analyzed']} ({c['hours_analyzed']} h) of "
             f"{c['total_audio_hours']} h of audio. 'All' columns scale by {c['extrapolation_factor']}.\n")

    def table(rows, keys):
        w = max(len(str(r[keys[0]])) for r in rows) + 2 if rows else 10
        out = [f"    {str(r[keys[0]]):<{w}}" + "   ".join(f"{k.replace('_', ' ')}: "
               f"{'-' if r[k] is None else r[k]}" for k in keys[1:]) for r in rows]
        return "\n".join(out) + "\n" if rows else "    (none)\n"

    t = rep["turns"]
    L.append("1. TURNS BY PITCH CLASS (no AI model; child-range = 250 Hz and above)")
    L.append(table(t["by_class"], ["class", "turns", "hours_analyzed", "est_hours_all", "median_turn_s"]))
    L.append(f"  Switches between child-range and adult-range turns per minute of speech: "
             f"{t['child_adult_switches_per_minute']}")
    L.append("  Child-range speech by turn length (hours): " +
             ", ".join(f"{k}: {v}" for k, v in t["child_speech_by_turn_length_hours"].items()) + "\n")
    sg = rep["segments"]
    L.append("2. WHISPER-READY SEGMENTS (" + sg["rules"] + ")")
    L.append(table(sg["by_class"], ["class", "segments", "hours"]))
    L.append("  By length: " + ", ".join(f"{k}: {v}" for k, v in sg["by_length"].items()) + "\n")
    tr = rep["transcripts"]
    L.append("3. TRANSCRIPTS AND CAPTIONS (language by word lists)")
    L.append(f"  Utterances: {tr['utterances']}, words: {tr['words']}, vocabulary: {tr['vocabulary_size']}")
    L.append(table(tr["by_language"], ["language", "utterances", "hours", "share"]))
    rate = tr["code_switching_rate"]
    L.append(f"  English-Swahili code-switching rate: {'-' if rate is None else f'{rate * 100:.0f}%'}")
    if tr["utterances_with_other_language_markers"]:
        L.append("  Utterances containing words from other lists: " +
                 ", ".join(f"{k}: {v}" for k, v in tr["utterances_with_other_language_markers"].items()))
    L.append("  Word lists used (words): " + ", ".join(f"{k} ({v})" for k, v in tr["word_lists"].items()) + "\n")
    nd = rep["near_duplicates"]
    L.append("4. NEAR-DUPLICATES AND OVERLAPS (audio fingerprint)")
    if not nd["enabled"]:
        L.append("    turned off for this run\n")
    else:
        L.append(f"  Near-identical copies (e.g. re-encoded): {nd['near_identical_pairs']} pairs, "
                 f"{nd['hours_in_near_identical_copies']} h")
        L.append(f"  Partial overlaps (e.g. excerpts of longer recordings): {nd['partial_overlap_pairs']} pairs\n")
    L.append("5. FOLDERS: see folder_summary.csv\n")
    lg = rep.get("language")
    L.append("6. LANGUAGE BY SPEAKER CLASS (hours; files with language checks only)")
    if not lg:
        L.append(f"    {rep.get('language_skipped') or 'language checks not run (use --language-id)'}\n")
    else:
        L.append(f"  Files checked: {lg['files_checked']}. Each file's turn hours are divided by its")
        L.append("  checked windows' labels; see section 10 of the feasibility report for totals.")
        for row in lg["by_turn_class"]:
            L.append(f"    {row['class']:<14}" + "   ".join(f"{k}: {v}" for k, v in row.items() if k != "class"))
        if lg["child_range_non_english_by_language"]:
            L.append("  Child-range non-English hours by most likely language (labels unreliable):")
            L.append("    " + ", ".join(f"{k}: {v}" for k, v in lg["child_range_non_english_by_language"].items()))
        L.append("")
    L.append("NOTES")
    L.append("  * Pitch classes are acoustic, not speaker identities: adult women overlap the")
    L.append("    lower child range, and shouting or singing raises pitch. Spot-check turns")
    L.append("    by listening in Google Drive (links in recordings.csv).")
    L.append("  * Word-list language labels only cover the listed languages and need")
    L.append("    transcripts; Sheng markers are a rough signal.")
    L.append("  * Exact copies are removed before analysis using Drive checksums; the")
    L.append("    fingerprint finds copies that differ in encoding or length.")
    return "\n".join(L) + "\n"


def write_folder_map(path, folders):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["level", "folder", "recordings", "analyzed", "audio_hours", "speech_hours",
                    "child_range_turn_hours", "adult_range_turn_hours", "mixed_turn_hours",
                    "mean_snr_db", "narrowband_recordings", "labeled_recordings",
                    "language_checked_recordings", "non_english_share_of_checked_speech"])
        for (level, name), f in sorted(folders.items(), key=lambda kv: (kv[0][0] != "top", -kv[1]["speech_s"])):
            w.writerow([level, name, f["recordings"], f["analyzed"], _h(f["audio_s"]), _h(f["speech_s"]),
                        _h(f["turn_c"]), _h(f["turn_a"]), _h(f["turn_m"]),
                        round(f["snr_sum"] / f["snr_n"], 1) if f["snr_n"] else "",
                        f["narrow"], f["labeled"], f["lid_files"],
                        round(f["non_en_s"] / f["lid_speech_s"], 3) if f["lid_speech_s"] else ""])


def write_detail_files(out_dir, records, explore):
    link = "https://drive.google.com/file/d/{}/view"
    dup_of = {}
    for p in explore["pairs"]:
        for x, y in (("a_rec", "b_rec"), ("b_rec", "a_rec")):
            dup_of.setdefault(p[x]["display_path"], []).append(f"{p['kind']}: {p[y]['display_path']}")
    with open(out_dir / "recordings.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "drive_link", "status", "duration_min", "speech_ratio", "child_range_turn_min",
                    "adult_range_turn_min", "mixed_turn_min", "segments", "snr_db", "bandwidth_hz",
                    "year", "has_transcript", "caption_cues", "near_duplicates", "lid_windows",
                    "lid_share_english", "lid_share_non_english", "lid_top_non_english"])
        for r in records:
            a, x = r.get("analysis") or {}, r.get("x") or {}
            tc = Counter()
            for s, e, c in x.get("turns", []):
                tc[c] += e - s
            w.writerow([r["display_path"], link.format(r["drive_id"]), a.get("status", "not analyzed"),
                        round((r["probe"].get("duration_s") or 0) / 60, 2),
                        a.get("speech_ratio", ""), round(tc["c"] / 60, 2), round(tc["a"] / 60, 2),
                        round(tc["m"] / 60, 2), len(x.get("segments", [])),
                        "" if a.get("snr_db") is None else round(a["snr_db"], 1), a.get("bandwidth_hz") or "",
                        r.get("mf_year") or r["probe"].get("meta_year") or r.get("file_year") or "",
                        bool(r.get("mf_text")), len(r.get("cues", [])),
                        "; ".join(dup_of.get(r["display_path"], [])), *af.lid_columns(r)])
    with open(out_dir / "segments_manifest.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            x = r.get("x") or {}
            segs = x.get("segments", [])
            one_text = r["mf_texts"][0] if len(r.get("mf_texts", [])) == 1 and len(segs) == 1 else None
            sh = lid_shares(r)
            file_lang = None if sh is None else {"en": "English", "mixed": "mixed or uncertain"}.get(
                sh[4], af.lang_name(sh[4]))
            for sg in segs:
                text = one_text or " ".join(t for a, b, t in r.get("cues", [])
                                            if a < sg["offset"] + sg["duration"] and b > sg["offset"])
                fh.write(json.dumps({"audio_filepath": r["display_path"], "drive_file_id": r["drive_id"],
                                     "offset": sg["offset"], "duration": sg["duration"],
                                     "speaker_class": CLASS_NAMES[sg["class"]],
                                     "median_f0": sg["median_f0"], "file_language": file_lang,
                                     "text": text or ""},
                                    ensure_ascii=False) + "\n")
    with open(out_dir / "near_duplicates.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "longer_recording_a", "recording_b", "overlap_s", "b_starts_at_s_in_a",
                    "matching_landmarks", "match_density"])
        for p in sorted(explore["pairs"], key=lambda p: -p["overlap_s"]):
            w.writerow([p["kind"], p["a_rec"]["display_path"], p["b_rec"]["display_path"],
                        p["overlap_s"], p["offset_s"], p["matches"], p["density"]])
    with open(out_dir / "top_words.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["word", "count"])
        for word, n in explore["words"].most_common(500):
            w.writerow([word, n])
    return ["recordings.csv", "segments_manifest.jsonl", "near_duplicates.csv", "top_words.csv"]


if __name__ == "__main__":
    sys.exit(main())
