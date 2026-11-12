#!/usr/bin/env python3
"""
ASR data feasibility check for Google Drive
===========================================

Runs the same analysis as asr_feasibility.py on recordings stored in a Google
Drive folder (My Drive or a shared drive), including all subfolders.

How it works
------------
1. Signs in to Google with read-only access (a browser window opens once).
2. Lists every file in the folder tree. This uses only Drive metadata, so the
   inventory, file sizes, video durations and duplicates are known without
   downloading anything. Exact duplicates (same content) are downloaded once.
3. Downloads small label files only: caption files, JSON/CSV/TSV manifests, and
   Google Sheets (exported as CSV). Google Docs are counted by name, not opened.
4. Downloads each recording to a temporary folder, analyzes it, and deletes it
   straight away. Only one recording per worker is on disk at any time.
5. Writes the same aggregate report as asr_feasibility.py.

Use --sample-files or --max-download-gb to limit how much is downloaded on a
slow or metered connection. Files not downloaded still count toward totals
(video durations come from Drive; audio-only durations are estimated from file
size), and speech figures are extrapolated from the analyzed sample.

Privacy
-------
* Read-only access. Nothing in Drive is changed, moved or shared.
* Downloads go to a temporary folder on this computer and are deleted after
  each file is analyzed; the folder itself is removed at the end.
* The report contains only totals and distributions, exactly as for
  asr_feasibility.py: no file names, folder names, transcripts or audio.
* The sign-in token is stored on this computer only, and deleted after a
  successful run unless --keep-login is given. Access can also be revoked at
  https://myaccount.google.com/permissions

Setup (once, by the engineer): see README.md, "Google Drive". In short, create
an OAuth client of type "Desktop app" in a Google Cloud project and save its
JSON file as client_secret.json next to this script.

Usage
-----
    python drive_feasibility.py                         # asks for the folder link
    python drive_feasibility.py "https://drive.google.com/drive/folders/XXXX"
    python drive_feasibility.py FOLDER_LINK --max-download-gb 20
    python drive_feasibility.py FOLDER_LINK --sample-files 300 --language-id
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))   # bundled code
# Folder the user sees: next to the .exe when frozen, else next to this script.
APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else \
    Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import asr_feasibility as af  # noqa: E402  (shared analysis and reporting)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FOLDER_MIME = "application/vnd.google-apps.folder"
SHORTCUT_MIME = "application/vnd.google-apps.shortcut"
GDOC_MIME = "application/vnd.google-apps.document"
GSHEET_MIME = "application/vnd.google-apps.spreadsheet"
LIST_FIELDS = ("nextPageToken, files(id, name, mimeType, size, md5Checksum, createdTime, "
               "modifiedTime, videoMediaMetadata(durationMillis), "
               "shortcutDetails(targetId, targetMimeType), capabilities(canDownload))")
FILE_FIELDS = LIST_FIELDS[len("nextPageToken, files("):-1]
CAPTION_MAX_BYTES = 50_000_000
MIME_EXT = {"audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/wav": ".wav", "audio/x-wav": ".wav",
            "audio/wave": ".wav", "audio/mp4": ".m4a", "audio/x-m4a": ".m4a", "audio/aac": ".aac",
            "audio/ogg": ".ogg", "audio/opus": ".opus", "audio/webm": ".webm", "audio/flac": ".flac",
            "audio/x-flac": ".flac", "audio/amr": ".amr", "audio/3gpp": ".3gp",
            "video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm",
            "video/3gpp": ".3gp", "video/x-msvideo": ".avi", "video/x-matroska": ".mkv",
            "video/mpeg": ".mpg", "video/x-ms-wmv": ".wmv"}
TOKEN_PATH = Path.home() / ".asr_feasibility" / "drive_token__LOCAL_ONLY.json"


# ---------------------------------------------------------------------------
# Google Drive access
# ---------------------------------------------------------------------------
class DriveClient:
    """Thin read-only wrapper over the Drive v3 API. One API client per thread,
    since the underlying HTTP library is not thread-safe."""

    def __init__(self, client_secret, token_path=TOKEN_PATH):
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow

        self.token_path = token_path
        creds = None
        if token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
            except Exception:
                creds = None
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None  # e.g. a testing-mode token older than 7 days
        if not creds or not creds.valid:
            if not Path(client_secret).exists():
                raise SystemExit(f"Sign-in file not found: {client_secret}\n"
                                 "See README.md, 'Google Drive', for how to create it.")
            print("A browser window will open. Sign in with the Google account that can see")
            print("the recordings, and allow read-only access to Google Drive.")
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secret), SCOPES)
            creds = flow.run_local_server(port=0, open_browser=True,
                                          authorization_prompt_message="If no browser opened, visit: {url}",
                                          success_message="Signed in. You can close this tab.")
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")
        try:
            os.chmod(token_path, 0o600)
        except OSError:
            pass
        self.creds = creds
        self._local = threading.local()

    @property
    def svc(self):
        if not hasattr(self._local, "svc"):
            from googleapiclient.discovery import build
            self._local.svc = build("drive", "v3", credentials=self.creds, cache_discovery=False)
        return self._local.svc

    def get(self, file_id):
        return self.svc.files().get(fileId=file_id, fields=FILE_FIELDS,
                                    supportsAllDrives=True).execute(num_retries=5)

    def list_children(self, folder_id):
        out, token = [], None
        while True:
            resp = self.svc.files().list(
                q=f"'{folder_id}' in parents and trashed = false", fields=LIST_FIELDS,
                pageSize=1000, pageToken=token, corpora="allDrives",
                supportsAllDrives=True, includeItemsFromAllDrives=True).execute(num_retries=5)
            out.extend(resp.get("files", []))
            token = resp.get("nextPageToken")
            if not token:
                return out

    def download(self, file_id, dest):
        from googleapiclient.http import MediaIoBaseDownload
        req = self.svc.files().get_media(fileId=file_id, supportsAllDrives=True)
        with open(dest, "wb") as fh:
            dl = MediaIoBaseDownload(fh, req, chunksize=8 * 1024 * 1024)
            done = False
            while not done:
                if af.STOP.is_set():
                    raise af._Stopped()
                _, done = dl.next_chunk(num_retries=5)

    def export_csv(self, file_id, dest):
        data = self.svc.files().export(fileId=file_id, mimeType="text/csv").execute(num_retries=5)
        Path(dest).write_bytes(data)


def make_client(args):
    """Factory, so tests can substitute a local stand-in for Google Drive."""
    return DriveClient(args.client_secret)


def parse_folder_id(text):
    text = (text or "").strip().strip('"')
    for pat in (r"/folders/([A-Za-z0-9_-]{10,})", r"[?&]id=([A-Za-z0-9_-]{10,})"):
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return text if re.fullmatch(r"[A-Za-z0-9_-]{10,}", text) else None


# ---------------------------------------------------------------------------
# Folder walk
# ---------------------------------------------------------------------------
def _safe(name):
    return re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name).strip() or "_"


def walk(drive, root_id):
    """All files under root_id (recursively, following shortcuts) with their
    relative paths. Returns (items, folder_count)."""
    items, used = [], set()
    stack, visited = [(root_id, "")], {root_id}
    while stack:
        if af.STOP.is_set():
            raise af._Stopped()
        fid, rel = stack.pop()
        for f in drive.list_children(fid):
            name = _safe(f.get("name", ""))
            if f.get("mimeType") == SHORTCUT_MIME:
                sd = f.get("shortcutDetails") or {}
                tid = sd.get("targetId")
                if not tid:
                    continue
                if sd.get("targetMimeType") == FOLDER_MIME:
                    if tid not in visited:
                        visited.add(tid)
                        stack.append((tid, f"{rel}{name}/"))
                    continue
                try:
                    f = drive.get(tid)
                except Exception:
                    continue
                name = _safe(f.get("name", name))
            if f.get("mimeType") == FOLDER_MIME:
                if f["id"] not in visited:
                    visited.add(f["id"])
                    stack.append((f["id"], f"{rel}{name}/"))
                continue
            path = f"{rel}{name}"
            if path.lower() in used:  # Drive allows identical names in one folder
                p = Path(path)
                path = str(p.with_name(f"{p.stem}~{f['id'][:6]}{p.suffix}")).replace(os.sep, "/")
            used.add(path.lower())
            items.append(dict(f, rel=path))
    return items, len(visited)


def media_ext(item):
    ext = Path(item["name"]).suffix.lower()
    if ext in af.AUDIO_EXT or ext in af.VIDEO_EXT:
        return ext
    if ext in af.TRANSCRIPT_EXT | af.MANIFEST_EXT | af.ARCHIVE_EXT:
        return None   # e.g. Drive may label .ass subtitle files as audio/aac
    mime = (item.get("mimeType") or "").lower()
    if mime.startswith(("audio/", "video/")):
        return MIME_EXT.get(mime, ".video" if mime.startswith("video/") else ".audio")
    return None


def drive_year(item):
    years = []
    for k in ("createdTime", "modifiedTime"):
        m = re.match(r"(\d{4})-", item.get(k) or "")
        if m:
            years.append(int(m.group(1)))
    y = min(years) if years else None
    return y if y and 1995 <= y <= time.localtime().tm_year + 1 else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="ASR feasibility check for a Google Drive folder.")
    ap.add_argument("folder", nargs="?", help="Drive folder link or ID (asks if omitted)")
    ap.add_argument("--client-secret", default=str(APP_DIR / "client_secret.json"),
                    help="OAuth client file (default: client_secret.json next to this program)")
    ap.add_argument("--output", help="Report folder (default: ./asr_feasibility_output)")
    ap.add_argument("--work-dir", help="Where to put temporary downloads (default: system temp)")
    ap.add_argument("--sample-files", type=int, default=None,
                    help="Download and analyze a random sample of this many recordings")
    ap.add_argument("--max-download-gb", type=float, default=None,
                    help="Stop selecting recordings once this much would be downloaded")
    ap.add_argument("--max-minutes-per-file", type=float, default=None,
                    help="Analyze at most this many minutes of each file (still downloads it)")
    ap.add_argument("--workers", type=int, default=2, help="Parallel downloads (default 2)")
    ap.add_argument("--vad", choices=["auto", "webrtc", "energy"], default="auto")
    ap.add_argument("--no-pitch", action="store_true")
    ap.add_argument("--language-id", action="store_true",
                    help="Estimate English vs. non-English speech hours (needs faster-whisper)")
    ap.add_argument("--lid-model", default="small")
    ap.add_argument("--lid-windows-per-file", type=int, default=4)
    ap.add_argument("--lid-max-windows", type=int, default=1500)
    ap.add_argument("--fresh", action="store_true", help="Ignore saved progress")
    ap.add_argument("--keep-checkpoint", action="store_true")
    ap.add_argument("--keep-login", action="store_true",
                    help="Keep the Google sign-in token after a successful run")
    ap.add_argument("--per-file", action="store_true",
                    help="Also write a LOCAL ONLY per-file table (not for sharing)")
    ap.add_argument("--check", action="store_true",
                    help="Check that everything needed is installed, then exit (no sign-in)")
    args = ap.parse_args()
    if args.check:
        return self_check(args)
    interactive = args.folder is None

    print(f"ASR data feasibility check for Google Drive v{af.TOOL_VERSION}\n")
    folder_text = args.folder
    if not folder_text:
        try:
            folder_text = input("Paste the link to the Google Drive folder: ")
        except EOFError:
            folder_text = ""
    root_id = parse_folder_id(folder_text)
    if not root_id:
        print("That doesn't look like a Google Drive folder link.")
        return af.finish(interactive, 1)

    ffmpeg = af.find_ffmpeg()
    if not ffmpeg:
        print("ffmpeg was not found. Install it, or run: pip install imageio-ffmpeg")
        return af.finish(interactive, 1)
    ffprobe = af.find_ffprobe(ffmpeg)
    vad_method = args.vad
    if vad_method in ("auto", "webrtc"):
        try:
            import webrtcvad  # noqa: F401
            vad_method = "webrtc"
        except ImportError:
            vad_method = "energy"

    out_dir = Path(args.output or (Path.cwd() / "asr_feasibility_output")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="asr_drive_", dir=args.work_dir)).resolve()
    shadow = work / "tree"      # mirrors the Drive folder structure for label matching
    dl_dir = work / "downloads"
    shadow.mkdir()
    dl_dir.mkdir()
    t0 = time.time()

    try:
        return run(args, interactive, root_id, ffmpeg, ffprobe, vad_method, out_dir,
                   work, shadow, dl_dir, t0)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def self_check(args):
    """Offline check of the installation: no sign-in and no network access."""
    ok = True
    def line(good, what, detail=""):
        print(f"  [{'OK' if good else '!!'}] {what}{(': ' + detail) if detail else ''}")
    print(f"drive_feasibility v{af.TOOL_VERSION} installation check")
    ff = af.find_ffmpeg()
    line(bool(ff), "ffmpeg", "bundled/installed" if ff else "missing")
    ok &= bool(ff)
    try:
        import webrtcvad  # noqa: F401
        line(True, "webrtcvad speech detector")
    except ImportError:
        line(False, "webrtcvad speech detector", "missing (energy-based detection will be used)")
    try:
        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials
        import google_auth_oauthlib.flow  # noqa: F401
        svc = build("drive", "v3", credentials=Credentials(token="offline-check"),
                    cache_discovery=False, static_discovery=True)
        line(hasattr(svc, "files"), "Google Drive API client (offline)")
    except Exception as e:
        line(False, "Google Drive API client", type(e).__name__)
        ok = False
    cs = Path(args.client_secret)
    if cs.exists():
        try:
            kind = next(iter(json.loads(cs.read_text(encoding="utf-8"))))
            line(kind == "installed", "client_secret.json",
                 "Desktop app client" if kind == "installed" else
                 f"'{kind}' client; create a Desktop app client instead")
            ok &= kind == "installed"
        except Exception:
            line(False, "client_secret.json", "unreadable")
            ok = False
    else:
        line(False, "client_secret.json", f"not found at {cs} (needed before signing in)")
    if not ok:
        print("Some items need attention (see above).")
    elif not cs.exists():
        print("Ready, once client_secret.json is placed next to this program.")
    else:
        print("Ready.")
    return 0 if ok else 1


def run(args, interactive, root_id, ffmpeg, ffprobe, vad_method, out_dir, work, shadow, dl_dir, t0):
    try:
        drive = make_client(args)
    except SystemExit as e:          # e.g. client_secret.json missing
        print(e)
        return af.finish(interactive, 1)
    except KeyboardInterrupt:
        return af.finish(interactive, 130)
    except Exception as e:           # sign-in cancelled, browser closed, no network
        print(f"Sign-in did not complete ({type(e).__name__}). Check the internet connection")
        print("and try again; on the Google page, choose the account that can see the folder.")
        return af.finish(interactive, 1)
    print("Listing files in the Drive folder...")
    try:
        items, n_folders = walk(drive, root_id)
    except af._Stopped:
        return af.finish(interactive, 130)
    except Exception as e:
        print(f"Could not list the folder ({type(e).__name__}). Check the link, and that the")
        print("signed-in account can open it.")
        return af.finish(interactive, 1)
    print(f"  {len(items)} files in {n_folders} folders.")

    # --- sort into recordings and label files
    media_items, caption_items, doc_paths, manifest_items = [], [], [], []
    other_ext = Counter()
    gdocs = gsheets = 0
    for it in items:
        mime = it.get("mimeType", "")
        if mime == GDOC_MIME:                     # counted by name only; never opened
            gdocs += 1
            other_ext[".gdoc"] += 1
            doc_paths.append(shadow / (it["rel"] + ".gdoc"))
            continue
        if mime == GSHEET_MIME:                   # exported as CSV and checked as a manifest
            gsheets += 1
            other_ext[".gsheet"] += 1
            manifest_items.append((it, shadow / (it["rel"] + ".csv"), "sheet"))
            continue
        ext = media_ext(it)
        if ext:
            media_items.append((it, ext))
            continue
        fext = Path(it["name"]).suffix.lower() or "(no extension)"
        other_ext[fext] += 1
        if fext in af.CAPTION_EXT:
            caption_items.append((it, shadow / it["rel"]))
        elif fext in af.DOC_EXT:
            doc_paths.append(shadow / it["rel"])
        elif fext in af.MANIFEST_EXT:
            manifest_items.append((it, shadow / it["rel"], "file"))

    # --- download small label files (captions, manifests, Sheets as CSV)
    def fetch(it, p, kind, limit):
        if int(it.get("size") or 0) > limit or not (it.get("capabilities") or {}).get("canDownload", True):
            return False
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            if kind == "sheet":
                drive.export_csv(it["id"], p)
            else:
                drive.download(it["id"], p)
            return True
        except af._Stopped:
            raise
        except Exception:
            return False

    labels, manifests, label_dl_failed = list(doc_paths), [], 0
    if caption_items or manifest_items:
        print(f"Downloading {len(caption_items) + len(manifest_items)} caption/manifest files (small)...")
    try:
        for it, p in caption_items:
            if fetch(it, p, "file", CAPTION_MAX_BYTES):
                labels.append(p)
            else:
                label_dl_failed += 1
        for it, p, kind in manifest_items:
            if fetch(it, p, kind, af.MANIFEST_MAX_BYTES):
                manifests.append(p)
            else:
                label_dl_failed += 1
    except af._Stopped:
        return af.finish(interactive, 130)

    # --- recording records (deduplicated by Drive's content checksum)
    records, seen, aliases, dup_count = [], {}, [], 0
    for it, ext in sorted(media_items, key=lambda t: t[0]["rel"]):
        md5 = it.get("md5Checksum")
        path = shadow / it["rel"]
        if md5 and md5 in seen:
            dup_count += 1
            aliases.append((path, seen[md5]))
            continue
        mime = (it.get("mimeType") or "").lower()
        dur_ms = (it.get("videoMediaMetadata") or {}).get("durationMillis")
        rec = {"path": path, "display_path": it["rel"], "ext": ext, "size": int(it.get("size") or 0),
               "key": hashlib.sha1(f"{it['id']}|{md5}|{it.get('modifiedTime')}".encode()).hexdigest()[:24],
               "file_year": drive_year(it),
               "kind": "video" if ext in af.VIDEO_EXT and not mime.startswith("audio/") else "audio",
               "drive_id": it["id"], "drive_duration_s": int(dur_ms) / 1000 if dur_ms else None,
               "can_download": (it.get("capabilities") or {}).get("canDownload", True)}
        records.append(rec)
        if md5:
            seen[md5] = rec
    if dup_count:
        print(f"  {dup_count} duplicate copies will not be downloaded.")
    if not records:
        print("No audio or video files found in that folder.")
        return af.finish(interactive, 1)

    # --- choose what to download
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
    total_gb = sum(r["size"] for r in records) / 1e9
    sel_gb = sum(r["size"] for r in order) / 1e9
    print(f"Recordings: {len(records)} ({total_gb:.1f} GB). Selected for download and "
          f"analysis: {len(order)} ({sel_gb:.1f} GB).")

    # --- checkpoint
    max_s = args.max_minutes_per_file * 60 if args.max_minutes_per_file else None
    lid_sig = af.lid_signature(args.lid_model, args.lid_windows_per_file) if args.language_id else None
    ckpt = af.Checkpoint(out_dir, {
        "tool_version": af.TOOL_VERSION, "source": "google-drive",
        "folder_id": hashlib.sha1(root_id.encode()).hexdigest()[:16],
        "max_minutes_per_file": args.max_minutes_per_file, "speech_detector": vad_method,
        "pitch_analysis": not args.no_pitch, "format_reader": "ffprobe" if ffprobe else "ffmpeg",
        "language_id": lid_sig}, fresh=args.fresh)
    if ckpt.reason:
        print(f"  Note: {ckpt.reason}.")

    def interrupted():
        ckpt.close()
        print("\n\nStopped. Progress has been saved. Run the tool again with the same folder")
        print("and settings and it will continue where it left off.")
        return af.finish(interactive, 130)

    # --- language model and window budget (files are processed in random order,
    #     so stopping when the budget runs out gives a random sample)
    model, lid_status = None, {"status": "not run (enable with --language-id)"}
    lock, budget = threading.Lock(), {"used": 0}
    if args.language_id:
        model, err = af.load_lid_model(args.lid_model)
        lid_status = {"status": err} if model is None else af.lid_meta(
            args.lid_model, args.lid_windows_per_file, args.lid_max_windows, 0)

    pending, done = [], 0
    for r in order:
        a = ckpt.analyses.get(r["key"])
        if a is not None:
            r["probe"] = ckpt.probes.get(r["key"]) or af._empty_probe()
            r["analysis"] = a
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
                        "analysis": {"status": "download_error", "error_detail": type(e).__name__}}
            try:
                probe = af.probe_ffprobe(tmp, ffprobe) if ffprobe else af.probe_ffmpeg_stderr(tmp, ffmpeg)
            except Exception:
                probe = af._empty_probe()
            analysis = None
            if probe["has_audio"]:
                analysis = af.analyze_file(tmp, ffmpeg, vad_method, max_s, not args.no_pitch)
            lid = None
            if model is not None and analysis and analysis.get("status") == "ok":
                n = len(af.lid_windows(analysis, args.lid_windows_per_file))
                take = False
                with lock:
                    if n and budget["used"] + n <= args.lid_max_windows:
                        budget["used"] += n
                        take = True
                lid = (af.lid_for_file(tmp, ffmpeg, analysis, model, args.lid_windows_per_file)
                       if take else {"status": "not_selected", "windows": []})
            return {"probe": probe, "analysis": analysis or {"status": "no_audio"}, "lid": lid}
        except af._Stopped:
            return None
        except Exception as e:
            return {"probe": dict(af._empty_probe(), ok=True, has_audio=True),
                    "analysis": {"status": "decode_error", "error_detail": type(e).__name__}}
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

    if pending:
        print(f"Downloading and analyzing {len(pending)} recordings "
              f"(each is deleted right after analysis)...")
    hours, dl_bytes = 0.0, 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        try:
            futs = {ex.submit(process, r): r for r in pending}
            for fut in as_completed(futs):
                r, res = futs[fut], fut.result()
                if res is None:
                    continue
                r["probe"], r["analysis"] = res["probe"], res["analysis"]
                if r["analysis"].get("status") != "download_error":   # retried next run
                    ckpt.save_probe(r["key"], r["probe"])
                    ckpt.save_analysis(r["key"], r["analysis"])
                if lid_sig and res.get("lid") is not None:
                    r["lid"] = res["lid"]
                    ckpt.save_lid(f"{r['key']}|{lid_sig}", res["lid"])
                done += 1
                dl_bytes += r["size"]
                hours += r["analysis"].get("analyzed_s", 0) / 3600
                print(f"  {done}/{len(order)} recordings, {dl_bytes / 1e9:.1f} GB downloaded, "
                      f"{hours:.1f} h analyzed, {(time.time() - t0) / 60:.1f} min",
                      end="\r", flush=True)
        except KeyboardInterrupt:
            af.STOP.set()
            ex.shutdown(wait=True, cancel_futures=True)
            return interrupted()
    print()

    # --- recordings not downloaded: duration from Drive (video) or estimated from size
    # bytes per second from analyzed files: same folder and type first (usually the
    # same recorder and settings), then same type anywhere, then everything
    bps_dir, bps = {}, {}
    for r in records:
        a = r.get("analysis") or {}
        d = (r.get("probe") or {}).get("duration_s")
        if d and r["size"] and a.get("status") == "ok":
            bps_dir.setdefault((str(r["path"].parent), r["ext"]), []).append(r["size"] / d)
            bps.setdefault(r["ext"], []).append(r["size"] / d)
    all_bps = [v for vs in bps.values() for v in vs]
    estimated = 0
    for r in records:
        if r.get("probe") and r.get("analysis") is not None:
            if r["analysis"].get("status") == "no_audio":
                r["analysis"] = None   # treated like a file with no audio track
            continue
        dur = r["drive_duration_s"]
        if dur is None:
            ref = bps_dir.get((str(r["path"].parent), r["ext"])) or bps.get(r["ext"]) or all_bps
            if ref and r["size"]:
                dur = r["size"] / sorted(ref)[len(ref) // 2]
                estimated += 1
        r["probe"] = dict(af._empty_probe(), ok=True, has_audio=True, duration_s=dur)
        r["analysis"] = None

    # --- labels and manifests
    index = {(str(r["path"].parent), r["path"].stem.lower()): r for r in records}
    for p, rec in aliases:
        index.setdefault((str(p.parent), p.stem.lower()), rec)
    label_stats = af.match_labels(labels, index) if labels else {}
    manifest_stats = {}
    if manifests:
        manifest_stats = af.match_manifests(manifests, af.MediaIndex(shadow, records, aliases))

    if model is not None:
        lid_status["files_selected"] = sum(1 for r in records if (r.get("lid") or {}).get("status") == "ok")
    settings = {
        "source": "Google Drive",
        "max_minutes_per_file": args.max_minutes_per_file,
        "sample_files": args.sample_files or (len(order) if len(order) < len(records) else None),
        "max_download_gb": args.max_download_gb,
        "speech_detector": vad_method,
        "pitch_analysis": not args.no_pitch,
        "format_reader": "ffprobe" if ffprobe else "ffmpeg",
        "language_id": args.language_id,
    }
    report = af.build_report(len(media_items), other_ext, records, dup_count, settings,
                             lid_status, label_stats, manifest_stats)
    drive_info = {
        "folders_scanned": n_folders,
        "files_listed": len(items),
        "recordings_selected_for_download": len(order),
        "recordings_not_downloadable": sum(1 for r in records if not r["can_download"]),
        "gb_downloaded_this_run": round(dl_bytes / 1e9, 2),
        "durations_from_drive_metadata": sum(1 for r in records if r["drive_duration_s"]
                                             and not (r.get("analysis") or {}).get("status") == "ok"),
        "durations_estimated_from_file_size": estimated,
        "google_docs_counted_by_name": gdocs,
        "google_sheets_checked_as_manifests": gsheets,
        "label_files_not_downloaded": label_dl_failed,
    }
    report["google_drive"] = drive_info
    text = af.render_text(report)
    extra = ["", "GOOGLE DRIVE SCAN"]
    extra += [f"  {k.replace('_', ' ').capitalize()}: {v}" for k, v in drive_info.items()]
    extra += ["  Recordings that were not downloaded count toward totals using their Drive",
              "  video duration, or a duration estimated from file size for audio-only",
              "  files. They are assumed to contain audio. Speech, quality and language",
              "  figures come from the downloaded sample and are extrapolated."]
    text += "\n".join(extra) + "\n"

    json_path = out_dir / "feasibility_report.json"
    txt_path = out_dir / "feasibility_report.txt"
    af.write_atomic(json_path, json.dumps(report, indent=2))
    af.write_atomic(txt_path, text)
    ckpt.close(delete=not args.keep_checkpoint)
    if args.per_file:
        local = af.write_per_file_csv(out_dir, records, shadow)
        print(f"  Local-only per-file table: {local}")
    if not args.keep_login:
        try:
            TOKEN_PATH.unlink()
        except OSError:
            pass

    print(f"\nDone in {(time.time() - t0) / 60:.1f} minutes.")
    print(f"Report written to:\n  {txt_path}\n  {json_path}")
    print("\nPlease open the .txt report and read it before sharing the two report files.")
    return af.finish(interactive, 0)


if __name__ == "__main__":
    sys.exit(main())
