# asr_feasibility


# ASR data feasibility check

ASR Feasbility is a tool that looks through a folder of audio and video recordings and writes a short report describing the collection as a whole: how much audio there is, how much of it is speech, recording quality, formats, and rough indicators of child versus adult voices. It helps decide whether the recordings could be useful for training a speech recognition model, without anyone outside the organization hearing or seeing them.


Four programs that share the same analysis:

- **`asr_feasibility.py`** scans a folder on this computer (or an external drive, or a Google Drive folder synced with Google Drive for desktop).
- **`drive_feasibility.py`** scans a Google Drive folder directly. It downloads one recording at a time, analyzes it, and deletes it, so nothing accumulates on the computer, and it can analyze a sample when bandwidth is limited. Its report is the same as the local tool's.
- **`drive_explore.py`** goes further, for when you're permitted to see the data: turn-level child/adult segmentation, Whisper-ready segments, transcript language and code-switching, near-duplicate detection and a folder map. Its outputs contain personal data. For the pre-existing executables, is only used in conjunction with drive_explore_lid.py.
- **`drive_explore_lid.py`** is an explorer for full, unrestricted access, with Whisper language checks always on. It's built separately, with the language model shipped alongside it.



---

## Part 1: For the organization running the tool

### What the tool does

- Reads the audio and video files in a folder you choose, including subfolders.
- Measures things like length, loudness, background noise and voice pitch.
- Writes two report files to a new folder called `asr_feasibility_output`.

### What it does not do

- It does not upload, send or copy any recordings anywhere. It works without an internet connection, if using asr_feasibility only (the only local option).
- It does not change, move or delete any of your files.
- The report contains **no file names, folder names, transcripts, audio, or details about any single recording**, only totals and percentages across the whole collection.
- It does not identify people, count speakers, or estimate ages. The pitch figures are broad ranges across all recordings combined.
- If there are subtitle or caption files, it reads only their timings (when each caption starts and ends), never the caption text.
- If there are dataset files that list transcriptions (JSON, CSV or TSV "manifests"), it checks which recordings have a transcription and how long it is. It never copies the text, speaker IDs or place names into the report.
- It does not look inside .zip, .tar or similar archives. If recordings are stored that way, extract them first; the report will remind you if it finds archives.

### How to run it

**If you were given `asr_feasibility.exe`:** double-click it, choose the folder that holds the recordings, and wait. A window shows progress. When it says "Done", press Enter to close it.

**If you were given the Python script:** see Part 2, or ask the person who sent it for help.

Expect it to take very roughly 10 to 30 seconds per hour of recordings on a typical computer. Large collections can take a few hours; it is fine to leave it running.

### If the recordings are in Google Drive

Use `drive_feasibility.exe` (or `drive_feasibility.py`) instead. Double-click it, paste the link to the Drive folder (open the folder in a browser and copy the address), and sign in when a browser window opens. It needs read-only access; it can't change, move or share anything. Each recording is downloaded, analyzed, and deleted straight away.

On a slow or expensive connection, ask the person who sent the tool which limit to use, for example "only download 20 GB" or "only analyze 300 recordings". The report then estimates the rest from that sample.

When the run finishes, the sign-in is removed from the computer. You can also check or remove the tool's access at https://myaccount.google.com/permissions.

### If it gets interrupted

Progress is saved after every file. If the power goes off, the computer shuts down, or the window is closed, just run the tool again and choose the same folder. It will say "Resuming from saved progress" and continue where it stopped, without redoing finished files.

While a run is unfinished, the output folder contains a file called `resume_checkpoint__LOCAL_ONLY.jsonl`. It holds only coded file IDs and measurements, no names or recordings, and it is deleted automatically when the run finishes. You don't need to send it.

### What to send back

Open `feasibility_report.txt` in the `asr_feasibility_output` folder and read it through. If you are comfortable with its contents, send that file and `feasibility_report.json` (the same information in a machine-readable form).

If a file called `LOCAL_ONLY_per_file_details__DO_NOT_SHARE.csv` appears, **keep it to yourself**. It lists individual file names and is only there to help you find problem files.


## Part 2: For the engineer

### Install

```bash
pip install -r requirements.txt
python asr_feasibility.py "/path/to/recordings"
```

ffmpeg is required. The tool uses a system ffmpeg/ffprobe if present, otherwise the static binary bundled with `imageio-ffmpeg`. Without ffprobe it parses ffmpeg's own output for format information, which works but is less complete.

It is recommended to use .exe for Windows systems, as ffmpeg is not fully supported on this system.

### Options

| Option | Purpose |
|---|---|
| `--max-minutes-per-file N` | Analyze only the first N minutes of each file. Speech hours are extrapolated. |
| `--sample-files N` | Acoustically analyze a random sample of N files (formats are still read for all). |
| `--workers N` | Parallel decoding (default: up to 4). |
| `--vad {auto,webrtc,energy}` | Speech detector. `auto` uses webrtcvad if installed. |
| `--no-pitch` | Skip pitch estimation. |
| `--language-id` | Estimate speech hours in English, in other languages, and mixed or uncertain (see "Languages" below). No transcripts are produced or stored. Needs `pip install faster-whisper` and a one-time model download. |
| `--lid-model NAME` | Whisper model for language checks (default `small`; `base` is about twice as fast and somewhat less accurate). |
| `--lid-windows-per-file N` | 30 s windows checked per file, spread across its speech (default 4). |
| `--lid-max-windows N` | Overall window budget (default 1500). If the collection needs more, a random sample of files is checked and the result extrapolated. |
| `--fresh` | Ignore saved progress from an interrupted run and start over. |
| `--keep-checkpoint` | Keep the progress file after a successful run (by default it is deleted). |
| `--per-file` | Also write the local-only per-file CSV. |
| `--output DIR` | Report location. |

### Building a double-click executable (Windows)

PyInstaller builds for the operating system it runs on, so a Windows `.exe` has to be built on Windows. There are two ways.

**On a Windows computer.** Install Python 3.12 (64-bit) from python.org, put `build_windows.bat` in the same folder as `asr_feasibility.py`, and double-click it. It creates a clean build environment, installs only what the tool needs, and produces `dist\asr_feasibility.exe`. The executable bundles Python, numpy, the webrtcvad speech detector and ffmpeg, so the computer it runs on needs nothing else installed. Expect it to be roughly 50 to 100 MB.

**Without a Windows computer.** Use GitHub Actions; a private repository is fine, since it contains no data. Two workflows are provided:

- `.github/workflows/build-windows-all.yml` (**Build Windows executables**) builds `asr_feasibility.exe`, `drive_feasibility.exe` and `drive_explore.exe` in parallel, tests each one on GitHub's Windows machines with no system ffmpeg available, and offers each as a downloadable artifact. A version tag also publishes all three on one release. To add a program, add an entry under `include`.
- `.github/workflows/build-drive-explore-lid.yml` (**Build drive_explore_lid**) builds only the backup explorer and its language model; see that section.

Open the repository's **Actions** tab, choose the workflow and click **Run workflow**. Artifacts are listed at the bottom of the finished run's page; they expire after 90 days by default, so use releases for anything you'll share.

Github Actions was utilized to create all Windows executables. 

The build options, used by both:

```bat
pyinstaller --noconfirm --clean --onefile --console --collect-all imageio_ffmpeg ^
  --exclude-module faster_whisper --exclude-module ctranslate2 --exclude-module torch ^
  --exclude-module googleapiclient --exclude-module google_auth_oauthlib ^
  --name asr_feasibility asr_feasibility.py
```

- `--console` keeps the window that shows progress and asks for Enter at the end; don't use `--noconsole`.
- `--collect-all imageio_ffmpeg` bundles ffmpeg.
- The exclusions keep the optional language-check and Google libraries out, so the executable stays small. Language checks then report that faster-whisper isn't installed; run the Python script for those. Including them would add the Whisper engine to the executable, and the model (about half a gigabyte) would still be downloaded on first use.

**Testing** On a Windows computer without Python, double-click the executable, choose a folder of test recordings, and check the report. Also close the window partway through a run and start it again, to confirm it resumes.

**Organizational View**

- **SmartScreen:** an unsigned executable triggers "Windows protected your PC". Clicking **More info** then **Run anyway** proceeds. Signing it with a code-signing certificate avoids the warning.
- **Antivirus:** some antivirus products flag one-file PyInstaller executables by mistake. If that happens, build without `--onefile` (the result is a folder; zip it and ship it, and the executable is inside) or submit the file to the antivirus vendor as a false positive.
- **Extract first:** if the tool is sent as a zip, it has to be extracted before running. Run from inside the zip, it would write its report to a temporary folder.
- **Where the report goes:** the `asr_feasibility_output` folder is created next to wherever the executable is run from, so put the executable somewhere writable like the Desktop or Documents.
- **Startup:** a one-file executable unpacks itself each time it starts, so it takes a few seconds before anything appears.

For macOS, build on a Mac of the same type the organization uses (Apple Silicon or Intel) with the same PyInstaller options, minus the `^` line continuations.

### How the measurements work

- **Decoding:** each file's first audio track is converted to 16 kHz mono (Whisper's input format) and processed in 30 s chunks, so memory use stays flat regardless of file length.
- **Duplicates:** detected by file size plus a hash of the first and last megabyte, then skipped.
- **Speech detection:** webrtcvad (aggressiveness 2) combined with a requirement that frames sit at least 3 dB above the file's 10th-percentile level, because webrtcvad alone tends to label steady noise as speech. The energy fallback uses an adaptive threshold above the noise floor and treats files with under 4 dB of level variation as non-speech.
- **SNR:** median power of speech frames divided by median power of non-speech frames. Not measurable when a file is almost all speech or almost all non-speech.
- **Pitch:** a vectorized YIN estimator on every second 30 ms frame of detected speech (70 to 700 Hz), with an adaptive threshold to reduce octave-down errors in noise. Reported as a collection-wide histogram and as files grouped by median pitch.
- **Child/adult mix:** for each file, the share of voiced speech frames at 250 Hz or above. Files are grouped as mostly adult-range (under 30%), mixed (30 to 70%) or mostly child-range (70% or more). The child-range speech estimate is frame-weighted: each file contributes its speech hours times its child-range share, so the child speech in a teacher-led lesson is counted rather than lost to a median.
- **Speech segments:** continuous speech runs, bridging pauses under 300 ms and ignoring blips under 200 ms, grouped by length. Runs over 30 s can't become Whisper training segments without forced alignment or splitting at shorter pauses. Also reports pauses per minute of speech.
- **Bandwidth:** the long-term average spectrum of speech frames in 250 Hz bands; the effective bandwidth is the highest band within 50 dB of the 250 to 3000 Hz level. This catches phone-call and 8 kHz voice-note audio that has been stored at a higher sample rate. Narrowband is under 5 kHz, reduced is 5 to 6.5 kHz.
- **Existing labels:** caption files (`.srt`, `.vtt`, `.ass`, `.ssa`, `.sbv`, `.ttml`, `.dfxp`) are paired with recordings of the same name in the same folder (`lesson1.mp4` with `lesson1.srt`, `lesson1.en.srt` or `lesson1.mp4.srt`), and the union of cue time ranges gives the hours covered. Only timestamps are parsed; the text is discarded. Where several caption files match one recording (for example two languages), the largest coverage is used. Captions next to a duplicate copy count toward the original. Document files (`.txt`, `.docx`, `.eaf`, `.TextGrid`, `.cha` and similar) with a matching name are counted but never opened. Media files with embedded subtitle tracks are counted too.
- **Dataset manifests:** every `.json`, `.jsonl`, `.csv` and `.tsv` file is examined, and treated as a manifest if its entries link to recordings in the folder. This covers AfriVoice-style JSON arrays, NeMo-style JSON Lines (including JSON Lines saved as `.json`), objects wrapping a list (`{"data": [...]}`), objects keyed by utterance ID (`{"utt_001": {...}}`), Hugging Face-style audio objects (`{"audio": {"path": ...}}`) and CSV/TSV tables. Field names are matched case-insensitively from lists of common names:
  - *Audio reference:* `audio_filepath`, `audio_path`, `path`, `audio`, `file`, `file_name` and similar, or any field whose value ends in a media extension, or failing that an ID field (`audio_id`, `utt_id`, `id`, `key`, or the object key) matched against file names without extensions.
  - *Linking:* the reference is resolved relative to the manifest's folder, the scan root, and a `dir` field if present (AfriVoice uses one), then by file name, then by name without extension. A name that matches several recordings is resolved in favour of one under the manifest's own folder; if that's still ambiguous, the entry is counted as ambiguous and skipped. Windows backslashes and `file://` prefixes are handled. Entries pointing at duplicate copies count toward the original.
  - *Text:* transcription fields (`transcription`, `transcript`, `text`, `raw_text`, `normalized_text` and similar) are kept separate from prompt fields (`script`, `prompt`, `sentence`), because a prompt is what a speaker was asked to read, not necessarily what a child said. AfriVoice's scripted entries have `transcription: null` and a `script`, so they're reported as prompt-only.
  - *What's measured:* whether each recording has a transcription or prompt, and the transcription's length in characters, compared against the recording's detected speech time. Fewer than 4 characters per second of speech suggests partial or condensed text, over 25 suggests the text doesn't match the audio. The text itself is discarded immediately.
  - *Metadata:* distinct speakers (from `speaker_id`, `voice_creator_id`, `client_id` and similar), how concentrated the hours are among the top speakers, age groups (numeric ages are bucketed into 0-5, 6-9, 10-12, 13-17 and 18 and over), gender, and the number of distinct locations. Speaker IDs and locations are hashed in memory and only counted; age and gender groups with fewer than 5 recordings are combined. This metadata is the right source for speaker diversity, rather than voice analysis.
  - Manifests over 1 GB are skipped and noted. Manifest matching runs after the audio analysis and is fast (about 2 seconds for 100,000 entries), so it isn't checkpointed.
- **Recording dates:** for each recording, the year is taken from the first available of these sources:
  1. *Manifest metadata,* if a linked manifest entry has a recording-date field. Fields are checked most-specific first (`recording_date`, `recorded_at`, `recorded_on`, `date_recorded`, `recording_year`, `session_date`, then `year`, `date`, `datetime`, `timestamp`, `created_at` and similar). Values can be a year (`2023`), an ISO date or datetime (`2023-05-01`, `2023-05-01T10:22:00Z`), a slash date (`2023/05/01`, `15/06/2021`), a compact date (`20230501`), text containing a year (`June 2020`), or a Unix timestamp in seconds or milliseconds. Transcription and upload dates aren't looked for.
  2. *The recording's embedded creation date* from its container metadata.
  3. *The file's dates,* the earlier of its modification and creation times.

  At every step, only years from 1995 to next year are accepted, so unset camera or phone clocks (1970, 1904 and similar) fall through to the next source. The report shows how many recordings got their year from each source, and how many have a manifest year that differs from their embedded date (the manifest year is used). Reported by year only.
- **Source type:** audio hours, speech hours and child-range speech hours split between audio inside video files and audio-only files.
- **Language-ID caveat:** when `--language-id` runs, the report states that Whisper cannot recognize most Kenyan languages other than Swahili and Somali and that Sheng and code-switched speech appear as mixtures.

The tool deliberately does not count distinct speakers or estimate ages: both require voiceprint-style biometric processing of children's voices, which existing consent doesn't cover.

### Languages (`--language-id`)

For each selected file, up to four 30-second windows are chosen evenly across the parts of the file that contain speech, and Whisper's language detector scores each one. Only the language probabilities are read; the model is never asked to transcribe. Each window is labelled:

- **English** if English is the most likely language at 60% or more,
- **non-English** if English scores under 25% (attributed to the most likely language), or
- **mixed or uncertain** otherwise.

The file's detected speech hours are divided between its windows' labels, and child-range speech hours are split the same way. The report gives speech hours for English, non-English and mixed/uncertain, both for the assessed files and extrapolated to all speech, plus a breakdown of non-English hours by detected language, and the summary shows estimated non-English hours and non-English child-range hours.

How far to trust it:

- **The English vs. non-English split is the reliable part.** Whisper can't recognize most Kenyan languages other than Swahili and Somali, but speech in an unrecognized language is usually labelled as some other non-English language, so it still counts as non-English. The specific language labels for those hours are unreliable.
- **Code-switching inside a window isn't resolved.** Swahili-English mixing and Sheng tend to land in "mixed or uncertain" or be labelled Swahili. A large mixed/uncertain share is itself a signal of code-switched speech.
- **Children's Kenyan-accented English** can occasionally be labelled Swahili, which would overstate non-English hours. The `--per-file` table includes each file's English and non-English shares, so Kenyan staff can listen to a few files labelled mostly non-English and confirm.
- **Per-recording results are sensitive.** A child's home language can reveal ethnic background, which Kenya's Data Protection Act treats as sensitive personal data. The shared reports only contain collection-wide totals, but the `--per-file` table and the explorer's `recordings.csv` and `segments_manifest.jsonl` link language results to individual recordings. Keep them on the organization's machine, and include language checks in the DPIA.
- **Speed:** each window takes roughly 1 to 3 seconds with the `small` model on a typical laptop CPU, so the default budget of 1,500 windows takes very roughly 30 to 75 minutes. A GPU is used automatically if available. Language results are checkpointed per file, so an interrupted run resumes.

### Google Drive (`drive_feasibility.py`)

**How it works.** It lists the folder tree through the Drive API (following folder shortcuts, and supporting shared drives), so the inventory, file sizes, Drive's video durations and exact duplicates (by Drive's MD5 checksum) are known without downloading anything. It then downloads only the small label files (captions, JSON/CSV/TSV manifests, and Google Sheets exported as CSV), and counts Google Docs by name as possible transcripts without opening them. Finally it downloads each selected recording to a temporary folder, runs exactly the same analysis as `asr_feasibility.py` (including language checks, ONLY when specifically enabled), and deletes the file before moving on. The temporary folder is removed at the end of the run, or when it's stopped with Ctrl+C. The report is the same as the local tool's, plus a "Google Drive scan" section.

**Limiting downloads.** `--sample-files N` analyzes a random sample of N recordings; `--max-download-gb G` stops selecting recordings once G gigabytes would be downloaded; both can be combined. Recordings that aren't downloaded still count toward totals: videos use Drive's duration, and audio-only files get a duration estimated from file size using the bytes per second of analyzed files in the same folder and format (falling back to the same format anywhere). They're assumed to contain audio. Speech, quality and language figures are extrapolated from the analyzed sample.

**Options** are the same as for `asr_feasibility.py`, plus `--client-secret PATH` (default: `client_secret.json` next to the script), `--work-dir PATH` (where temporary downloads go; needs space for one file per worker), `--keep-login` (keep the sign-in token after a successful run), and `--workers` defaults to 2, since the work is mostly downloading. Progress is checkpointed per recording and failed downloads are retried on the next run.

**One-time setup (engineer).** The tool needs an OAuth client so the organization's staff can sign in. Ideally create it in a Google Cloud project owned by the organization's Google account, so they stay in control of it.

1. In the Google Cloud console, create a project, then enable the **Google Drive API** for it.
2. Configure the consent screen (Google Auth Platform): give an app name and support email. If the organization uses Google Workspace and the project belongs to the same organization, choose **Internal**; otherwise choose **External**, leave it in **Testing**, and add the Google account that holds the recordings under **Test users**.
3. Add the scope `https://www.googleapis.com/auth/drive.readonly`.
4. Create an OAuth client of type **Desktop app**, download its JSON file, and save it as `client_secret.json` next to `drive_feasibility.py` (or the executable).

Things to expect: an "unverified app" warning on the sign-in screen, which is normal for a testing-mode app; sign-ins for testing-mode apps expire after 7 days, so a run started a week later asks to sign in again; and a Google Workspace administrator may need to allow the app. The token is saved to `~/.asr_feasibility/drive_token__LOCAL_ONLY.json` and deleted after a successful run unless `--keep-login` is used.

**Install (Python):** `pip install -r requirements.txt` with the Google lines uncommented. Run `python drive_feasibility.py --check` to confirm everything is in place; it checks ffmpeg, the speech detector, the Google libraries and `client_secret.json` without signing in or using the network.

**Building the Windows executable with GitHub.** The `build-windows-all.yml` workflow builds `drive_feasibility.exe` along with the other standard programs (the Drive tool imports the main script, so both need to be in the repository) and runs `--check` on it with no system ffmpeg available. The build options are:

```bat
pyinstaller --noconfirm --clean --onefile --console --collect-all imageio_ffmpeg ^
  --collect-data googleapiclient ^
  --exclude-module faster_whisper --exclude-module ctranslate2 --exclude-module torch ^
  --name drive_feasibility drive_feasibility.py
```

`--collect-data googleapiclient` bundles Google's Drive API description, so the tool doesn't need to fetch it at run time. Never commit `client_secret.json` to the repository; the included `.gitignore` keeps it and sign-in tokens out. Expect the executable to be roughly 80 to 120 MB.

**For an organization:** a folder containing `drive_feasibility.exe`, `client_secret.json` next to it (the program looks for it in its own folder), and optionally `run_drive_limited.bat`, a double-click launcher that applies a download limit (edit the number inside it). Running `drive_feasibility.exe --check` from a Command Prompt in that folder confirms the setup before anyone signs in.


### In-depth exploration (`drive_explore.py`)

For when you're permitted to see the data itself. It reads from Google Drive exactly like `drive_feasibility.py` (each recording is downloaded, analyzed and deleted, no audio is kept, and progress is checkpointed after every recording) and uses the same sign-in, `client_secret.json` and download limits. It writes the standard feasibility report plus:

- **`exploration_report.txt` / `.json`** with four sections:
  1. *Turns by pitch class.* Speech is split into turns (at pauses of 300 ms or more, and wherever the pitch class changes for at least 0.6 s, so a child answering a teacher without pausing still becomes two turns). Each turn is child-range (70%+ of voiced frames at 250 Hz or above), adult-range (30% or less), mixed, or unclear (too little voicing). Reports hours and median turn length per class, child/adult switches per minute, and child-range speech by turn length. No AI model is involved.
  2. *Whisper-ready segments.* Same-class turns joined across pauses under 0.5 s, capped at 30 s (longer stretches are split at the quietest point after 20 s), padded by 0.15 s, and dropped below 1 s. Reports segment counts and hours per class and by length.
  3. *Transcripts and captions.* Every manifest transcription and caption cue is labelled English, Swahili, English-Swahili mixed, another supplied language, or unrecognized, by counting words from short lists of very common words. Reports utterances and hours per label, the code-switching rate, word count and vocabulary size.
  4. *Near-duplicates and overlaps.* See below.
- **`folder_summary.csv`:** per top-level folder and per immediate folder: recordings, audio and speech hours, child-range, adult-range and mixed turn hours, mean SNR, narrowband recordings, and labelled recordings. Folders often map to schools, classes or sessions, which is what you need for speaker-disjoint train/test splits.
- **`recordings.csv`:** one row per recording with its Drive link (opening it plays the file in Google Drive, so spot-checks need no local copy), turn minutes by class, segment count, SNR, bandwidth, year, labels, and any near-duplicates.
- **`segments_manifest.jsonl`:** NeMo-style, one line per proposed segment, with `audio_filepath` (the path in Drive), `drive_file_id`, `offset`, `duration`, `speaker_class`, `median_f0` and `text`. Text is filled from a manifest transcription when a recording is a single segment with a single transcription (as in AfriVoice), otherwise from caption cues overlapping the segment.
- **`near_duplicates.csv`** and **`top_words.csv`** (the 500 most frequent transcript words).

`--report-only` writes only the two reports and the folder map. Everything else contains personal data: file names, Drive links, transcript text. The tool writes a `README_FIRST.txt` into the output folder saying so. The checkpoint stores only hashed IDs, measurements and fingerprints, never names or text.

**Near-duplicate detection.** Exact copies are already skipped using Drive's checksums. The fingerprint catches copies that differ in encoding, volume, quality or length: it marks loudness peaks in speech (roughly syllables), and hashes the intervals to the next three peaks together with whether each is louder than the last. Recordings sharing many hashes at a consistent time offset, covering at least a quarter of the landmarks in the overlapping stretch, are paired. A pair is a *near-identical copy* if durations agree within 5% and the overlap covers most of the shorter recording, otherwise a *partial overlap* (for example an excerpt), and `b_starts_at_s_in_a` gives where the shorter recording starts within the longer one. Because it relies on the irregular rhythm of speech, it isn't meaningful for music or synthetic, perfectly repetitive audio. `--no-fingerprints` turns it off.

**Word lists.** The built-in lists cover English, Swahili and a rough set of Sheng markers. Add others with `--wordlists FILE.json`, a JSON object mapping a language name to its common words, for example:

```json
{"Kikuyu": ["nĩ", "mũndũ", "andũ", "wakwa", "ũguo", "rĩu", "ciana", "mwana"],
 "Dholuo": ["..."]}
```

Have a fluent speaker choose 30 to 60 very common words that don't also occur in English or Swahili; diacritics are kept, so include both spellings if transcripts sometimes leave them out. Only English and Swahili hits decide the English/Swahili/mixed label; other lists label an utterance only when it contains no English or Swahili words, and their hits are counted separately.

**Language checks:** off by default. With `--language-id` (and the same `--lid-model`, `--lid-windows-per-file` and `--lid-max-windows` options as the other tools), each recording's language is checked while it's downloaded, and the explorer adds a "language by speaker class" section: English, non-English and mixed hours for child-range and adult-range turns, and child-range non-English hours by detected language. The folder map gains each folder's non-English share, `recordings.csv` gains each recording's language shares, and each line of `segments_manifest.jsonl` gains its recording's majority language (`file_language`). Without `--language-id`, the explorer uses no AI model at all. The standard `drive_explore.exe` doesn't include the Whisper engine, so use `drive_explore_lid.exe` for language checks.

**Options:** the same as `drive_feasibility.py` (`--sample-files`, `--max-download-gb`, `--max-minutes-per-file`, `--workers`, `--vad`, `--fresh`, `--keep-checkpoint`, `--keep-login`, `--check`, and the language options above), plus `--wordlists`, `--no-fingerprints` and `--report-only`. It's built by the `build-windows-all.yml` workflow alongside the other two programs, and needs all three scripts in the repository.


Note: This script is currently only used in conjunction with drive_explore_lid.py below. 

### Explorer with language checks always on (`drive_explore_lid.exe`)

For when you have full, unrestricted access. It's `drive_explore.py` with language checks permanently on: there's no option to turn them off. If the language model can't be loaded, it shows the reason and asks "Language Detector cannot be loaded: Would you like me to continue? (y/n)". Typing `y` runs the full exploration without language checks, and both reports then say why language results are missing; typing `n` stops before anything is downloaded. It never continues without an answer: if no answer can be given (for example, input isn't available), it stops. Its outputs are the explorer's, including the language sections. Everything else (folder link, `--sample-files`, `--max-download-gb`, checkpoints, `--check`) works as for `drive_explore.py`.

**Building it:** the repository needs `asr_feasibility.py`, `drive_feasibility.py`, `drive_explore.py` and `drive_explore_lid.py`, plus the `build-drive-explore-lid.yml` workflow, which builds only this program. Run it from the Actions tab ("Build drive_explore_lid"); it also runs when a version tag is pushed. Each run produces two artifacts, `drive_explore_lid-windows` (the program) and `model-faster-whisper-small` (the model), and a version tag also publishes both on a release, with the model as `models-faster-whisper-small.zip`. On GitHub's Windows machine the workflow runs `--check` offline with the real model and fails the build unless the model loads and runs.

**Getting the model:**

- *From the release:* unzip `models-faster-whisper-small.zip` next to the program; it already contains the `models` folder.
- *From the artifact:* the zip contains a `faster-whisper-small` folder. Create a `models` folder next to the program and put `faster-whisper-small` inside it.
- *By hand:* download `config.json`, `model.bin` (484 MB), `tokenizer.json` and `vocabulary.txt` from the **Files and versions** tab of **Systran/faster-whisper-small** on Hugging Face (MIT license) into `models/faster-whisper-small`. Don't use **faster-whisper-small.en**, which is English-only and can't tell English from other languages.

**Setting it up:** put these in one folder:

```
drive_explore_lid.exe
client_secret.json
models/
    faster-whisper-small/
        config.json
        model.bin
        tokenizer.json
        vocabulary.txt
```

Then run `drive_explore_lid.exe --check`. It passes only when the program, Google libraries and language model are all working, including a short test run of the model. With exactly one model in `models/`, the program selects it automatically. To ship a smaller model, change `LID_MODEL: small` to `base` in the workflow (about 145 MB instead of 485 MB, somewhat less accurate) and use `faster-whisper-base` in place of `faster-whisper-small` above.

**Size:** expect the executable to be roughly 150 to 200 MB, mostly the Whisper engine (CTranslate2), plus the model. The build leaves out PyAV, which faster-whisper uses only to decode audio files itself; this tool always passes it already-decoded audio, so a placeholder module stands in for it.

### Checkpoint and resume

Progress is written to `resume_checkpoint__LOCAL_ONLY.jsonl` in the output folder as an append-only JSON Lines log, with each line flushed and fsynced as soon as a file finishes. A power cut therefore loses at most the files in progress at that moment; a line cut off mid-write is detected and discarded on the next run. Reports are written via a temporary file and an atomic rename, so a half-written report never replaces a good one.

- **What's stored:** format information and the per-file statistics used to build the report. Files are identified by a hash of their relative path, size and modification time, so no file or folder names are written. ffmpeg error text, which can contain paths, is not stored.
- **When it resumes:** only if the folder and the analysis settings (tool version, `--max-minutes-per-file`, speech detector, pitch on/off, format reader) match the saved run. Otherwise it starts fresh and says why. A file that has been modified since gets a new ID and is re-analyzed. `--sample-files` can change between runs, since per-file results don't depend on it.
- **Ctrl+C or closing the window:** workers stop within about a second, finished files are kept, and the tool exits with a message explaining how to resume.
- **Not checkpointed:** the optional `--language-id` step, which runs last and is redone in full if interrupted.

  Reclick the executable to restart, otherwise use the command ".\drive_feasibility-windows.exe", along with the same arguments previously used like --worker, --keep-checkpoint, etc

### Interpreting the report

| Signal | Encouraging | Concerning |
|---|---|---|
| Estimated speech hours in child-range files | Tens of hours or more | Under a few hours |
| Speech activity | Most hours in files over 30% speech | Most hours in files under 10% speech (mostly silence, music or video without talk) |
| SNR | Most hours above 10 dB, or noise levels that match your deployment | Large share below 5 dB |
| Clipping | None or trace | Many files with frequent clipping |
| Child/adult mix | Most speech hours in mostly-child or mixed files | Most speech hours in mostly-adult files |
| Speech segments | Most speech time in runs under 10 s, several pauses per minute | Large share in runs over 30 s (lectures, monologues), which means alignment work |
| Bandwidth | Mostly full bandwidth | Large share narrowband; children's speech carries more high-frequency detail than adults' |
| Existing labels | Substantial caption-covered or manifest-transcribed hours on child-heavy recordings | No captions, manifests or documents, which means budgeting for human transcription |
| Manifest linking | Nearly all entries linked to a recording | Many entries whose audio wasn't found (incomplete download or unextracted archives) or ambiguous matches |
| Transcription density | Nearly all in the typical range | Many sparse (condensed) or dense (mismatched) entries |
| Speakers (manifest) | Many distinct speakers, hours spread evenly | A few speakers contribute most of the hours |
| Recording dates | Recent recordings, or dates that match the current consent process; most years from manifest or embedded dates | Mostly older recordings made under earlier or unknown consent terms, or most years from file dates only |
| Formats | Uncompressed or 96 kb/s and above | Many files under 64 kb/s |

The child-range figures are upper bounds on child speech: adult women's voices overlap the lower part of the child range, and shouting, singing, and some noise push pitch up. Treat them as screening signals and confirm with a listened-to sample. Caption coverage shows how much audio has some timed text, not whether that text is accurate or verbatim, since subtitles are often condensed.
