#!/usr/bin/env python3
"""
drive_explore with language checks always on
============================================

A backup program for when you have full, unrestricted access to the data. It is
drive_explore.py with Whisper language identification permanently switched on:
there is no option to turn it off. If the language model can't be loaded, it
asks whether to continue without language checks, and only continues if told to.

The model is loaded from a folder next to the program, for example
models/faster-whisper-small, so nothing has to be downloaded. If no such folder
is there, the model is downloaded once (about 485 MB for 'small').

Usage: as drive_explore.py, for example
    drive_explore_lid "https://drive.google.com/drive/folders/XXXX" --sample-files 300
    drive_explore_lid --check        # installation check, including a test run of the model
"""
import sys
from pathlib import Path

HERE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
sys.path.insert(0, str(HERE))
import asr_feasibility as af       # noqa: E402
import drive_feasibility as dfz    # noqa: E402
import drive_explore as dx         # noqa: E402

NAME = "drive_explore_lid"


def model_size(argv):
    """--lid-model if given; else the single model shipped in models/; else 'small'."""
    if "--lid-model" in argv:
        i = argv.index("--lid-model")
        if i + 1 < len(argv):
            return argv[i + 1]
    folder = af.app_dir() / "models"
    shipped = sorted(p.name[len("faster-whisper-"):] for p in folder.glob("faster-whisper-*")
                     if (p / "model.bin").exists()) if folder.is_dir() else []
    return shipped[0] if len(shipped) == 1 else "small"


def load_model_or_ask(size):
    """Load the model. If that fails, ask whether to continue without language
    checks. Returns (model or None, continue?)."""
    model, err = af.load_lid_model(size)
    if model is not None:
        return model, True
    print(f"\n  (Reason: {err.replace('skipped: ', '')}. The model folder is models/faster-whisper-{size},")
    print(f"  next to the program, from the model zip.)")
    while True:
        try:
            answer = input("Language Detector cannot be loaded: Would you like me to continue? (y/n) ")
        except EOFError:
            print()
            return None, False          # no answer available: don't continue unasked
        answer = answer.strip().lower()
        if answer in ("y", "yes"):
            print("Continuing without language checks.\n")
            return None, True
        if answer in ("n", "no"):
            return None, False
        print("Please type y or n.")


def check(argv):
    import argparse
    import numpy as np
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--client-secret", default=str(dfz.APP_DIR / "client_secret.json"))
    args, _ = p.parse_known_args(argv[1:])
    code = dfz.self_check(args, name=NAME, summary=False)
    size = model_size(argv)
    ok = False
    if not af.local_lid_model(size):
        mb = af.LID_MODEL_MB.get(size)
        print(f"  [!!] language model '{size}': no models/faster-whisper-{size} folder next to the "
              f"program; it would be downloaded on first use" + (f" (about {mb} MB)" if mb else ""))
    else:
        model, err = af.load_lid_model(size)
        if model is None:
            print(f"  [!!] language model '{size}': {err}")
        else:
            try:
                t = np.arange(af.SR * 3) / af.SR
                _, info = model.transcribe((0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32),
                                           beam_size=1, vad_filter=False, without_timestamps=True)
                ok = bool(info.all_language_probs or info.language)
            except Exception as e:
                print(f"  [!!] language model '{size}': loaded but failed a test run ({type(e).__name__})")
            if ok:
                print(f"  [OK] language model '{size}' (offline test run)")
    if code or not ok:
        print("Some items need attention (see above).")
        return 1
    print("Ready." if Path(args.client_secret).exists() else
          "Ready, once client_secret.json is placed next to this program.")
    return 0


def main():
    argv = list(sys.argv)
    if "--check" in argv:
        return check(argv)
    if any(a in ("-h", "--help") for a in argv):
        dx.FORCE_LID = True
        sys.argv = [NAME] + argv[1:]
        return dx.main()
    size = model_size(argv)
    print(f"{NAME}: language checks are always on. Loading the language model...")
    model, go_on = load_model_or_ask(size)
    if not go_on:
        print("Stopped. Nothing was downloaded or analyzed.")
        return af.finish(True, 1)
    if model is not None:
        af.load_lid_model = lambda _size: (model, None)   # the explorer reuses the loaded model
    else:
        dx.LID_SKIP_REASON = ("not run: the language detector could not be loaded, and the "
                              "run continued without it at the user's request")
    dx.FORCE_LID = True
    if "--lid-model" not in argv:
        argv[1:1] = ["--lid-model", size]
    sys.argv = [NAME] + argv[1:]
    return dx.main()


if __name__ == "__main__":
    sys.exit(main())
