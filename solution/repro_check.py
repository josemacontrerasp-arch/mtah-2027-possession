"""Regenerate the submission and compare it byte for byte with the recorded SHA-256.

    python -m solution.repro_check                       # this interpreter
    python -m solution.repro_check --python <other python>   # e.g. a fresh CPU-only venv
    python -m solution.repro_check --fresh-cache         # also rebuild every cache from raw files

The expected hash lives in solution/models/submission.sha256 (written by `--record`). Runs only
format-level checks on the provided match: row count, the official validator, the hash.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SHA_FILE = os.path.join(HERE, "models", "submission.sha256")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--name", default="repro_check")
    ap.add_argument("--fresh-cache", action="store_true")
    ap.add_argument("--record", action="store_true", help="store this run's hash as expected")
    a = ap.parse_args()
    if a.fresh_cache:
        cache = os.path.join(ROOT, "data", "cache")
        shutil.rmtree(cache, ignore_errors=True)
        if os.path.isdir(cache):
            sys.exit("could not delete data/cache (is another program using it?)")
    out = os.path.join(ROOT, "submissions", a.name + ".csv")
    if os.path.isfile(out):
        os.remove(out)                     # never compare a stale file from an earlier run
    r = subprocess.run([a.python, "-W", "ignore", "-m", "solution.predict", "--name", a.name],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(out):
        # keep the child's output in a file rather than on screen: an error message from a parser
        # can quote the data it choked on
        log = os.path.join(ROOT, "submissions", a.name + "_predict.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "w", encoding="utf-8", errors="replace") as fh:
            fh.write(r.stdout + "\n--- stderr ---\n" + r.stderr)
        sys.exit("prediction failed (exit code {}); its output is in {}".format(
            r.returncode, os.path.relpath(log, ROOT)))
    got = sha256(out)
    v = subprocess.run([a.python, os.path.join(ROOT, "evaluation", "validate_submission.py"), out],
                       cwd=ROOT, capture_output=True, text=True)
    valid = v.stdout.strip().startswith("OK")
    print("validator:", "OK" if valid else "FAIL")
    print("sha256   :", got)
    if a.record:
        with open(SHA_FILE, "w") as fh:
            fh.write(got + "\n")
        print("recorded as expected hash")
        return
    want = open(SHA_FILE).read().strip() if os.path.isfile(SHA_FILE) else None
    if want is None:
        print("no recorded hash yet (run with --record)")
    elif got == want and valid:
        print("REPRODUCIBLE: byte-identical to the recorded submission")
    else:
        sys.exit("NOT REPRODUCED: expected {}".format(want))


if __name__ == "__main__":
    main()
