"""One command before uploading: regenerate the submission under the team name and check every
requirement of SUBMISSION.md that a script can check.

    python -m solution.preflight --name <team_name>            # regenerate + all checks
    python -m solution.preflight --name <team_name> --fresh-cache   # also rebuild caches from raw files

Checks: pinned environment, regeneration, the official validator, the file's bytes (header, row
count, frame order, labels, line endings), no abstention on graded frames, the recorded SHA-256,
the repository (clean, pushed, no match data tracked, public) and the technical document.

Format-level checks only: it never reads the ground truth and prints counts, not data.
"""

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
from importlib import metadata

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES  # noqa: E402  (the organisers' constants; opens no data)

SHA_FILE = os.path.join(HERE, "models", "submission.sha256")
DOC = os.path.join(HERE, "README.md")
OUT_DIR = os.path.join(ROOT, "submissions")
REQ = os.path.join(ROOT, "requirements-solution.txt")
# what SUBMISSION.md asks the technical document to cover -> a heading that must be present
DOC_SECTIONS = {"the approach": "Approach", "assets used and dropped": "Assets",
                "training data": "Training data", "possession rules": "Possession rules",
                "limitations": "Limitations", "reproduction": "Reproduction"}
ORGANISER_CSVS = {"evaluation/gt/BAR-ATH_possession_gt_public.csv",
                  "evaluation/results/baseline_raw.csv", "evaluation/results/baseline_smooth25.csv",
                  "evaluation/results/baseline_smooth50.csv",
                  "evaluation/results/baseline_smooth125.csv"}
BUNDLE = re.compile(r"(\.mp4|_atd\.txt|_atd\.xml|_homography\.csv|_smart_tagging\.xml|\.npy|\.npz|"
                    r"\.parquet)$")

RESULTS = []


def report(status, item, detail=""):
    RESULTS.append(status)
    print("{:<5} {}{}".format(status, item, "  -- " + detail if detail else ""), flush=True)


def git(*args):
    r = subprocess.run(["git"] + list(args), cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def check_environment():
    ok = sys.version_info[:2] == (3, 12)
    report("PASS" if ok else "WARN", "Python 3.12", "running {}.{}.{}".format(*sys.version_info[:3]))
    bad = []
    for line in open(REQ, encoding="utf-8"):
        m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", line.strip())
        if not m:
            continue
        try:
            have = metadata.version(m.group(1))
        except metadata.PackageNotFoundError:
            have = "missing"
        if have.split("+")[0] != m.group(2):      # "+cpu" / "+cu128" are builds of the same version
            bad.append("{} {} (pinned {})".format(m.group(1), have, m.group(2)))
    report("PASS" if not bad else "FAIL", "packages match requirements-solution.txt", "; ".join(bad))


def regenerate(name, fresh):
    if fresh:
        shutil.rmtree(os.path.join(ROOT, "data", "cache"), ignore_errors=True)
    out = os.path.join(OUT_DIR, name + ".csv")
    if os.path.isfile(out):
        os.remove(out)                      # never check a stale file from an earlier run
    r = subprocess.run([sys.executable, "-W", "ignore", "-m", "solution.predict", "--name", name],
                       cwd=ROOT, capture_output=True, text=True)
    ok = r.returncode == 0 and os.path.isfile(out)
    report("PASS" if ok else "FAIL", "regenerated submissions/{}.csv".format(name),
           "" if ok else (r.stdout[-600:] + r.stderr[-600:]).replace("\n", " | "))
    return out if ok else None


def check_validator(path):
    v = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"),
                        path], cwd=ROOT, capture_output=True, text=True)
    ok = v.returncode == 0 and v.stdout.strip().startswith("OK")
    report("PASS" if ok else "FAIL", "official validator (evaluation/validate_submission.py)",
           v.stdout.strip().splitlines()[0][:120] if v.stdout.strip() else "no output")


def check_bytes(path):
    """The format rules of SUBMISSION.md, checked on the bytes, independently of the validator."""
    raw = open(path, "rb").read()
    n = GAMES[GAME]["n_video_frames"]
    lines = raw.split(b"\n")
    problems = []
    if raw[:3] == b"\xef\xbb\xbf":
        problems.append("byte-order mark")
    if b"\r" in raw:
        problems.append("carriage returns")
    if not raw.endswith(b"\n") or lines[-1] != b"":
        problems.append("no final newline")
    if lines[0] != b"frame,label":
        problems.append("header is not exactly frame,label")
    body = lines[1:-1]
    if len(body) != n:
        problems.append("{} data rows, expected {}".format(len(body), n))
    else:
        frames = np.array([ln.split(b",")[0] for ln in body])
        labels = np.array([ln.split(b",")[1] if ln.count(b",") == 1 else b"?" for ln in body])
        if not (frames == np.char.encode(np.arange(n).astype(str))).all():
            problems.append("frame column is not 0..{} in order".format(n - 1))
        if not np.isin(labels, [b"H", b"A", b"D", b"X"]).all():
            problems.append("labels outside H/A/D/X or blank cells")
    report("PASS" if not problems else "FAIL",
           "file bytes: header, {} rows, frame order, labels, \\n line ends, no BOM".format(n),
           "; ".join(problems) if problems else "{:.2f} MB".format(len(raw) / 1e6))
    if problems:
        return
    # graded windows from the organisers' constants (games.py): inclusive video-frame spans
    g = GAMES[GAME]
    x_in, graded = 0, 0
    inside = np.zeros(n, bool)
    for h, (lo, hi) in g["halves"].items():
        a, b = lo - g["offsets"][h], hi - g["offsets"][h]
        inside[a:b + 1] = True
    x_in = int((labels[inside] == b"X").sum())
    graded = int(inside.sum())
    report("PASS" if x_in == 0 else "FAIL", "no X (abstention) on graded frames",
           "{} X among {} graded frames; {} of {} ungraded frames are X".format(
               x_in, graded, int((labels[~inside] == b"X").sum()), n - graded))


def check_sha(path):
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    want = open(SHA_FILE).read().strip() if os.path.isfile(SHA_FILE) else None
    report("PASS" if got == want else "FAIL", "SHA-256 equals solution/models/submission.sha256",
           got if got == want else "got {} expected {}".format(got, want))
    return got


def check_repo():
    code, _ = git("rev-parse", "--is-inside-work-tree")
    if code != 0:
        report("WARN", "git repository", "not a git checkout; repository checks skipped")
        return
    _, dirty = git("status", "--porcelain")
    report("PASS" if not dirty else "FAIL", "working tree clean (the file comes from committed code)",
           "{} uncommitted change(s)".format(len(dirty.splitlines())) if dirty else "")
    _, head = git("rev-parse", "HEAD")
    code, up = git("rev-parse", "@{u}")
    report("PASS" if code == 0 and head == up else "FAIL", "HEAD is pushed to its remote branch",
           head[:10] if code == 0 and head == up else "local {} / remote {}".format(head[:10], up[:10]))
    _, files = git("ls-files")
    tracked = files.splitlines()
    leaked = [f for f in tracked if (f.startswith("data/") and f != "data/README.md")
              or BUNDLE.search(f) or (f.endswith(".csv") and f not in ORGANISER_CSVS)
              or f.startswith("submissions/")]
    report("PASS" if not leaked else "FAIL", "no match data or outputs tracked by git",
           ", ".join(leaked[:5]))
    # the repository this branch is pushed to -- not whichever remote `gh` would pick by default
    # (a fork also has the organisers' repository as a remote, and that one is public)
    code, tracking = git("rev-parse", "--abbrev-ref", "@{u}")
    remote = tracking.split("/")[0] if code == 0 and "/" in tracking else "origin"
    _, url = git("remote", "get-url", remote)
    url = url[:-4] if url.endswith(".git") else url
    if shutil.which("gh") and url:
        r = subprocess.run(["gh", "repo", "view", url, "--json", "visibility,url", "--jq",
                            ".visibility + \" \" + .url"], cwd=ROOT, capture_output=True, text=True)
        out = r.stdout.strip()
        if r.returncode == 0 and out:
            public = out.startswith("PUBLIC")
            report("PASS" if public else "TODO", "repository is public", out if public else
                   out + "  (make it public when you submit)")
            return
    report("TODO", "repository is public", "could not check {}; the form needs a public URL".format(
        url or "the remote"))


def check_document(name, sha):
    text = open(DOC, encoding="utf-8").read()
    heads = [ln for ln in text.splitlines() if ln.startswith("#")]
    missing = [k for k, word in DOC_SECTIONS.items()
               if not any(word.lower() in h.lower() for h in heads)]
    report("PASS" if not missing else "FAIL", "technical document covers the six required topics",
           "missing: " + ", ".join(missing) if missing else "")
    if sha:
        report("PASS" if sha in text else "FAIL", "technical document quotes this file's SHA-256")
    os.makedirs(OUT_DIR, exist_ok=True)
    md = os.path.join(OUT_DIR, name + ".md")
    shutil.copyfile(DOC, md)
    pdf = os.path.join(OUT_DIR, name + ".pdf")
    report("PASS", "technical document copied to submissions/{}.md".format(name),
           "PDF also present" if os.path.isfile(pdf) else "upload the .md, or a PDF named {}.pdf"
           .format(name))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True, help="team name exactly as registered")
    ap.add_argument("--fresh-cache", action="store_true", help="rebuild caches from the raw files")
    ap.add_argument("--skip-git", action="store_true")
    a = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.-]*", a.name):
        sys.exit("team name has characters that are unsafe in a file name: {!r}".format(a.name))
    check_environment()
    path = regenerate(a.name, a.fresh_cache)
    sha = None
    if path:
        check_validator(path)
        check_bytes(path)
        sha = check_sha(path)
    if not a.skip_git:
        check_repo()
    check_document(a.name, sha)
    fails, todos = RESULTS.count("FAIL"), RESULTS.count("TODO")
    print()
    if fails:
        print("NOT READY: {} check(s) failed.".format(fails))
        sys.exit(1)
    print("READY{}. Upload submissions/{}.csv and the technical document; give the form the "
          "public repository URL and the team name exactly as registered.".format(
              " once the TODO item(s) above are done" if todos else "", a.name))


if __name__ == "__main__":
    main()
