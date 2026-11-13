"""One command before uploading: regenerate the submission under the team name and check every
requirement of SUBMISSION.md that a script can check.

    python -m solution.preflight --name <team_name>
    python -m solution.preflight --name <team_name> --no-clone     # skip the fresh-clone test

Checks
* environment: Python 3.12, every requirement pinned and installed at the pinned version;
* the file: regenerated from the raw match files (caches rebuilt), the organisers' validator, its
  bytes (header, row count, frame order, labels, line endings), no abstention on graded frames,
  the recorded SHA-256;
* the repository: clean, its commit is the head of the remote's default branch, no match data
  tracked now or at any point in its history, readable without credentials (public), nothing new
  in the organisers' repository, and a fresh clone of the remote regenerates the same file;
* the technical document: a section for each required topic, quotes the file's SHA-256, PDF fresh.

Format-level checks only. It never reads the ground truth and prints counts and file names, never
data; a failed child process leaves its output in a log file under submissions/ instead.
"""

import argparse
import hashlib
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from importlib import metadata

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import DATA_DIR, GAME, GAMES, asset_paths  # noqa: E402  (constants only; opens no data)

SHA_FILE = os.path.join(HERE, "models", "submission.sha256")
DOC = os.path.join(HERE, "README.md")
OUT_DIR = os.path.join(ROOT, "submissions")
REQ = os.path.join(ROOT, "requirements-solution.txt")
LOCK = os.path.join(ROOT, "requirements-solution.lock.txt")
ORGANISERS = "https://github.com/metrica-sports/more-than-a-hack-2027"
# what SUBMISSION.md asks the technical document to cover -> a word its section heading must contain
DOC_SECTIONS = {"the approach": "approach", "assets used and dropped": "assets",
                "training data": "training data", "possession rules": "possession rules",
                "limitations": "limitations", "reproduction": "reproduction"}
ORGANISER_CSVS = {"evaluation/gt/bar-ath_possession_gt_public.csv",
                  "evaluation/results/baseline_raw.csv", "evaluation/results/baseline_smooth25.csv",
                  "evaluation/results/baseline_smooth50.csv",
                  "evaluation/results/baseline_smooth125.csv"}
# file kinds that must never be tracked: the match bundle, derived arrays, videos, archives, outputs
LEAK = re.compile(r"(\.(mp4|mov|avi|mkv|npy|npz|parquet|feather|pkl|pickle|h5|hdf5|zip|7z|tar|gz|"
                  r"gzip|pl3container)|_atd\.txt|_atd\.xml|_homography\.csv|_smart_tagging\.xml|"
                  r"_possession_gt\.csv|_possession_gt_reference\.csv)$")
MAX_TRACKED_BYTES = 30 * 1024 * 1024
BAD_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

RESULTS = []


def report(status, item, detail=""):
    RESULTS.append((status, item))
    print("{:<5} {}{}".format(status, item, "  -- " + detail if detail else ""), flush=True)


def run(cmd, cwd=ROOT, env=None, timeout=None):
    """(exit code, stdout, stderr) of a child process; never raises."""
    e = dict(os.environ, PYTHONIOENCODING="utf-8")
    e.update(env or {})
    try:
        r = subprocess.run(cmd, cwd=cwd, env=e, capture_output=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except FileNotFoundError:
        return 127, "", "{} not found".format(cmd[0])
    except subprocess.TimeoutExpired:
        return 124, "", "timed out"


def git(*args, cwd=ROOT, env=None, timeout=120):
    code, out, _ = run(["git"] + list(args), cwd=cwd, env=env, timeout=timeout)
    return code, out.strip()


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def recorded_sha():
    return open(SHA_FILE).read().strip() if os.path.isfile(SHA_FILE) else None


def _pins(path):
    """[(name, version)] of the `name==version` lines of a requirements file, and the other lines."""
    pins, other = [], []
    for line in open(path, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.fullmatch(r"([A-Za-z0-9_.-]+)\s*==\s*([A-Za-z0-9_.!+-]+)", line)
        if m:
            pins.append((m.group(1), m.group(2)))
        else:
            other.append(line)
    return pins, other


def _installed(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def check_environment():
    ok = sys.version_info[:2] == (3, 12)
    report("PASS" if ok else "WARN", "Python 3.12", "running {}.{}.{}".format(*sys.version_info[:3]))
    pins, other = _pins(REQ)
    bad = ["not pinned with ==: " + o for o in other]
    for name, want in pins:
        have = _installed(name)
        # "+cpu" / "+cu128" are builds of the same version
        if have is None or have.split("+")[0] != want.split("+")[0]:
            bad.append("{} {} (pinned {})".format(name, have or "missing", want))
    report("PASS" if not bad else "FAIL",
           "every requirement is pinned and installed ({} packages)".format(len(pins)),
           "; ".join(bad))
    if os.path.isfile(LOCK):
        lock, _ = _pins(LOCK)
        diff = ["{} {} (lock {})".format(n, _installed(n), w) for n, w in lock
                if _installed(n) is not None and _installed(n).split("+")[0] != w.split("+")[0]]
        report("PASS" if not diff else "WARN", "indirect dependencies equal the lock file",
               "; ".join(diff[:6]))


def regenerate(name, keep_cache):
    paths = asset_paths(GAME, source="tactical")
    missing = [k for k in ("atd_meta", "atd_raw") if not os.path.isfile(paths[k])]
    if missing:
        report("FAIL", "match bundle found", "tactical {} not under {} (unpack the bundle into "
               "data/ or set MTAH_DATA_DIR)".format(" and ".join(missing), DATA_DIR))
        return None
    cache = os.path.join(ROOT, "data", "cache")
    if not keep_cache:
        shutil.rmtree(cache, ignore_errors=True)
        if os.path.isdir(cache):
            report("FAIL", "caches removed before regenerating",
                   "could not delete data/cache (is another program using it?)")
            return None
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, name + ".csv")
    try:
        if os.path.isfile(out):
            os.remove(out)                   # never check a stale file from an earlier run
    except OSError:
        report("FAIL", "old submissions/{}.csv removed".format(name), "the file is open elsewhere")
        return None
    code, so, se = run([sys.executable, "-W", "ignore", "-m", "solution.predict", "--name", name])
    ok = code == 0 and os.path.isfile(out)
    detail = "from the raw files, caches rebuilt" if not keep_cache else "caches reused"
    if not ok:
        log = os.path.join(OUT_DIR, name + "_predict.log")
        with open(log, "w", encoding="utf-8") as fh:
            fh.write(so + "\n--- stderr ---\n" + se)
        detail = "exit code {}; output kept in {}".format(code, os.path.relpath(log, ROOT))
    report("PASS" if ok else "FAIL", "regenerated submissions/{}.csv".format(name), detail)
    return out if ok else None


def check_validator(path):
    code, so, _ = run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"),
                       path])
    lines = [ln.strip() for ln in so.strip().splitlines() if ln.strip()]
    ok = code == 0 and bool(lines) and lines[0].startswith("OK")
    said = next((ln for ln in lines if ln.startswith("reports ")), "")
    report("PASS" if ok else "FAIL", "organisers' validator (evaluation/validate_submission.py)",
           said if ok else " | ".join(lines[:2]) or "no output")


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
    labels = None
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
           "; ".join(problems) if problems else "{:,} bytes".format(len(raw)))
    if problems:
        return
    # graded windows from the organisers' constants (games.py): inclusive video-frame spans
    g = GAMES[GAME]
    inside = np.zeros(n, bool)
    for h, (lo, hi) in g["halves"].items():
        inside[lo - g["offsets"][h]:hi - g["offsets"][h] + 1] = True
    x_in = int((labels[inside] == b"X").sum())
    report("PASS" if x_in == 0 else "FAIL", "an answer on every graded frame (no X)",
           "{} X among {} graded frames; {} of {} ungraded frames are X".format(
               x_in, int(inside.sum()), int((labels[~inside] == b"X").sum()), int((~inside).sum())))


def check_sha(path):
    got, want = sha256(path), recorded_sha()
    report("PASS" if got == want else "FAIL", "SHA-256 equals solution/models/submission.sha256",
           got if got == want else "got {} expected {}".format(got, want))
    return got


def _flag(paths, sizes=None):
    """Tracked paths that look like match data, derived data or outputs."""
    out = []
    for p in paths:
        q = p.lower()
        if ((q.startswith("data/") and q != "data/readme.md") or q.startswith("submissions/")
                or q.startswith("evaluation/reports/") or LEAK.search(q)
                or (q.endswith(".csv") and q not in ORGANISER_CSVS)
                or (sizes is not None and sizes.get(p, 0) > MAX_TRACKED_BYTES)):
            out.append(p)
    return out


def _remote():
    code, tracking = git("rev-parse", "--abbrev-ref", "@{u}")
    name = tracking.split("/")[0] if code == 0 and "/" in tracking else "origin"
    _, url = git("remote", "get-url", name)
    return name, (url[:-4] if url.endswith(".git") else url)


def check_repo():
    """Returns (remote url, True if this commit is the head of its default branch)."""
    code, _ = git("rev-parse", "--is-inside-work-tree")
    if code != 0:
        report("SKIP", "repository checks", "not a git checkout (or git is not installed)")
        return None, False
    code, dirty = git("status", "--porcelain")
    report("PASS" if code == 0 and not dirty else "FAIL",
           "working tree clean (the file comes from committed code)",
           "git status failed" if code != 0 else
           "{} uncommitted change(s)".format(len(dirty.splitlines())) if dirty else "")
    _, head = git("rev-parse", "HEAD")
    name, url = _remote()

    # the commit a clone of the remote checks out: the head of its default branch
    code, out = git("ls-remote", name, "HEAD", timeout=60)
    remote_head = out.split()[0] if code == 0 and out else None
    pushed = remote_head == head
    report("PASS" if pushed else "FAIL", "this commit is the head of the remote's default branch",
           "{} at {}".format(head[:10], url) if pushed else
           "could not reach {}".format(url or name) if remote_head is None else
           "local {} / remote default branch {} -- push first".format(head[:10], remote_head[:10]))

    # nothing that looks like match data, now or ever (a public repository shows its history too)
    code, listing = git("ls-files", "-z")
    tracked = [p for p in listing.split("\0") if p]
    sizes = {p: os.path.getsize(os.path.join(ROOT, p)) for p in tracked
             if os.path.isfile(os.path.join(ROOT, p))}
    leaked = _flag(tracked, sizes)
    report("PASS" if code == 0 and not leaked else "FAIL",
           "no match data, derived data or outputs tracked ({} files)".format(len(tracked)),
           "git ls-files failed" if code != 0 else ", ".join(leaked[:5]))
    code, hist = git("log", "--all", "--diff-filter=A", "--name-only", "--format=", "-z")
    ever = sorted({p for p in hist.replace("\n", "\0").split("\0") if p})
    leaked = _flag(ever)
    report("PASS" if code == 0 and not leaked else "FAIL",
           "none in the history of any branch either ({} paths ever added)".format(len(ever)),
           "git log failed" if code != 0 else ", ".join(leaked[:5]))

    # public = readable without credentials, which is how the organisers will clone it
    anon = {"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never", "GIT_ASKPASS": "echo"}
    public = False
    if url.startswith("http"):
        code, out = git("-c", "credential.helper=", "ls-remote", url, "HEAD", env=anon, timeout=60)
        public = code == 0 and bool(out)
    report("PASS" if public else "TODO", "repository is public (readable without credentials)",
           url if public else "{} is private or unreachable -- make it public when you submit"
           .format(url or "the remote"))

    # has the organisers' repository changed since this one last took it in?
    code, out = git("-c", "credential.helper=", "ls-remote", ORGANISERS, "HEAD", env=anon, timeout=60)
    if code != 0 or not out:
        report("SKIP", "organisers' repository checked for new commits", "could not reach it")
    else:
        theirs = out.split()[0]
        code, _ = git("merge-base", "--is-ancestor", theirs, "HEAD")
        report("PASS" if code == 0 else "WARN", "nothing new in the organisers' repository",
               "their head {} is part of this repository".format(theirs[:10]) if code == 0 else
               "their head {} is not in this repository: read what changed".format(theirs[:10]))
    return url, pushed


def _force_remove(func, path, _exc):
    os.chmod(path, stat.S_IWRITE)      # git's pack files are read-only on Windows
    func(path)


def check_clone(url, pushed):
    """What the organisers will do: clone the remote, install nothing new, run, compare."""
    item = "a fresh clone of the remote regenerates the same file"
    if not url or not pushed:
        report("SKIP", item, "needs this commit pushed to the remote's default branch first")
        return
    tmp = tempfile.mkdtemp(prefix="mtah_clone_")
    try:
        repo = os.path.join(tmp, "repo")
        code, _, se = run(["git", "clone", "--quiet", "--depth", "1", url, repo], cwd=tmp,
                          timeout=600)
        if code != 0:
            report("FAIL", item, "git clone failed: " + (se.strip().splitlines() or ["?"])[-1][:120])
            return
        code, so, se = run([sys.executable, "-W", "ignore", "-m", "solution.predict", "--name",
                            "clone_check"], cwd=repo,
                           env={"MTAH_DATA_DIR": os.path.abspath(DATA_DIR)})
        out = os.path.join(repo, "submissions", "clone_check.csv")
        if code != 0 or not os.path.isfile(out):
            log = os.path.join(OUT_DIR, "clone_check_predict.log")
            with open(log, "w", encoding="utf-8") as fh:
                fh.write(so + "\n--- stderr ---\n" + se)
            report("FAIL", item, "exit code {}; output kept in {}".format(
                code, os.path.relpath(log, ROOT)))
            return
        got, want = sha256(out), recorded_sha()
        report("PASS" if got == want else "FAIL", item,
               "committed files only, no cache" if got == want else
               "got {} expected {}".format(got, want))
    finally:
        shutil.rmtree(tmp, onexc=_force_remove) if sys.version_info >= (3, 12) else \
            shutil.rmtree(tmp, ignore_errors=True)


def check_document(name, sha):
    text = open(DOC, encoding="utf-8").read()
    heads, fenced = [], False
    for ln in text.splitlines():
        if ln.lstrip().startswith("```"):
            fenced = not fenced
        elif ln.startswith("#") and not fenced:
            heads.append(ln.lower())
    missing = [k for k, word in DOC_SECTIONS.items() if not any(word in h for h in heads)]
    report("PASS" if not missing else "FAIL",
           "technical document has a section for each of the six required topics",
           "missing: " + ", ".join(missing) if missing else "")
    if sha:
        report("PASS" if sha in text else "FAIL", "technical document quotes this file's SHA-256")
    os.makedirs(OUT_DIR, exist_ok=True)
    md = os.path.join(OUT_DIR, name + ".md")
    shutil.copyfile(DOC, md)
    pdf = os.path.join(OUT_DIR, name + ".pdf")
    if not os.path.isfile(pdf):
        report("PASS", "technical document copied to submissions/{}.md".format(name),
               "no {}.pdf there: upload the .md, or make the PDF from it".format(name))
        return md
    stale = os.path.getmtime(pdf) < os.path.getmtime(DOC)
    report("WARN" if stale else "PASS", "submissions/{}.pdf is newer than the document".format(name),
           "the PDF is older than solution/README.md: rebuild it" if stale else "")
    return pdf


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True, help="team name exactly as registered")
    ap.add_argument("--keep-cache", action="store_true",
                    help="reuse data/cache instead of rebuilding it from the raw files")
    ap.add_argument("--no-clone", action="store_true", help="skip the fresh-clone regeneration")
    ap.add_argument("--skip-git", action="store_true", help="skip every repository check")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    name = a.name
    if (BAD_NAME.search(name) or name != name.strip() or name[:1] in ("-", ".") or name.endswith(".")
            or name.lower().endswith((".csv", ".md", ".pdf")) or not name):
        sys.exit("the team name cannot be used as a file name as it is: {!r} (no < > : \" / \\ | ? * "
                 "and no leading '-' or '.', trailing '.' or file extension)".format(name))
    check_environment()
    path = regenerate(name, a.keep_cache)
    sha = None
    if path:
        check_validator(path)
        check_bytes(path)
        sha = check_sha(path)
    url = None
    if a.skip_git:
        report("SKIP", "repository checks", "--skip-git")
    else:
        url, pushed = check_repo()
        if a.no_clone:
            report("SKIP", "a fresh clone of the remote regenerates the same file", "--no-clone")
        elif url is not None:
            check_clone(url, pushed)
    doc = check_document(name, sha)

    fails = [i for s, i in RESULTS if s == "FAIL"]
    open_items = [i for s, i in RESULTS if s in ("TODO", "SKIP", "WARN")]
    print()
    if fails:
        print("NOT READY: {} check(s) failed.".format(len(fails)))
        sys.exit(1)
    print("READY{}".format(", except for: " + "; ".join(open_items) if open_items else "."))
    print("Upload:")
    for f in (path, doc):
        if f:
            print("  {}  ({:,} bytes, sha256 {})".format(f, os.path.getsize(f), sha256(f)[:16]))
    print("Repository URL for the form: {}".format(url or "(not checked)"))
    print("Team name for the form: {}".format(name))


if __name__ == "__main__":
    main()
