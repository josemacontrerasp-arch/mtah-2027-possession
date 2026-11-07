"""Download the external training data (DFL open dataset, CC BY 4.0) -- only needed to retrain.

    python -m solution.download_external                 # -> ../external/dfl  (~2.6 GB)
    MTAH_EXTERNAL_DIR=/somewhere python -m solution.download_external

Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). An integrated dataset of spatiotemporal and
event data in elite soccer. Scientific Data 12, 195. doi:10.6084/m9.figshare.28196177 (CC BY 4.0)

Files are fetched from figshare's public API and checked against the sizes it reports; an
interrupted download resumes. Producing the submission does NOT need any of this: the trained
models are committed in solution/models/.
"""

import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from solution.external_dfl import DFL_DIR  # noqa: E402

ARTICLE = "https://api.figshare.com/v2/articles/28196177"
WANT = ("DFL_02_01_matchinformation", "DFL_04_03_positions_raw_observed")


def main():
    os.makedirs(DFL_DIR, exist_ok=True)
    meta = json.load(urllib.request.urlopen(ARTICLE))
    print("{} -- licence {}".format(meta["title"], meta["license"]["name"]))
    for f in meta["files"]:
        if not f["name"].startswith(WANT):
            continue
        path = os.path.join(DFL_DIR, f["name"])
        have = os.path.getsize(path) if os.path.isfile(path) else 0
        if have == f["size"]:
            print("ok      ", f["name"])
            continue
        req = urllib.request.Request(f["download_url"], headers={"Range": "bytes={}-".format(have)})
        with urllib.request.urlopen(req) as r, open(path, "ab" if have else "wb") as out:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        size = os.path.getsize(path)
        print("{}  {}".format("ok      " if size == f["size"] else "MISMATCH", f["name"]))
        if size != f["size"]:
            sys.exit("size mismatch for {}: {} != {}".format(f["name"], size, f["size"]))


if __name__ == "__main__":
    main()
