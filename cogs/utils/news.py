"""`/news`: the release notes in `changelog/`, from Beta v0.7.0 on -- the first written to
fit one embed. Read once: the folder changes only with a deploy, which restarts the process.
"""

import functools
import os
import re
from typing import Dict, Optional

CHANGELOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             "changelog")
FIRST_RELEASE = (0, 7, 0)
_RELEASE_FILE = re.compile(r"v(\d+)\.(\d+)\.(\d+)\.txt")


@functools.lru_cache(maxsize=1)
def releases() -> Dict[str, Dict[str, str]]:
    """{"Beta v0.7.0": {its first line: the rest}}, newest first -- the shape
    DropdownContentView takes. Compared as numbers, so v0.10 follows v0.9."""
    try:
        names = os.listdir(CHANGELOG_DIR)
    except OSError:
        return {}
    found = sorted(((tuple(map(int, m.groups())), name) for name in names
                    if (m := _RELEASE_FILE.fullmatch(name))), reverse=True)
    out: Dict[str, Dict[str, str]] = {}
    for version, name in (v for v in found if v[0] >= FIRST_RELEASE):
        if len(out) == 25:  # one dropdown's worth
            break
        with open(os.path.join(CHANGELOG_DIR, name), encoding="utf-8") as f:
            title, _, body = f.read().strip().partition("\n")
        out["Beta v" + ".".join(map(str, version))] = {title.strip()[:100]: body.strip()[:4096]}
    return out


def current_version() -> Optional[str]:
    """"Beta v0.7.0", or None where no release notes shipped."""
    return next(iter(releases()), None)


def latest_release_text() -> Optional[str]:
    """The newest release's notes as one document, for `/help`'s shards."""
    version = current_version()
    if version is None:
        return None
    (title, body), = releases()[version].items()
    return f"{title}\n{body}"

