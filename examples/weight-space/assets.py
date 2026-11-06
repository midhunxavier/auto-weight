"""Fetch a single published artifact by revision, exact size, and SHA-256."""
import hashlib
import json
import ssl
from pathlib import Path
from urllib.request import urlopen

MANIFEST = json.loads(Path(__file__).with_name("assets.json").read_text())
UPSTREAM = "https://raw.githubusercontent.com/RikiyaT/lm_understand-logs"


def asset(name):
    reference = MANIFEST["files"][name]
    url = f"{UPSTREAM}/{MANIFEST['revision']}/{name}"
    trust = ssl.create_default_context()
    try:
        from certifi import where
    except ImportError:
        pass
    else:
        trust.load_verify_locations(cafile=where())
    with urlopen(url, timeout=45, context=trust) as source:
        data = source.read(reference["bytes"] + 1)
    if len(data) != reference["bytes"] or hashlib.sha256(data).hexdigest() != reference["sha256"]:
        raise ValueError(f"published asset verification failed: {name}")
    return data
