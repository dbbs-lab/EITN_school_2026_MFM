"""
Download pre-compiled BSB reconstruction of cerebellar network
"""

import sys
from pathlib import Path

DRIVE_FILE_URL = "https://drive.google.com/file/d/12cRxjT5MTBNpklGmGwDPpoZS75-oT47G/view?usp=drive_link"
DEST_DIR = Path(__file__).resolve().parent


def download_reco() -> None:
    try:
        import gdown
    except ImportError:
        sys.exit("gdown is required: pip install gdown")

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    gdown.download(url=DRIVE_FILE_URL, output=str(DEST_DIR) + "/", quiet=False, fuzzy=True)

