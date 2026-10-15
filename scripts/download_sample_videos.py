"""Download sample MP4 demonstration videos for the 20 target ISLR glosses.

Saves clips to data/sample_videos/<gloss>.mp4.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time
import urllib.request

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging import get_logger

logger = get_logger("islr.sample_videos")

SAMPLE_URLS = {
    "awake": "https://www.handspeak.com/word/a/awa/awake.mp4",
    "bird": "https://www.handspeak.com/word/b/bir/bird.mp4",
    "brown": "https://www.handspeak.com/word/b/bro/brown.mp4",
    "bye": "https://www.handspeak.com/word/g/goo/goodbye.mp4",
    "cow": "https://www.handspeak.com/word/c/cow/cow.mp4",
    "doll": "https://www.handspeak.com/word/d/dol/doll.mp4",
    "donkey": "https://www.handspeak.com/word/d/don/donkey.mp4",
    "drink": "https://www.handspeak.com/word/d/dri/drink.mp4",
    "duck": "https://www.handspeak.com/word/d/duc/duck.mp4",
    "fireman": "https://www.handspeak.com/word/f/fir/firefighter.mp4",
    "lips": "https://www.handspeak.com/word/l/lip/lip.mp4",
    "listen": "https://www.handspeak.com/word/l/lis/listen.mp4",
    "mouse": "https://www.handspeak.com/word/m/mou/mouse.mp4",
    "nuts": "https://www.handspeak.com/word/p/pea/peanut.mp4",
    "pretend": "https://www.handspeak.com/word/p/pre/pretend.mp4",
    "shhh": "https://www.handspeak.com/word/q/qui/quiet.mp4",
    "sleepy": "https://www.handspeak.com/word/s/sle/sleepy.mp4",
    "uncle": "https://www.handspeak.com/word/u/unc/uncle.mp4",
    "wake": "https://www.handspeak.com/word/w/wak/wake-up.mp4",
    "who": "https://www.handspeak.com/word/w/who/who.mp4",
}


def download_sample_videos(dest_dir: Path | str = "data/sample_videos") -> dict[str, Path]:
    target_dir = Path(dest_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    downloaded = {}
    total = len(SAMPLE_URLS)
    logger.info("Starting download of %d sample videos to %s...", total, target_dir)

    for i, (gloss, url) in enumerate(SAMPLE_URLS.items(), start=1):
        file_path = target_dir / f"{gloss}.mp4"
        if file_path.is_file() and file_path.stat().st_size > 1024:
            logger.info("[%d/%d] Existing: %s (%s)", i, total, gloss, file_path.name)
            downloaded[gloss] = file_path
            continue

        logger.info("[%d/%d] Downloading %s from %s...", i, total, gloss, url)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp, open(file_path, "wb") as f_out:
                f_out.write(resp.read())
            logger.info("Saved %s (%.1f KB)", file_path.name, file_path.stat().st_size / 1024)
            downloaded[gloss] = file_path
            time.sleep(0.3)
        except Exception as e:
            logger.error("Failed downloading %s: %s", gloss, e)

    logger.info("Download completed: %d / %d videos available.", len(downloaded), total)
    return downloaded


def main() -> int:
    download_sample_videos()
    return 0


if __name__ == "__main__":
    sys.exit(main())
