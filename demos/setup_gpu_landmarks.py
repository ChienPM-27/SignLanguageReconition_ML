"""Install the pinned official MediaPipe Web runtime used by local GPU extraction."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.10.35"
SDK_ROOT = ROOT / "models/mediapipe-web" / VERSION
ARCHIVE = ROOT / f"tmp/mediapipe-web/mediapipe-tasks-vision-{VERSION}.tgz"
INTEGRITY = "HOvadwVRE6JC+45nyYhmnywnr5h/J8KZvOeUNVOG9q/0875pZgItznFB9bRTvLc264YSJqiZ1NsIpCStJw/egg=="
FILES = ("vision_bundle.mjs", "wasm/vision_wasm_internal.js", "wasm/vision_wasm_internal.wasm",
         "wasm/vision_wasm_module_internal.js", "wasm/vision_wasm_module_internal.wasm",
         "wasm/vision_wasm_nosimd_internal.js", "wasm/vision_wasm_nosimd_internal.wasm")


def ensure_sdk() -> Path:
    manifest = SDK_ROOT / "MANIFEST.json"
    if manifest.is_file():
        hashes = json.loads(manifest.read_text(encoding="utf-8"))
        if set(hashes) == set(FILES) and all((SDK_ROOT / name).is_file() and
                hashlib.sha256((SDK_ROOT / name).read_bytes()).hexdigest() == hashes[name] for name in FILES):
            return SDK_ROOT
    if not ARCHIVE.is_file():
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://registry.npmjs.org/@mediapipe/tasks-vision/-/tasks-vision-{VERSION}.tgz"
        urllib.request.urlretrieve(url, ARCHIVE)
    checksum = base64.b64encode(hashlib.sha512(ARCHIVE.read_bytes()).digest()).decode()
    if checksum != INTEGRITY:
        raise ValueError("MediaPipe Web archive does not match the pinned official npm integrity.")
    hashes = {}
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for name in FILES:
            member = archive.getmember("package/" + name)
            if not member.isfile():
                raise ValueError("MediaPipe runtime asset is not a regular file: " + name)
            with archive.extractfile(member) as stream:
                payload = stream.read()
            target = SDK_ROOT / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            hashes[name] = hashlib.sha256(payload).hexdigest()
    manifest.write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    return SDK_ROOT


def ensure_task_assets() -> dict[str, Path]:
    from src.features.video_auxiliary import _task_asset

    hand = ROOT / "models/hand_landmarker.task"
    if not hand.is_file():
        hand.parent.mkdir(parents=True, exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, hand)
    return {"hand_landmarker.task": hand, "face_landmarker.task": _task_asset("face"),
            "pose_landmarker_lite.task": _task_asset("pose")}


if __name__ == "__main__":
    print(ensure_sdk())
