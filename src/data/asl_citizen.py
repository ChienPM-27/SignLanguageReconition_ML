"""ASL Citizen: tải subset qua HTTP Range, metadata, probe chất lượng, chọn tier class.

Nguyên tắc: idempotent + resume được. Chạy lại bất kỳ lúc nào chỉ làm phần còn thiếu.
Không bao giờ tải cả file ZIP (~46 GB): chỉ đọc central directory + các video của gloss được chọn.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x=None, **kw):
        return x if x is not None else _NoBar()

    class _NoBar:
        def update(self, *a): ...
        def close(self): ...

ZIP_URL = "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
EXPECTED_ZIP_BYTES = 45_924_134_223  # Content-Length đã kiểm tra ngày 2026-10-03
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/1/hand_landmarker.task")
VIDEO_EXT = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v"}
SPLIT_NAMES = {"train": "train", "val": "val", "valid": "val", "validation": "val",
               "dev": "val", "test": "test"}
_ALIASES = {
    "signer_id": ["participantid", "signerid", "signer", "participant", "user", "userid"],
    "video_file": ["videofile", "filename", "file", "video", "videoname"],
    "gloss": ["gloss", "sign", "label"],
    "asl_lex_code": ["asllexcode", "lexcode", "code"],
}

_env_zip = os.environ.get("ASL_CITIZEN_ZIP")  # đã có sẵn file zip local? đặt biến này
DEFAULT_CFG = dict(
    zip_src=_env_zip or ZIP_URL,
    expected_zip_bytes=None if _env_zip else EXPECTED_ZIP_BYTES,
    pool_glosses=100,            # số gloss ứng viên tải về (superset cho mọi tier)
    min_train=10, min_val=3, min_test=8,   # ngưỡng số video/gloss mỗi split
    max_download_gb=8.0,         # chặn cứng, tránh tải nhầm quá nhiều
    workers=8, seed=42,
    run_probe=True, probe_frames=16,
    tier_sizes=(5, 20, 50), min_det_rate=0.85, default_tier="core5",
)


# ----------------------------------------------------------------------------- paths
def find_repo_root(start=None) -> Path:
    p = Path(start or Path.cwd()).resolve()
    for d in [p, *p.parents]:
        if (d / "configs" / "config.yaml").exists():
            return d
    raise FileNotFoundError("Không thấy repo root (configs/config.yaml). Hãy cd vào thư mục repo đã clone.")


def resolve_paths(root) -> SimpleNamespace:
    """Kaggle: ghi vào /kaggle/working/asl_outputs. Local: ghi vào <repo>/data. Splits luôn vào <repo>/data/splits."""
    root = Path(root)
    kaggle = Path("/kaggle/working").exists()
    base = Path("/kaggle/working/asl_outputs") if kaggle else root / "data"
    out = Path("/kaggle/working/asl_outputs") if kaggle else root / "outputs"
    src = base
    if Path("/kaggle/input").exists():  # đã publish subset làm Kaggle Dataset và attach vào notebook?
        for d in sorted(Path("/kaggle/input").iterdir()):
            hit = [d / "_SUBSET_READY.json", *d.glob("*/_SUBSET_READY.json")]
            hit = [m for m in hit if m.exists()]
            if hit:
                src = hit[0].parent
                break
    P = SimpleNamespace(root=root, kaggle=kaggle, base=base, src=src, from_input=(src != base))
    P.raw, P.videos, P.meta = src / "raw", src / "raw" / "videos", src / "raw" / "metadata"
    P.manifests = src / "manifests"
    P.metrics, P.plots = out / "metrics", out / "plots"
    P.splits = root / "data" / "splits"
    if not P.from_input:
        for d in (P.videos, P.meta, P.manifests, P.metrics, P.plots):
            d.mkdir(parents=True, exist_ok=True)
    else:
        for d in (P.metrics, P.plots):
            d.mkdir(parents=True, exist_ok=True)
    P.splits.mkdir(parents=True, exist_ok=True)
    return P


# ----------------------------------------------------------------------------- zip access
def open_zip(src):
    if str(src).startswith(("http://", "https://")):
        try:
            from remotezip import RemoteZip
        except ImportError as e:  # pragma: no cover
            raise ImportError("Thiếu remotezip: pip install remotezip") from e
        return RemoteZip(src)
    return zipfile.ZipFile(src)


def preflight(src, expected_bytes=None) -> dict:
    """Kiểm tra mạng + server hỗ trợ Range + đúng phiên bản file zip. Fail sớm, rõ lý do."""
    if not str(src).startswith(("http://", "https://")):
        return dict(kind="local", bytes=Path(src).stat().st_size)
    import requests
    try:
        h = requests.head(src, allow_redirects=True, timeout=30)
        h.raise_for_status()
        r = requests.get(src, headers={"Range": "bytes=0-3"}, stream=True, timeout=30)
        range_ok = r.status_code == 206
        r.close()
    except requests.exceptions.RequestException as e:
        raise RuntimeError("Không kết nối được tới Microsoft. Trên Kaggle: Settings → Internet → ON "
                           f"(cần xác minh số điện thoại). Chi tiết: {e}") from e
    info = dict(kind="remote", bytes=int(h.headers.get("Content-Length", -1)),
                etag=h.headers.get("ETag"), last_modified=h.headers.get("Last-Modified"), range_ok=range_ok)
    if not range_ok:
        raise RuntimeError("Server không hỗ trợ HTTP Range → không tải từng phần được.")
    if expected_bytes and info["bytes"] != expected_bytes:
        raise RuntimeError(f"File zip đã đổi (size {info['bytes']} ≠ {expected_bytes}). Dataset có thể được cập nhật; "
                           "kiểm tra lại cấu trúc rồi sửa EXPECTED_ZIP_BYTES.")
    return info


def index_zip(zf) -> pd.DataFrame:
    rows = [dict(name=i.filename, file_size=i.file_size, compress_size=i.compress_size, crc=i.CRC)
            for i in zf.infolist()
            if not i.is_dir() and not i.filename.startswith("__MACOSX") and not Path(i.filename).name.startswith("._")]
    df = pd.DataFrame(rows)
    df["base"] = df["name"].str.rsplit("/", n=1).str[-1]
    df["ext"] = df["base"].str.extract(r"(\.[^.]+)$")[0].str.lower()
    return df


def extract_small_files(zf, idx: pd.DataFrame, dest: Path, exts=(".csv", ".md", ".txt"), max_bytes=20_000_000):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for r in idx[idx.ext.isin(exts) & (idx.file_size <= max_bytes)].itertuples():
        p = dest / r.name
        if not (p.exists() and p.stat().st_size == r.file_size):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(zf.read(r.name))
        out.append(p)
    return out


# ----------------------------------------------------------------------------- metadata
def _norm(c: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def load_metadata(meta_dir, idx: pd.DataFrame) -> pd.DataFrame:
    """Gộp train/val/test CSV → bảng chuẩn: video_id, gloss, split, signer_id, video_file, zip_name."""
    found = {}
    for p in sorted(Path(meta_dir).rglob("*.csv")):
        key = SPLIT_NAMES.get(p.stem.lower())
        if key:
            found.setdefault(key, []).append(p)
    missing = {"train", "val", "test"} - set(found)
    if missing:
        raise ValueError(f"Thiếu CSV split {sorted(missing)}. CSV tìm thấy: "
                         f"{[str(p.relative_to(meta_dir)) for p in Path(meta_dir).rglob('*.csv')]}")
    vids = idx[idx.ext.isin(VIDEO_EXT)]
    by_base = vids.drop_duplicates("base").set_index("base")["name"].to_dict()
    by_stem = {Path(b).stem: n for b, n in by_base.items()}
    frames = []
    for split, paths in found.items():
        p = next((q for q in paths if "split" in str(q).lower()), paths[0])
        df = pd.read_csv(p)
        cmap = {}
        for std, aliases in _ALIASES.items():
            for c in df.columns:
                if _norm(c) in aliases and std not in cmap.values():
                    cmap[c] = std
                    break
        df = df.rename(columns=cmap)
        for need in ("signer_id", "video_file", "gloss"):
            if need not in df.columns:
                raise ValueError(f"{p.name}: không map được cột '{need}'. Cột hiện có: {list(df.columns)}")
        df["split"] = split
        frames.append(df)
    meta = pd.concat(frames, ignore_index=True)
    meta["gloss"] = meta["gloss"].astype(str).str.strip()
    meta["signer_id"] = meta["signer_id"].astype(str)
    meta["video_file"] = meta["video_file"].astype(str).map(lambda s: Path(s).name)
    meta["zip_name"] = meta["video_file"].map(by_base)
    miss = meta["zip_name"].isna()
    meta.loc[miss, "zip_name"] = meta.loc[miss, "video_file"].map(lambda s: by_stem.get(Path(s).stem))
    meta["video_id"] = meta["video_file"].map(lambda s: Path(s).stem)
    return meta


def gloss_split_counts(meta: pd.DataFrame) -> pd.DataFrame:
    c = meta.pivot_table(index="gloss", columns="split", values="video_id", aggfunc="count", fill_value=0)
    for s in ("train", "val", "test"):
        if s not in c:
            c[s] = 0
    c["total"] = c[["train", "val", "test"]].sum(axis=1)
    return c[["train", "val", "test", "total"]]


def signer_overlap(meta: pd.DataFrame) -> dict:
    s = {k: set(v) for k, v in meta.groupby("split")["signer_id"]}
    return {f"{a}&{b}": sorted(s[a] & s[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
            if a in s and b in s}


def pick_pool(counts: pd.DataFrame, k: int, mins=(10, 3, 8), seed=42) -> list[str]:
    el = counts[(counts.train >= mins[0]) & (counts.val >= mins[1]) & (counts.test >= mins[2])]
    if len(el) < k:
        print(f"⚠ Chỉ có {len(el)} gloss đạt ngưỡng (< pool_glosses={k}); dùng tất cả.")
        k = len(el)
    return sorted(el.sample(k, random_state=seed).index.tolist())


# ----------------------------------------------------------------------------- download
_tls = threading.local()


def _thread_zip(src):
    if getattr(_tls, "zf", None) is None:
        _tls.zf = open_zip(src)
    return _tls.zf


def _drop_thread_zip():
    try:
        if getattr(_tls, "zf", None) is not None:
            _tls.zf.close()
    except Exception:
        pass
    _tls.zf = None


def download_videos(src, pool_df: pd.DataFrame, out_dir, workers=8, retries=4, log_path=None) -> pd.DataFrame:
    """Tải đúng các video trong pool_df (cột zip_name, file_size). Resume: bỏ qua file đã đủ size.
    Ghi atomic (.part → rename). zipfile tự kiểm CRC-32 khi đọc hết file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def job(row):
        dst = out_dir / Path(row.zip_name).name
        if dst.exists() and dst.stat().st_size == row.file_size:
            return dict(video_file=dst.name, status="cached", bytes=row.file_size, attempts=0, error="")
        err = ""
        for a in range(1, retries + 1):
            try:
                data = _thread_zip(src).read(row.zip_name)
                if len(data) != row.file_size:
                    raise IOError(f"size {len(data)} ≠ {row.file_size}")
                tmp = dst.with_suffix(dst.suffix + ".part")
                tmp.write_bytes(data)
                os.replace(tmp, dst)
                return dict(video_file=dst.name, status="downloaded", bytes=len(data), attempts=a, error="")
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
                _drop_thread_zip()
                time.sleep(min(2 ** a, 20))
        return dict(video_file=dst.name, status="failed", bytes=0, attempts=retries, error=err)

    rows = list(pool_df.itertuples())
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(job, r) for r in rows]
        for f in tqdm(as_completed(futs), total=len(futs), desc="download"):
            results.append(f.result())
    log = pd.DataFrame(results)
    if log_path:
        log.to_csv(log_path, index=False)
    return log


# ----------------------------------------------------------------------------- video stats / probe
def video_stats(path) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    ok_open = cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS) if ok_open else 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if ok_open else 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if ok_open else 0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if ok_open else 0
    ok_read, _ = cap.read() if ok_open else (False, None)
    cap.release()
    ok = bool(ok_open and ok_read and n > 0 and fps > 0)
    return dict(ok=ok, fps=round(fps, 3), n_frames=n, width=w, height=h,
                duration=round(n / fps, 3) if fps > 0 else 0.0)


class _HandProbe:
    """Bọc MediaPipe: legacy `mp.solutions.hands` nếu còn, nếu không dùng Tasks HandLandmarker (IMAGE mode)."""

    def __init__(self, model_dir):
        import mediapipe as mp
        self.mp = mp
        self.legacy = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
        if self.legacy:
            self.det = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                                min_detection_confidence=0.5)
        else:
            from mediapipe.tasks.python import BaseOptions, vision
            model = Path(model_dir) / "hand_landmarker.task"
            if not model.exists() or model.stat().st_size < 1_000_000:
                model.parent.mkdir(parents=True, exist_ok=True)
                tmp, last = model.with_suffix(".part"), None
                for a in range(3):
                    try:
                        urllib.request.urlretrieve(HAND_MODEL_URL, tmp)
                        if tmp.stat().st_size < 1_000_000:
                            raise IOError(f"file model quá nhỏ ({tmp.stat().st_size} byte)")
                        os.replace(tmp, model)
                        break
                    except Exception as e:  # noqa: BLE001
                        last = e
                        time.sleep(2 * (a + 1))
                else:
                    raise RuntimeError(f"Không tải được model hand_landmarker.task ({last}). Tải tay từ {HAND_MODEL_URL} "
                                       f"rồi đặt vào {model}")
            opts = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model)),
                running_mode=vision.RunningMode.IMAGE, num_hands=2,
                min_hand_detection_confidence=0.5)
            self.det = vision.HandLandmarker.create_from_options(opts)

    def hands(self, rgb):
        """→ list[np.ndarray(21,2)] toạ độ chuẩn hoá [0,1]."""
        if self.legacy:
            res = self.det.process(rgb)
            return [np.array([[p.x, p.y] for p in h.landmark]) for h in (res.multi_hand_landmarks or [])]
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self.det.detect(img)
        return [np.array([[p.x, p.y] for p in h]) for h in res.hand_landmarks]


def make_probe(model_dir):
    try:
        return _HandProbe(model_dir)
    except Exception as e:  # noqa: BLE001
        hint = " → chạy: pip install mediapipe" if isinstance(e, ModuleNotFoundError) else ""
        print(f"⚠ Bỏ qua MediaPipe probe ({type(e).__name__}: {e}){hint}. Chọn tier sẽ chỉ dùng motion_diff.")
        return None


def probe_video(path, probe, n=16) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 0
        while cap.read()[0]:
            total += 1
        cap.release()
        cap = cv2.VideoCapture(str(path))
    want = set(np.linspace(0, max(total - 1, 0), n).astype(int).tolist())
    grays, wrists, sizes, n_det, n_two, n_s = [], [], [], 0, 0, 0
    for i in range(total):
        ok, f = cap.read()
        if not ok:
            break
        if i not in want:
            continue
        n_s += 1
        grays.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32))
        if probe is not None:
            hs = probe.hands(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            h, w = f.shape[:2]
            if hs:
                n_det += 1
                n_two += int(len(hs) >= 2)
                wrists.append(np.mean([[x[0, 0] * w, x[0, 1] * h] for x in hs], axis=0))
                sizes.append(np.mean([np.hypot((x[9, 0] - x[0, 0]) * w, (x[9, 1] - x[0, 1]) * h) for x in hs]))
            else:
                wrists.append(None)
    cap.release()
    diff = float(np.mean([np.abs(a - b).mean() for a, b in zip(grays[:-1], grays[1:])])) if len(grays) > 1 else np.nan
    out = dict(motion_diff=round(diff, 4), det_rate=np.nan, two_hand_rate=np.nan, motion_wrist=np.nan)
    if probe is not None and n_s:
        out["det_rate"] = round(n_det / n_s, 4)
        out["two_hand_rate"] = round(n_two / n_s, 4)
        pts = [w for w in wrists if w is not None]
        if len(pts) > 1 and sizes:
            path_len = sum(float(np.linalg.norm(b - a)) for a, b in zip(pts[:-1], pts[1:]))
            out["motion_wrist"] = round(path_len / max(float(np.mean(sizes)), 1e-6), 4)  # đơn vị: "độ dài bàn tay"
    return out


# ----------------------------------------------------------------------------- build (heavy, idempotent)
def cfg_hash(cfg: dict) -> str:
    keys = ["pool_glosses", "min_train", "min_val", "min_test", "seed"]
    return hashlib.sha1(json.dumps({k: cfg[k] for k in keys}, default=str, sort_keys=True).encode()).hexdigest()[:12]


def read_ready(P):
    f = P.src / "_SUBSET_READY.json"
    return json.loads(f.read_text()) if f.exists() else None


def build_subset(cfg: dict, P) -> pd.DataFrame:
    """Tải metadata + video của pool gloss, tính stats/probe, ghi manifests/manifest_pool.csv.
    Nếu đã có _SUBSET_READY.json khớp cấu hình → trả về ngay, không cần mạng."""
    cfg = {**DEFAULT_CFG, **cfg}
    h = cfg_hash(cfg)
    ready = read_ready(P)
    pool_csv = P.manifests / "manifest_pool.csv"
    if ready and ready.get("cfg_hash") == h and pool_csv.exists():
        print(f"✅ Subset đã sẵn sàng ({ready['n_videos']} video, {ready['n_glosses']} gloss) → bỏ qua tải.")
        return pd.read_csv(pool_csv)
    if P.from_input:
        raise RuntimeError("Dataset input trên Kaggle không khớp cấu hình hiện tại (cfg_hash). "
                           "Sửa CFG về đúng cấu hình đã build hoặc build lại ở working dir.")

    t0 = time.time()
    pf = preflight(cfg["zip_src"], cfg["expected_zip_bytes"])
    print(f"[1/6] Preflight OK: {pf}")
    zf = open_zip(cfg["zip_src"])
    idx = index_zip(zf)
    idx.to_csv(P.manifests / "zip_index.csv", index=False)
    print(f"[2/6] Zip index: {len(idx):,} entries, {int(idx.ext.isin(VIDEO_EXT).sum()):,} video "
          f"(đọc central directory, chưa tải video)")
    extract_small_files(zf, idx, P.meta)
    meta = load_metadata(P.meta, idx)
    meta.to_csv(P.manifests / "meta_all.csv", index=False)
    n_unmatched = int(meta.zip_name.isna().sum())
    print(f"[3/6] Metadata: {len(meta):,} dòng, {meta.gloss.nunique():,} gloss, {meta.signer_id.nunique()} signer; "
          f"không khớp video trong zip: {n_unmatched}")
    ov = signer_overlap(meta)
    if any(ov.values()):
        print(f"⚠ Signer trùng giữa các split: { {k: len(v) for k, v in ov.items() if v} }")

    counts = gloss_split_counts(meta)
    pool = pick_pool(counts, cfg["pool_glosses"], (cfg["min_train"], cfg["min_val"], cfg["min_test"]), cfg["seed"])
    pdf = meta[meta.gloss.isin(pool)].dropna(subset=["zip_name"]).merge(
        idx[["name", "file_size"]], left_on="zip_name", right_on="name").drop(columns="name")
    gb = pdf.file_size.sum() / 1e9
    print(f"[4/6] Pool: {len(pool)} gloss, {len(pdf):,} video, ≈{gb:.2f} GB (giới hạn {cfg['max_download_gb']} GB)")
    if gb > cfg["max_download_gb"]:
        raise RuntimeError(f"Kế hoạch tải {gb:.1f} GB vượt max_download_gb. Giảm pool_glosses.")

    log = download_videos(cfg["zip_src"], pdf, P.videos, cfg["workers"], log_path=P.manifests / "download_log.csv")
    fails = log[log.status == "failed"]
    print(f"[5/6] Tải xong: { {k: int(v) for k, v in log.status.value_counts().items()} }")
    if len(fails):
        raise RuntimeError(f"{len(fails)} video lỗi (xem manifests/download_log.csv). Chạy lại cell này để thử lại phần lỗi.")

    stat_cols = ["ok", "fps", "n_frames", "width", "height", "duration",
                 "motion_diff", "det_rate", "two_hand_rate", "motion_wrist"]
    cache = pd.read_csv(pool_csv).set_index("video_id")[stat_cols] if pool_csv.exists() else None
    probe = make_probe(P.src / "models") if cfg["run_probe"] else None
    rows = []
    for r in tqdm(pdf.itertuples(), total=len(pdf), desc="stats+probe"):
        if cache is not None and r.video_id in cache.index:
            c = cache.loc[r.video_id]
            if pd.notna(c["ok"]) and (probe is None or pd.notna(c["det_rate"]) or not bool(c["ok"])):
                rows.append({"video_id": r.video_id, **c.to_dict()})
                continue
        p = P.videos / Path(r.zip_name).name
        s_ = video_stats(p)
        pr = probe_video(p, probe, cfg["probe_frames"]) if s_["ok"] else {}
        rows.append({"video_id": r.video_id, **s_, **pr})
    stats = pd.DataFrame(rows)
    pool_df = pdf.merge(stats, on="video_id", how="left")
    pool_df.to_csv(pool_csv, index=False)
    info = dict(cfg_hash=h, n_videos=int(len(pool_df)), n_glosses=len(pool), zip=pf,
                bad_videos=int((~pool_df.ok.astype(bool)).sum()), pool=pool,
                built_at=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - t0))
    (P.src / "_SUBSET_READY.json").write_text(json.dumps(info, indent=2, default=str))
    print(f"[6/6] Xong trong {info['seconds']}s → {pool_csv}")
    return pool_df


# ----------------------------------------------------------------------------- tiers + splits
def gloss_quality(pool_df: pd.DataFrame) -> pd.DataFrame:
    ok = pool_df[pool_df.ok.astype(bool)]
    g = ok.groupby("gloss").agg(n=("video_id", "size"), det_rate=("det_rate", "mean"),
                                two_hand_rate=("two_hand_rate", "mean"),
                                motion_wrist=("motion_wrist", "median"), motion_diff=("motion_diff", "median"),
                                duration=("duration", "median"))
    g["motion"] = g["motion_wrist"].fillna(g["motion_diff"])
    return g.sort_values("motion")


def select_tiers(gq: pd.DataFrame, sizes=(5, 20, 50), min_det=0.85, seed=42, override: dict | None = None) -> dict:
    """Tier lồng nhau: core5 ⊂ core20 ⊂ core50. core5 = ít chuyển động nhất (trong nhóm tay phát hiện tốt);
    phần mở rộng lấy ngẫu nhiên có seed để phản ánh độ khó thật."""
    override = override or {}
    unknown = {g for v in override.values() for g in v} - set(gq.index)
    if unknown:
        raise ValueError(f"OVERRIDE chứa gloss không có trong pool: {sorted(unknown)}")
    ok = gq[(gq.det_rate >= min_det) | gq.det_rate.isna()]
    if len(ok) < max(sizes):
        print(f"⚠ Chỉ {len(ok)} gloss đạt det_rate≥{min_det}; nới ngưỡng cho phần mở rộng.")
        ok = gq
    tiers, chosen = {}, []
    rng = np.random.default_rng(seed)
    for s in sorted(sizes):
        name = f"core{s}"
        if name in override:
            chosen = list(override[name])
        else:
            need = s - len(chosen)
            if not chosen:
                chosen = ok.index[:s].tolist()                   # đã sort theo motion tăng dần
            else:
                rest = [g for g in ok.index if g not in chosen]
                chosen = chosen + rng.permutation(rest)[:need].tolist()
        if len(chosen) < s:
            raise ValueError(f"Không đủ gloss cho {name}: {len(chosen)}/{s}")
        tiers[name] = sorted(chosen)
    return tiers


def contact_sheet(pool_df, glosses, videos_dir, per_gloss=3, seed=0):
    import cv2
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(glosses), per_gloss, figsize=(3 * per_gloss, 2.4 * len(glosses)), squeeze=False)
    for i, g in enumerate(glosses):
        sub = pool_df[(pool_df.gloss == g) & pool_df.ok.astype(bool)]
        pick = sub.iloc[rng.permutation(len(sub))[:per_gloss]]
        for j in range(per_gloss):
            ax = axes[i][j]
            ax.axis("off")
            if j < len(pick):
                cap = cv2.VideoCapture(str(Path(videos_dir) / Path(pick.iloc[j].zip_name).name))
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2, 0))
                ok, f = cap.read()
                cap.release()
                if ok:
                    ax.imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            if j == 0:
                ax.set_title(g, loc="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def write_splits(pool_df, tiers: dict, splits_dir, default_tier="core5") -> dict:
    splits_dir = Path(splits_dir)
    (splits_dir / "tiers").mkdir(parents=True, exist_ok=True)
    report = {}
    for name, glosses in tiers.items():
        sub = pool_df[pool_df.gloss.isin(glosses) & pool_df.ok.astype(bool)].copy()
        lm = {g: i for i, g in enumerate(sorted(glosses))}
        sub["label"] = sub.gloss.map(lm)
        sub["video_file"] = sub.zip_name.map(lambda s: Path(s).name)
        cols = ["video_id", "gloss", "split", "signer_id", "video_file", "label"]
        assert sub.video_id.is_unique, f"{name}: video_id trùng"
        ov = signer_overlap(sub)
        assert not any(ov.values()), f"{name}: signer rò rỉ giữa split {ov}"
        per = sub.groupby(["gloss", "split"]).size().unstack(fill_value=0).reindex(columns=["train", "val", "test"], fill_value=0)
        assert (per.values > 0).all(), f"{name}: có class thiếu mẫu ở 1 split:\n{per[(per == 0).any(axis=1)]}"
        for s in ("train", "val", "test"):
            part = sub[sub.split == s][cols].sort_values("video_id")
            part.to_csv(splits_dir / "tiers" / f"{name}_{s}.csv", index=False)
            if name == default_tier:
                part.to_csv(splits_dir / f"{s}.csv", index=False)
        (splits_dir / "tiers" / f"{name}_label_map.json").write_text(json.dumps(lm, indent=2))
        report[name] = dict(n_classes=len(glosses), n_videos=len(sub), per_class_min=per.min().to_dict())
    return report
