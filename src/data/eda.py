"""EDA toàn bộ ASL Citizen: bảng metadata (83k dòng) + zip index (kích thước file) + mẫu video phân tầng.

Không cần tải video để làm phần A. Phần B chỉ tải một mẫu nhỏ phân tầng theo signer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import asl_citizen as A

SPLITS = ("train", "val", "test")


# ----------------------------------------------------------------------------- load
def load_full_tables(P, zip_src=None):
    """Ưu tiên cache (manifests của notebook 01 hoặc full_eda_cache); thiếu thì đọc từ zip bằng HTTP Range
    (chỉ mục + vài CSV, vài MB). Trả về (meta, idx, nguồn)."""
    zip_src = zip_src or A.DEFAULT_CFG["zip_src"]
    cache = P.metrics / "full_eda_cache"
    cache.mkdir(parents=True, exist_ok=True)

    def find(name):
        for d in (P.manifests, cache):
            if (d / name).exists():
                return d / name

    mp, ip = find("meta_all.csv"), find("zip_index.csv")
    if mp and ip:
        return pd.read_csv(mp), pd.read_csv(ip), f"cache: {mp.parent}"
    A.preflight(zip_src, A.DEFAULT_CFG["expected_zip_bytes"])
    zf = A.open_zip(zip_src)
    idx = A.index_zip(zf)
    A.extract_small_files(zf, idx, cache / "metadata")
    meta = A.load_metadata(cache / "metadata", idx)
    meta.to_csv(cache / "meta_all.csv", index=False)
    idx.to_csv(cache / "zip_index.csv", index=False)
    return meta, idx, "remote (HTTP Range)"


def enrich(meta: pd.DataFrame, idx: pd.DataFrame) -> pd.DataFrame:
    m = meta.merge(idx[["name", "file_size"]], left_on="zip_name", right_on="name", how="left").drop(columns="name")
    m["gloss_base"] = m["gloss"].str.replace(r"\d+$", "", regex=True)
    return m


# ----------------------------------------------------------------------------- tables
def gini(x) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if n and x.sum() else float("nan")


def split_overview(m: pd.DataFrame) -> pd.DataFrame:
    o = m.groupby("split").agg(videos=("video_id", "size"), glosses=("gloss", "nunique"),
                               signers=("signer_id", "nunique"), gb=("file_size", lambda s: s.sum() / 1e9))
    o["median_vid_per_gloss"] = m.groupby(["split", "gloss"]).size().groupby("split").median()
    o["median_vid_per_signer"] = m.groupby(["split", "signer_id"]).size().groupby("split").median()
    o = o.reindex(list(SPLITS))
    o.loc["ALL"] = [len(m), m.gloss.nunique(), m.signer_id.nunique(), m.file_size.sum() / 1e9,
                    m.groupby("gloss").size().median(), m.groupby("signer_id").size().median()]
    return o.round(2)


def gloss_table(m: pd.DataFrame) -> pd.DataFrame:
    t = A.gloss_split_counts(m)
    t = t.join(m.groupby("gloss").signer_id.nunique().rename("signers_total"))
    t = t.join(m[m.split == "train"].groupby("gloss").signer_id.nunique().rename("signers_train"))
    t["signers_train"] = t["signers_train"].fillna(0).astype(int)
    t = t.join(m.groupby("gloss").gloss_base.first())
    t["n_variants"] = t.groupby("gloss_base")["gloss_base"].transform("size")
    return t.sort_values("total", ascending=False)


def signer_table(m: pd.DataFrame) -> pd.DataFrame:
    t = m.groupby("signer_id").agg(split=("split", "first"), n_splits=("split", "nunique"),
                                   videos=("video_id", "size"), glosses=("gloss", "nunique"),
                                   mb_mean=("file_size", lambda s: s.mean() / 1e6))
    t["gloss_coverage"] = t.glosses / m.gloss.nunique()
    t["share"] = t.videos / len(m)
    t["repeats_per_gloss"] = t.videos / t.glosses
    return t.sort_values("videos", ascending=False).round(4)


def balance_stats(gt: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for s in (*SPLITS, "total"):
        v = gt[s]
        rows[s] = dict(min=v.min(), p5=v.quantile(.05), median=v.median(), p95=v.quantile(.95), max=v.max(),
                       gini=gini(v), pct_gloss_le10=float((v <= 10).mean() * 100))
    return pd.DataFrame(rows).T.round(3)


def coverage_curve(gt: pd.DataFrame, ks=(5, 10, 20, 50, 100, 200, 500, 1000, 2000, None)) -> pd.DataFrame:
    """Kịch bản TỐT NHẤT khi chọn K class: lấy K gloss có min(train,val,test) lớn nhất."""
    ranked = gt.assign(score=gt[["train", "val", "test"]].min(axis=1)).sort_values(["score", "total"], ascending=False)
    rows = []
    kk = []
    for k in ks:
        k = len(ranked) if k is None else min(k, len(ranked))
        if k not in kk:
            kk.append(k)
    for k in kk:
        r = ranked.head(k)
        rows.append(dict(K=k, min_train=r.train.min(), median_train=r.train.median(), min_val=r.val.min(),
                         min_test=r.test.min(), min_signers_train=r.signers_train.min(),
                         total_videos=int(r.total.sum())))
    return pd.DataFrame(rows)


def eligibility_grid(gt: pd.DataFrame, combos=((5, 2, 3), (8, 2, 6), (10, 3, 8), (12, 4, 10), (15, 4, 12), (20, 5, 15))):
    return pd.DataFrame([{"train>=": a, "val>=": b, "test>=": c,
                          "gloss_đạt": int(((gt.train >= a) & (gt.val >= b) & (gt.test >= c)).sum())}
                         for a, b, c in combos])


def pair_repeats(m: pd.DataFrame) -> pd.Series:
    """Phân bố số video cho mỗi cặp (signer, gloss)."""
    return m.groupby(["signer_id", "gloss"]).size().value_counts().sort_index().rename("pairs")


def lex_code_check(m: pd.DataFrame) -> dict:
    if "asl_lex_code" not in m.columns:
        return {"asl_lex_code": "không có cột"}
    per_gloss = m.groupby("gloss").asl_lex_code.nunique()
    per_code = m.groupby("asl_lex_code").gloss.nunique()
    return dict(codes=int(m.asl_lex_code.nunique()), gloss_with_multi_codes=int((per_gloss > 1).sum()),
                codes_with_multi_gloss=int((per_code > 1).sum()))


# ----------------------------------------------------------------------------- sample videos (phần B)
def stratified_sample(m: pd.DataFrame, n=1500, seed=42) -> pd.DataFrame:
    """Chia đều theo signer (mỗi signer tối đa n/#signer video, ngẫu nhiên theo seed)."""
    ok = m.dropna(subset=["zip_name", "file_size"])
    per = max(1, n // ok.signer_id.nunique())
    parts = [g.sample(min(per, len(g)), random_state=seed) for _, g in ok.groupby("signer_id")]
    s = pd.concat(parts).reset_index(drop=True)
    s["file_size"] = s["file_size"].astype(int)
    return s


def sample_stats(sample: pd.DataFrame, videos_dir, probe=None, probe_frames=8, cache_csv=None) -> pd.DataFrame:
    from tqdm.auto import tqdm
    cols = ["video_id", "ok", "fps", "n_frames", "width", "height", "duration",
            "motion_diff", "det_rate", "two_hand_rate", "motion_wrist"]
    cache = pd.read_csv(cache_csv).set_index("video_id") if cache_csv and Path(cache_csv).exists() else None
    rows = []
    for r in tqdm(sample.itertuples(), total=len(sample), desc="sample stats"):
        if cache is not None and r.video_id in cache.index:
            c = cache.loc[r.video_id]
            if probe is None or pd.notna(c.get("det_rate")) or not bool(c.get("ok")):
                rows.append({"video_id": r.video_id, **c[[x for x in cols[1:] if x in c.index]].to_dict()})
                continue
        p = Path(videos_dir) / Path(r.zip_name).name
        s = A.video_stats(p)
        pr = A.probe_video(p, probe, probe_frames) if s["ok"] else {}
        rows.append({"video_id": r.video_id, **s, **pr})
    st = pd.DataFrame(rows)
    if cache_csv:
        st.to_csv(cache_csv, index=False)
    return sample.merge(st, on="video_id", how="left")
