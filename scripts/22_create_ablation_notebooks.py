#!/usr/bin/env python3
"""Create the three clean, deterministic Kaggle/Colab notebook entry points."""
from pathlib import Path
import hashlib
import json
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    source = textwrap.dedent(source).strip() + "\n"
    result = {"cell_type": kind, "id": hashlib.sha256((kind + source).encode()).hexdigest()[:12],
              "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build(member):
    cells = [cell("markdown", f"""
        # Ablation V2 - Member {member}

        Notebook cho **Kaggle GPU** hoặc **Colab GPU + Google Drive**.
        Member A: feature (8 configs / 24 lượt); B: architecture (30 / 90);
        C: GRU baseline + LSTM + Transformer (22 / 66). Mỗi config chạy đủ seeds42/43/44.

        **Kaggle:** bật Internet và GPU, attach dataset `asl-signs` đầy đủ.
        **Colab:** chọn GPU; Drive được mount khi tạo session để giữ kết quả.
        Config model/feature/training lấy từ YAML trong repo, không sửa hyperparameter ở notebook.

        A chạy `MODE="prepare"` với frozen split cũ để tạo shared ZIP giữ đúng vocabulary/signers.
        Sau đó attach shared ZIP V2 cho cả A/B/C và dùng `MODE="train"` trên 3 runtime riêng.
        B/C không phải đợi A train xong. Shared package/source khác version bị từ chối.
    """)]
    cells.append(cell("markdown", """
        ## 1. Thiết lập một lần

        `SHARED_INPUT`: file shared ZIP V2 hoặc folder đã giải nén; để `None` thì tìm trong Kaggle Input.
        `BASE_SPLIT_INPUT`: chỉ A cần khi prepare; dùng `split_manifest.json` hoặc ZIP baseline có frozen split.
        Notebook không tự tạo split mới hoặc chọn lại 20 glosses.
        `RESTORE_INPUT`: member results ZIP của lần chạy trước; restore trước khi nhận shared.
        `RUN_IDS=None` chạy cả queue; có thể đặt danh sách vài run IDs để chia thành các phiên GPU.
        `RETRY_FAILED` chỉ bật sau khi đã kiểm tra lỗi kỹ thuật, không dùng để chọn attempt có điểm tốt hơn.
    """))
    cells.append(cell("code", f"""
        MEMBER = "{member}"
        BRANCH = "codex/ablation-member-{member.lower()}"
        REPO_URL = "https://github.com/ChienPM-27/SignLanguageReconition_ML.git"
        SOURCE_REF = "ablation-v2-notebooks-v1"
        MODE = "{'prepare' if member == 'A' else 'train'}"  # A: prepare trước; A/B/C: train sau khi nhận shared V2
        SHARED_INPUT = None
        BASE_SPLIT_INPUT = None
        RESTORE_INPUT = None
        RUN_IDS = None  # Ví dụ C: ["G0__F0", "G0__F0-face"]; dùng đúng queue của member
        RETRY_FAILED = False
        RUNTIME_OVERRIDES = {{}}  # Ví dụ: {{"data.local_mirror_dir": "/content/drive/MyDrive/asl-signs-data"}}
        AGGREGATE_INPUTS = []  # Chỉ A: đường dẫn result ZIP của B và C khi tổng hợp
    """))
    cells.append(cell("markdown", """
        ## 2. Lấy source đã pin và cài dependencies còn thiếu

        Notebook pin source bằng tag release. Nếu repo private, thêm `GH_TOKEN` vào Kaggle/Colab Secrets
        hoặc environment; token không được ghi vào URL, command arguments hay logs.
        Nếu local workspace có source chưa commit, notebook dừng trước khi đổi checkout.
    """))
    cells.append(cell("code", """
        import importlib.util
        import os
        from pathlib import Path
        import subprocess
        import sys

        if Path("/kaggle/input").exists():
            WORKSPACE = Path("/kaggle/working")
            CODE_REPO = WORKSPACE / "islr_ablation_v2_code"
            INPUT_ROOTS = [Path("/kaggle/input")]
        elif importlib.util.find_spec("google") is not None and importlib.util.find_spec("google.colab") is not None:
            WORKSPACE = Path("/content")
            CODE_REPO = WORKSPACE / "islr_ablation_v2_code"
            INPUT_ROOTS = [Path("/content/drive/MyDrive/ablation-v2-inputs")]
        else:
            WORKSPACE = Path.cwd().resolve()
            CODE_REPO = WORKSPACE if (WORKSPACE / "configs/islr.yaml").is_file() else WORKSPACE / "islr_ablation_v2_code"
            INPUT_ROOTS = [WORKSPACE / "artifacts/ablation-study-v2/inputs"]

        clone_env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GRAPHIFY_SKIP_HOOK="1")
        token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        if not token:
            try:
                if Path("/kaggle/input").exists():
                    from kaggle_secrets import UserSecretsClient
                    token = UserSecretsClient().get_secret("GH_TOKEN")
                else:
                    from google.colab import userdata
                    token = userdata.get("GH_TOKEN")
            except Exception:
                token = None
        if token:
            import base64
            auth = base64.b64encode(("x-access-token:" + token).encode()).decode()
            clone_env.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
                             GIT_CONFIG_VALUE_0="AUTHORIZATION: basic " + auth)

        def source_git(*args):
            return subprocess.check_output(["git", *args], cwd=CODE_REPO, env=clone_env, text=True).strip()

        if not (CODE_REPO / ".git").exists():
            subprocess.run(["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, str(CODE_REPO)], env=clone_env, check=True)
        if source_git("status", "--porcelain", "--", "src", "scripts", "configs", "tests", "notebooks/ablation_v2_member_a.ipynb", "notebooks/ablation_v2_member_b.ipynb", "notebooks/ablation_v2_member_c.ipynb", "reports/ablation-study-v2/tools"):
            raise RuntimeError("Source có thay đổi local. Giữ bản sửa ở nơi riêng trước khi dùng notebook đã pin.")
        subprocess.run(["git", "fetch", "origin", "tag", SOURCE_REF], cwd=CODE_REPO, env=clone_env, check=True)
        subprocess.run(["git", "checkout", "--detach", SOURCE_REF], cwd=CODE_REPO, env=clone_env, check=True)
        # Remove authentication material from the child-process environment after source acquisition.
        for name in ("GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0"):
            clone_env.pop(name, None)
        token = None
        if "auth" in locals():
            del auth

        packages = {"numpy": "numpy", "pandas": "pandas", "pyarrow": "pyarrow", "yaml": "pyyaml",
                    "sklearn": "scikit-learn", "matplotlib": "matplotlib", "tqdm": "tqdm", "torch": "torch", "kaggle": "kaggle"}
        missing = [package for module, package in packages.items() if importlib.util.find_spec(module) is None]
        if missing:
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *missing], check=True)
        os.chdir(CODE_REPO)
        sys.path.insert(0, str(CODE_REPO))
        sys.path.insert(0, str(CODE_REPO / "reports/ablation-study-v2/tools"))
        print("Source commit:", source_git("rev-parse", "HEAD"))
    """))
    cells.append(cell("markdown", """
        ## 3. Runtime, config queue và phục hồi kết quả

        Colab mặc định dùng Drive; các project roots và folder kết quả tách theo member.
        Kaggle lưu trong `/kaggle/working`, cần **Save Version/Output** hoặc tải ZIP sau phiên chạy.
        `RESTORE_INPUT` chỉ dùng trên runtime mới; kết quả hoàn tất được kiểm chứng rồi bỏ qua khi chạy lại.
    """))
    cells.append(cell("code", """
        import pandas as pd
        from IPython.display import display
        from notebook_support import MemberSession, discover
        import study

        session = MemberSession(MEMBER, RUNTIME_OVERRIDES)
        if RESTORE_INPUT is not None:
            session.restore_results(Path(RESTORE_INPUT))
        if "data.local_mirror_dir" not in session.runtime_overrides:
            mirrors = sorted({path.parent for root in INPUT_ROOTS if root.exists()
                              for path in root.rglob("train.csv") if (path.parent / "train_landmark_files").is_dir()})
            if len(mirrors) == 1:
                session.runtime_overrides["data.local_mirror_dir"] = str(mirrors[0])
                print("Raw dataset:", mirrors[0])
            elif len(mirrors) > 1:
                raise RuntimeError("Có nhiều raw datasets; đặt canonical data.local_mirror_dir trong RUNTIME_OVERRIDES.")
        display(pd.DataFrame(session.queue)[["run_id", "profile", "feature", "dimension", "seeds"]])
    """))
    cells.append(cell("markdown", """
        ## 4. Shared package và raw data

        A/prepare: attach frozen baseline split cũ, tạo 5 geometry caches và intersection,
        giữ signer assignments/class order, rồi xuất shared ZIP. Cell này không training model.
        Train: nhận shared ZIP V2, kiểm tra source/protocol/cohort và đủ raw sequences trước khi vào queue.
    """))
    cells.append(cell("code", """
        if MODE not in ("prepare", "train"):
            raise ValueError("MODE phải là prepare hoặc train.")
        if MODE == "prepare":
            if MEMBER != "A":
                raise ValueError("Chỉ A tạo shared package; B/C dùng MODE=train.")
            baseline = Path(BASE_SPLIT_INPUT) if BASE_SPLIT_INPUT is not None else discover(INPUT_ROOTS, "split_manifest.json")
            shared_zip = session.prepare_shared(baseline)
            print("Giao shared ZIP này cho cả A/B/C, rồi dùng MODE=train:", shared_zip)
        else:
            if not session.shared.exists():
                shared_input = Path(SHARED_INPUT) if SHARED_INPUT is not None else discover(INPUT_ROOTS, "ablation_v2_shared*.zip")
                session.import_shared(shared_input)
            session.check_shared()
            session.ensure_data(session.shared / "config.yaml", study.read_json(session.shared / "split_manifest.json"))
            print("Preflight data/shared hoàn tất. Chưa truy cập test metrics.")
    """))
    cells.append(cell("markdown", """
        ## 5. Training queue và tự lưu kết quả

        `run-config` chạy từng tổ hợp tuần tự với đủ ba seeds. Tổ hợp hoàn tất hợp lệ được bỏ qua,
        lỗi kỹ thuật giữ nguyên logs/attempt và dừng để kiểm tra. Không resume checkpoint giữa các seeds.
        Sau mỗi tổ hợp, có per-run ZIP và member results ZIP; cache/raw data không nằm trong gói nghiệm thu.
    """))
    cells.append(cell("code", """
        if MODE == "train":
            status = session.run_queue(RUN_IDS, retry_failed=RETRY_FAILED)
            display(pd.DataFrame(status["queue"]))
            print("Hoàn tất:", status["completed"], "/", status["expected"], "tổ hợp của member")
        else:
            print("MODE=prepare: chỉ tạo shared package, không chạy training.")
    """))
    cells.append(cell("markdown", """
        ## 6. File nghiệm thu và tổng hợp

        Gửi member results ZIP nguyên vẹn: config/source/split/vocabulary, logs,
        checkpoints/history/train metrics và validation predictions cho từng seed.
        `MEMBER_STATUS.json` báo hoàn tất phần được giao. Ma trận chung vẫn PENDING khi chưa đủ 60 cells.
        Chỉ A điền `AGGREGATE_INPUTS` bằng result ZIP của B/C để chọn winner bằng validation.
        Notebook không đánh giá test; test thực hiện sau khi khóa winner.
    """))
    cells.append(cell("code", """
        from IPython.display import FileLink
        if MODE == "train":
            member_zip = session.export_results()
            display(FileLink(str(member_zip)))
            print("Thư mục kết quả:", session.root)
            print("Per-run ZIPs:", session.root / "bundles")
        else:
            display(FileLink(str(shared_zip)))
        if MEMBER == "A" and AGGREGATE_INPUTS:
            overall = session.aggregate([Path(path) for path in AGGREGATE_INPUTS])
            print("Ma trận chung:", overall["completed_cells"], "/ 60; winner:", overall["winner"])
            display(FileLink(str(session.root / "ALL_MEMBERS_RESULTS.md")))
    """))
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def main():
    for member in "ABC":
        target = ROOT / "notebooks" / f"ablation_v2_member_{member.lower()}.ipynb"
        target.write_text(json.dumps(build(member), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(target)


if __name__ == "__main__":
    main()
