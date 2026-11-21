"""Resolve camera models from the accepted V3 ablation inventory."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts/ablation-study-v3/results/aggregate/b4fd56f03a57/7f86f6ac76de/ALL_MEMBERS_FULL42_RESULTS.json"
DEPLOYMENT = ROOT / "artifacts/ablation-study-v3/deployment"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class CameraModel:
    run_id: str
    seed: int
    regions: tuple[str, ...]
    dimension: int
    macro_f1: float
    freeze_id: str
    archive: Path
    archive_prefix: str
    archive_sha256: str
    checkpoint_sha256: str
    glosses_sha256: str
    macro_f1_std: float = 0.0
    top1: float = 0.0
    top3: float = 0.0
    parameters: int = 0

    @property
    def label(self) -> str:
        return f"{self.run_id} · seed {self.seed}"


def load_catalog(results_path: Path = RESULTS) -> dict[tuple[str, int], CameraModel]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    if data["scope"] != "v3-full42" or data["status"] != "COMPLETED":
        raise ValueError("Cần inventory ablation V3 đã nghiệm thu đầy đủ.")
    catalog = {}
    for run_id in data["ranking"]:
        row = data["records"][run_id]
        archive = Path(row["archive"])
        # Stored inventories include absolute paths from the review machine.
        if not archive.is_file():
            parts = row["archive"].replace("\\", "/").split("/")
            if "artifacts" in parts:
                archive = ROOT.joinpath(*parts[parts.index("artifacts"):])
        hashes = {item["seed"]: item["checkpoint_sha256"]
                  for item in data["details"][run_id]["epoch_details"]}
        if row["status"] != "COMPLETED" or row["provenance"]["freeze_id"] != data["freeze_id"]:
            raise ValueError(f"Inventory không hợp lệ cho {run_id}.")
        for seed in row["seeds"]:
            catalog[run_id, seed] = CameraModel(
                run_id, seed, tuple(row["regions"]), row["dimension"],
                row["metrics"]["macro_f1"]["mean"], data["freeze_id"], archive,
                row["archive_prefix"], data["archive_sha256"][row["owner"]], hashes[seed],
                row["provenance"]["selection_glosses_csv_sha256"],
                row["metrics"]["macro_f1"].get("std", 0.0),
                row["metrics"].get("top1_accuracy", {}).get("mean", 0.0),
                row["metrics"].get("top3_accuracy", {}).get("mean", 0.0),
                int(row["metrics"].get("model_parameters", {}).get("mean", 0)),
            )
    return catalog


def materialize_model(model: CameraModel, deployment: Path = DEPLOYMENT) -> tuple[Path, Path]:
    """Extract only the selected checkpoint and labels, checking accepted hashes."""
    if any(Path(value).name != value or value in (".", "..") or "\\" in value
           for value in (model.freeze_id, model.run_id)):
        raise ValueError("Tên model hoặc freeze không hợp lệ.")
    folder = deployment / model.freeze_id / model.run_id
    checkpoint = folder / f"seed_{model.seed}/best_model.pt"
    glosses = folder / "glosses.csv"
    if checkpoint.is_file() and digest(checkpoint) != model.checkpoint_sha256:
        raise ValueError(f"Checksum checkpoint khác bản nghiệm thu: {model.label}.")
    if glosses.is_file() and digest(glosses) != model.glosses_sha256:
        raise ValueError(f"Checksum glosses khác bản nghiệm thu: {model.run_id}.")
    if checkpoint.is_file() and glosses.is_file():
        return checkpoint, glosses
    if not model.archive.is_file():
        raise FileNotFoundError(f"Thiếu ZIP kết quả: {model.archive}")
    if digest(model.archive) != model.archive_sha256:
        raise ValueError(f"Checksum ZIP khác bản nghiệm thu: {model.archive.name}.")
    with zipfile.ZipFile(model.archive) as archive:
        candidates = [name for name in archive.namelist()
                      if name.startswith(model.archive_prefix + "/runs/")
                      and name.endswith(f"/seed_{model.seed}/best_model.pt")]
        if len(candidates) != 1:
            raise ValueError(f"Cần đúng một checkpoint trong ZIP: {model.label}.")
        entries = ((candidates[0], checkpoint, model.checkpoint_sha256),
                   (model.archive_prefix + "/inputs/selection/glosses.csv", glosses, model.glosses_sha256))
        for name, target, expected in entries:
            if target.is_file():
                continue
            payload = archive.read(name)
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f"Artifact trong ZIP khác bản nghiệm thu: {name}.")
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
            try:
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    return checkpoint, glosses
