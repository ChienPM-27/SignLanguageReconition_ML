"""Self-contained regression coverage for frozen-campaign reporting compatibility."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from src.data.device_loader import DeviceDataLoader
import src.training.evaluate as evaluation


class Samples(Dataset):
    def __init__(self):
        self.preloaded_features = torch.eye(3).repeat(2, 1).unsqueeze(1)
        self.preloaded_labels = torch.arange(6) % 3
        self.preprocess_identity = "synthetic-identity"
        self.samples = [{"sequence_id": f"sample-{index}", "participant_id": "signer",
                         "label": int(label)} for index, label in enumerate(self.preloaded_labels)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.preloaded_features[index], self.preloaded_labels[index]


class Model(torch.nn.Module):
    num_classes = 3

    def forward(self, batch):
        return batch[:, 0, :] * 10


@pytest.fixture
def hotfix(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "report_predictions_v1.py"
    spec = importlib.util.spec_from_file_location("report_predictions_v1_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = evaluation.evaluate_test_set
    monkeypatch.setattr(evaluation, "evaluate_test_set", original)
    # Register absent attributes with monkeypatch so its cleanup removes the patch.
    monkeypatch.setattr(DeviceDataLoader, "_reporting_hotfix", None, raising=False)
    monkeypatch.setattr(DeviceDataLoader, "sampler", None, raising=False)
    monkeypatch.delattr(DeviceDataLoader, "sampler")
    return module


def report(loader, tmp_path):
    return evaluation.evaluate_test_set(Model(), loader, tmp_path / "unused.csv",
                                        ["a", "b", "c"], torch.device("cpu"),
                                        report_json_path=tmp_path / "report.json")


def test_reproduces_missing_predictions_and_repairs_export(hotfix, tmp_path):
    samples = Samples()
    loader = DeviceDataLoader(samples, batch_size=4, shuffle=False)
    before = report(loader, tmp_path)
    assert "predictions" not in before
    tensors_before = [batch[0].clone() for batch in loader]
    hotfix.install_export_compatibility()
    assert isinstance(loader.sampler, SequentialSampler)
    after = report(loader, tmp_path)
    assert [row["sequence_id"] for row in after["predictions"]] == before["sample_membership"]
    assert [row["y_true"] for row in after["predictions"]] == samples.preloaded_labels.tolist()
    assert all(torch.equal(left, right[0]) for left, right in zip(tensors_before, loader))
    for key in ("top1_accuracy", "top3_accuracy", "macro_f1", "loss", "confusion_matrix"):
        assert after[key] == before[key]
    assert json.loads((tmp_path / "report.json").read_text())["reporting_hotfix"]["id"] == hotfix.HOTFIX_ID
    hotfix.install_export_compatibility()
    assert evaluation.evaluate_test_set._reporting_hotfix == hotfix.HOTFIX_ID


@pytest.mark.parametrize("shuffle,drop_last,batch_size", [(True, False, 4), (False, True, 4)])
def test_preserves_alignment_rejection(hotfix, tmp_path, shuffle, drop_last, batch_size):
    hotfix.install_export_compatibility()
    loader = DeviceDataLoader(Samples(), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    assert "predictions" not in report(loader, tmp_path)


def test_standard_loader_predictions_and_metrics_unchanged(hotfix, tmp_path):
    loader = DataLoader(Samples(), batch_size=4, shuffle=False)
    before = report(loader, tmp_path)
    hotfix.install_export_compatibility()
    after = report(loader, tmp_path)
    assert {key: value for key, value in after.items() if key != "reporting_hotfix"} == before


def test_training_subprocess_gets_bootstrap_without_changing_arguments(hotfix, monkeypatch, tmp_path):
    tools = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    import study
    calls = []
    monkeypatch.setattr(study, "run_logged", lambda command, log: calls.append((command, log)))
    session = SimpleNamespace(root=tmp_path, shared=tmp_path / "shared")
    original_source = study.source_manifest()
    hotfix.install(session)
    command = ["python", "-X", "utf8", "-u", "scripts/20_train.py", "--config", "unchanged.yaml"]
    study.run_logged(command, tmp_path / "train.log")
    patched = calls[0][0]
    assert patched[:4] == command[:4]
    assert patched[5:] == ["--bootstrap", *command[4:]]
    prepare = ["python", "scripts/10_prepare_data.py", "--stage", "preprocess"]
    study.run_logged(prepare, tmp_path / "prepare.log")
    assert calls[1][0] == prepare
    hotfix.install(session)
    assert study.source_manifest() == original_source
    assert (tmp_path / "REPORTING_HOTFIX.json").is_file()


def test_bootstrap_exports_predictions_in_real_child_process(hotfix, tmp_path):
    script = tmp_path / "scripts/20_train.py"
    script.parent.mkdir()
    script.write_text('''import json
from pathlib import Path
import torch
from torch.utils.data import TensorDataset
from src.data.device_loader import DeviceDataLoader
from src.training.evaluate import evaluate_test_set

features = torch.eye(3).unsqueeze(1)
labels = torch.arange(3)
dataset = TensorDataset(features, labels)
dataset.preloaded_features = features
dataset.preloaded_labels = labels
dataset.preprocess_identity = "identity"
dataset.samples = [{"sequence_id": str(i), "participant_id": "person", "label": i} for i in range(3)]

class Model(torch.nn.Module):
    num_classes = 3
    def forward(self, batch):
        return batch[:, 0, :] * 10

report = evaluate_test_set(Model(), DeviceDataLoader(dataset, batch_size=2),
                           Path("absent.csv"), ["a", "b", "c"], torch.device("cpu"))
print(json.dumps({"count": len(report.get("predictions", [])), "audit": report["reporting_hotfix"]["id"]}))
''', encoding="utf-8")
    repo = Path(__file__).resolve().parents[4]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo) + os.pathsep + environment.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, str(Path(hotfix.__file__)), "--bootstrap", str(script)],
                            cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {"count": 3, "audit": hotfix.HOTFIX_ID}
