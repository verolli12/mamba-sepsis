import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

torch = pytest.importorskip("torch")

from src.dataset import create_dataloaders


def _write_psv(path: Path, n_rows: int, label: int):
    cols = [f"f{i}" for i in range(40)] + ["SepsisLabel"]
    data = np.random.randn(n_rows, 40).astype(float)
    df = pd.DataFrame(data, columns=cols[:-1])
    df["SepsisLabel"] = label
    df.to_csv(path, sep="|", index=False)


def test_split_deterministic_and_manifest(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    for i in range(20):
        _write_psv(data_dir / f"p_{i:03d}.psv", n_rows=10 + (i % 5), label=i % 2)

    manifest_a = tmp_path / "split_a.json"
    manifest_b = tmp_path / "split_b.json"

    create_dataloaders(
        data_dir=str(data_dir),
        seq_length=12,
        batch_size=4,
        val_split=0.2,
        test_split=0.1,
        normalize=True,
        seed=123,
        include_test=True,
        split_manifest_path=str(manifest_a),
    )
    create_dataloaders(
        data_dir=str(data_dir),
        seq_length=12,
        batch_size=4,
        val_split=0.2,
        test_split=0.1,
        normalize=True,
        seed=123,
        include_test=True,
        split_manifest_path=str(manifest_b),
    )

    a = json.loads(manifest_a.read_text(encoding="utf-8"))
    b = json.loads(manifest_b.read_text(encoding="utf-8"))

    assert a["train_files"] == b["train_files"]
    assert a["val_files"] == b["val_files"]
    assert a["test_files"] == b["test_files"]
    assert a["n_total"] == 20
    assert a["n_train"] + a["n_val"] + a["n_test"] == 20


def test_batch_shapes(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    for i in range(12):
        _write_psv(data_dir / f"q_{i:03d}.psv", n_rows=8, label=(i % 3 == 0))

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=str(data_dir),
        seq_length=16,
        batch_size=3,
        val_split=0.2,
        test_split=0.1,
        normalize=True,
        seed=42,
        include_test=True,
    )

    x, mask, y = next(iter(train_loader))
    assert x.ndim == 3 and x.shape[-1] == 40
    assert mask.shape == x.shape
    assert y.ndim == 1

    assert len(val_loader) > 0
    assert len(test_loader) > 0
