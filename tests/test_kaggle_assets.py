import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_l1_training.ipynb"
PREPARE_SCRIPT = ROOT / "scripts" / "prepare_l1_data.py"


def load_prepare_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_l1_data_test", PREPARE_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kaggle_notebook_is_clean_and_code_cells_compile() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 15
    sources = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell["source"])
        sources.append(source)
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile(source, f"notebook-cell-{index}", "exec")
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "probe_source"
                    for target in node.targets
                ):
                    compile(
                        ast.literal_eval(node.value),
                        f"notebook-cell-{index}-cuda-probe",
                        "exec",
                    )
    combined = "\n".join(sources)
    assert "prepare_l1_data.py" in combined
    assert "scripts/train_l1.py" in combined
    assert "configs/l1_attention_20m.json" in combined
    assert "configs/l1_edge_20m.json" in combined
    assert "scripts/evaluate_completions.py" in combined
    assert 'VARIANTS = ["attention"]' in combined
    assert "https://download.pytorch.org/whl/cu126" in combined
    assert '"torch==2.7.1"' in combined
    assert 'required_arch = f"sm_{major}{minor}"' in combined
    assert "cuda_smoke_value" in combined
    assert "precision = 'bf16' if capability[0] >= 8 else 'fp16'" in combined
    assert 'BUNDLE_NAME = "l1-github-pilot-data-v1.tar.gz"' in combined
    assert 'archive.extractall(destination, filter="data")' in combined


def test_prepare_script_fully_validates_packed_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_prepare_module()
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_bytes(b"pinned tokenizer")
    tokenizer_hash = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    tokens = tmp_path / "tokens"
    tokens.mkdir()

    split_identities = {}
    for index, split in enumerate(("train", "validation", "test"), start=1):
        array = np.arange(index * 5, dtype=np.uint32)
        path = tokens / f"{split}.bin"
        array.tofile(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        identity = {"tokens": len(array), "sha256": digest, "dtype": "uint32"}
        split_identities[split] = identity
        path.with_suffix(".bin.json").write_text(json.dumps(identity), encoding="utf-8")

    corpus_hash = "a" * 64
    (tokens / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_sha256": corpus_hash,
                "tokenizer_sha256": tokenizer_hash,
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "corpus": {"corpus_sha256": corpus_hash},
                "selected_tokenizer": {"tokenizer_sha256": tokenizer_hash},
                "packed_tokens": {"splits": split_identities},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "REPORT", report)
    monkeypatch.setattr(module, "TOKENIZER", tokenizer)

    verified = module.verify_token_streams(tokens)
    assert verified["splits"]["train"]["tokens"] == 5

    train = tokens / "train.bin"
    corrupted = bytearray(train.read_bytes())
    corrupted[0] ^= 0xFF
    train.write_bytes(corrupted)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.verify_token_streams(tokens)
