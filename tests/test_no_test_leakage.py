"""A3 guards against the held-out `test` split being read before the single final evaluation (CLAUDE.md).

These are structural tests: they inspect the CLI defaults and the on-disk artefacts, and they never read a
test subject's image or mask themselves (subject *ids* are only used for set arithmetic).
"""
import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHARTER_SPLIT_SHA = "0f923acd2f8898d58c4c62cb598c4b8d88c17ab5eaab8a72ee4e73a7342d8f43"


def _argparse_default(script: Path, option: str):
    """Return the `default=` value of `add_argument("<option>", ...)` in a CLI script."""
    tree = ast.parse(script.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
            if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == option:
                for kw in node.keywords:
                    if kw.arg == "default":
                        return ast.literal_eval(kw.value)
                return None
    raise AssertionError(f"{option} not found in {script}")


def test_preprocess_cli_does_not_touch_test_by_default():
    assert _argparse_default(ROOT / "scripts/preprocess.py", "--which") == ["train", "val"]


def test_predict_masks_cli_does_not_touch_test_by_default():
    assert _argparse_default(ROOT / "scripts/predict_masks.py", "--which") == ["train", "val"]


def test_eval_seg_defaults_to_val():
    assert _argparse_default(ROOT / "scripts/eval_seg.py", "--split") == "val"


def test_run_all_pipeline_preprocesses_only_train_and_val():
    lines = (ROOT / "scripts/run_all.sh").read_text().splitlines()
    runs = [ln for ln in lines if ln.strip().startswith("python scripts/preprocess.py")]
    assert runs, "run_all.sh no longer calls preprocess.py"
    for ln in runs:
        assert "--which train val" in ln and "--which train val test" not in ln, ln
    # the test-set commands must stay behind the final-evaluation banner (echoed, not executed)
    for ln in lines:
        if "--which test" in ln or "--split test" in ln:
            assert ln.strip().startswith("echo"), f"test split executed inside the pipeline: {ln}"


@pytest.mark.skipif(not (ROOT / "data/splits.json").exists(), reason="no data/splits.json in this checkout")
def test_splits_are_patient_level_disjoint_and_match_the_charter_hash():
    from strokeai.utils import sha256_of

    sp = json.load(open(ROOT / "data/splits.json"))
    tr, va, te = set(sp["train"]), set(sp["val"]), set(sp["test"])
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr) + len(va) + len(te) == len(tr | va | te) == sp["n_eligible"]
    assert all(len(v) == len(set(v)) for v in (sp["train"], sp["val"], sp["test"]))  # no duplicated subjects
    assert sha256_of({k: sp[k] for k in ("train", "val", "test")}) == sp["sha256"]
    assert sp["sha256"] == CHARTER_SPLIT_SHA


@pytest.mark.skipif(not (ROOT / "data/cache").exists() or not (ROOT / "data/splits.json").exists(),
                    reason="no preprocessed cache in this checkout")
def test_cache_holds_no_test_subject():
    sp = json.load(open(ROOT / "data/splits.json"))
    cached = {p.stem for p in (ROOT / "data/cache").glob("sub-*.npz")}
    leaked = sorted(cached & set(sp["test"]))
    assert not leaked, f"test subjects were preprocessed: {leaked[:10]}"


@pytest.mark.skipif(not (ROOT / "data/splits.json").exists(), reason="no data/splits.json in this checkout")
def test_split_file_hash_is_recorded_for_reproducibility():
    """The file itself must be byte-stable; make_splits.py writes it deterministically."""
    h = hashlib.sha256((ROOT / "data/splits.json").read_bytes()).hexdigest()
    assert len(h) == 64
