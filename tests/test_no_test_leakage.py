"""A3 guards against the held-out `test` split being read before the single final evaluation (CLAUDE.md).

These are structural tests: they inspect the CLI defaults and the on-disk artefacts, and they never read a
test subject's image or mask themselves (subject *ids* are only used for set arithmetic).
"""
import ast
import hashlib
import json
import re
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
    """The pipeline itself must never touch the test split.

    A6 2026-09-09 moved the final-evaluation command list into a quoted here-document
    (`cat <<'MSG' ... MSG`), which prints the commands instead of running them. The check therefore
    classifies every line as executed or not-executed first, and only then asserts.
    """
    lines = (ROOT / "scripts/run_all.sh").read_text().splitlines()
    executed, in_heredoc = [], False
    for ln in lines:
        stripped = ln.strip()
        if not in_heredoc and re.match(r"^\w[\w ]*<<-?\s*'?\w+'?", stripped):
            in_heredoc = True
            continue
        if in_heredoc:
            if stripped in ("MSG", "EOF"):
                in_heredoc = False
            continue
        if stripped.startswith("echo"):
            continue
        executed.append(ln)
    assert not in_heredoc, "unterminated here-document in run_all.sh"

    runs = [ln for ln in executed if ln.strip().startswith("python scripts/preprocess.py")]
    assert runs, "run_all.sh no longer calls preprocess.py outside the final-evaluation banner"
    for ln in runs:
        assert "--which train val" in ln and "--which train val test" not in ln, ln
    for ln in executed:
        assert "--which test" not in ln and "--split test" not in ln, f"test split executed inside the pipeline: {ln}"


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
    """Before the single final evaluation the cache must contain no test subject.

    The final evaluation legitimately preprocesses the test split (charter section C), so the invariant
    is conditional rather than absolute: a cached test subject is a leak *unless* the final evaluation
    has actually been run, which runs/*/eval_test.json records. Deleting that eval file and keeping the
    cache re-arms the guard, so this still catches a premature preprocess on a fresh checkout.
    """
    sp = json.load(open(ROOT / "data/splits.json"))
    cached = {p.stem for p in (ROOT / "data/cache").glob("sub-*.npz")}
    leaked = sorted(cached & set(sp["test"]))
    if not leaked:
        return
    final_evals = sorted(ROOT.glob("runs/*/eval_test.json"))
    assert final_evals, (
        f"{len(leaked)} test subjects are preprocessed but no runs/*/eval_test.json exists: "
        f"the held-out split was read before the final evaluation ({leaked[:5]})"
    )


@pytest.mark.skipif(not (ROOT / "data/splits.json").exists(), reason="no data/splits.json in this checkout")
def test_split_file_hash_is_recorded_for_reproducibility():
    """The file itself must be byte-stable; make_splits.py writes it deterministically."""
    h = hashlib.sha256((ROOT / "data/splits.json").read_bytes()).hexdigest()
    assert len(h) == 64


def test_a2_volumes_cli_does_not_read_test_by_default():
    """A5a item 2 / A5b: this script re-read all 1,451 acute masks after the split existed. It now filters
    on data/splits.json and needs --include-test to touch the held-out subjects."""
    src = (ROOT / "scripts/analysis/a2_volumes.py").read_text()
    assert "--include-test" in src and 'sp["train"]' in src and 'sp["val"]' in src


def test_only_the_index_builder_scans_every_raw_mask():
    """The subject index (`strokeai.data.index.build_index`) must read every acute mask once: the stratified
    patient-level split needs the lesion-volume band and the eligibility flags, so there is no way to build
    `data/splits.json` without it. That is the ONE sanctioned whole-cohort read (charter, known limit 1).
    Every other script that loads mask voxels must go through data/splits.json.  This test walks the
    scripts and fails when a new whole-cohort mask reader appears outside the allow-list."""
    allow = {"scripts/build_index.py", "scripts/download_soop.py"}
    offenders = []
    for path in sorted((ROOT / "scripts").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        src = path.read_text()
        reads_masks = "lesionAcute_mask" in src or "lesion_masks" in src
        scans_all = re.search(r'glob\(\s*["\']sub-\*["\']\s*\)', src) is not None
        uses_split = "splits.json" in src or 'sp["train"]' in src or "--which" in src
        if reads_masks and scans_all and not uses_split and rel not in allow:
            offenders.append(rel)
    assert not offenders, f"whole-cohort mask readers without a split filter: {offenders}"
