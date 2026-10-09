"""notebooks/01_eda.ipynb is committed executed: structural checks on that file (no data needed).

The EDA must never open the external test cohort (brief section 5) and is shared as a clean,
top-to-bottom run. These tests read only the notebook JSON, so they also run in CI.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "01_eda.ipynb"
MAX_NOTEBOOK_BYTES = 2_000_000  # committed with its figures; keeps the repository small

# A code cell naming any of these would open the external cohort or bypass the loaders' guard.
EXTERNAL_COHORT_NAMES = (
    "load_nlibd",
    "load_d1_test",
    "final_evaluation",
    "Validation",
    "_external",
    "external_diagnosis",
    "check_section3_facts",
    "evaluate_section3_facts",
    "_FactData",
    "_load_d2_cohort",
    "match_nlibd_to_validation",
)
# Reading files directly would skip the guarded loaders of m2m.data altogether.
DIRECT_READS = ("read_csv", "read_table", "read_parquet", "np.load", "open(")
# The repository is public: outputs must not reveal where (or by whom) the notebook was run.
LOCAL_PATH_MARKERS = ("/Users/", "/home/", "/private/", "/tmp/", ":\\Users\\")
TEXT_MIME_TYPES = ("text/plain", "text/html", "text/markdown")


@pytest.fixture(scope="module")
def notebook() -> dict[str, Any]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _text(value: str | list[str]) -> str:
    """nbformat stores multi-line strings either whole or as a list of lines."""
    return value if isinstance(value, str) else "".join(value)


def _code_cells(notebook: dict[str, Any]) -> list[dict[str, Any]]:
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _output_texts(cell: dict[str, Any]) -> list[str]:
    """The human-readable outputs of a cell; image payloads are skipped (base64 is noise)."""
    texts = []
    for output in cell.get("outputs", []):
        if "text" in output:
            texts.append(_text(output["text"]))
        data = output.get("data", {})
        texts.extend(_text(data[mime]) for mime in TEXT_MIME_TYPES if mime in data)
        if output["output_type"] == "error":
            texts.append("\n".join(output.get("traceback", [])))
    return texts


def test_notebook_is_valid_nbformat() -> None:
    nbformat = pytest.importorskip("nbformat")
    nbformat.validate(nbformat.read(NOTEBOOK, as_version=4))


def test_code_never_opens_the_external_cohort(notebook: dict[str, Any]) -> None:
    code = "\n".join(_text(cell["source"]) for cell in _code_cells(notebook))
    named = [name for name in (*EXTERNAL_COHORT_NAMES, *DIRECT_READS) if name in code]
    assert not named, f"the EDA code must not use {named}: data only via the PRISM/source loaders"
    franzosa_calls = re.findall(r"load_franzosa\(([^)]*)\)", code)
    assert all('part="PRISM"' in args for args in franzosa_calls), franzosa_calls


def test_notebook_ran_top_to_bottom_without_errors(notebook: dict[str, Any]) -> None:
    cells = _code_cells(notebook)
    # Counts 1..N in cell order: one fresh kernel ran every cell once, in order.
    assert [cell["execution_count"] for cell in cells] == list(range(1, len(cells) + 1))
    for number, cell in enumerate(cells, start=1):
        kinds = [(o["output_type"], o.get("name")) for o in cell["outputs"]]
        assert ("error", None) not in kinds, f"code cell {number} raised an error"
        assert ("stream", "stderr") not in kinds, f"code cell {number} wrote to stderr"


def test_notebook_is_small_and_reveals_no_local_paths(notebook: dict[str, Any]) -> None:
    assert NOTEBOOK.stat().st_size < MAX_NOTEBOOK_BYTES
    texts = [_text(cell["source"]) for cell in notebook["cells"]]
    texts += [text for cell in _code_cells(notebook) for text in _output_texts(cell)]
    leaked = sorted({marker for marker in LOCAL_PATH_MARKERS for text in texts if marker in text})
    assert not leaked, f"notebook text contains local paths: {leaked}"
