"""The counts the documents publish have to match the suite that produces them.

The README and the experiment report both state how many tests there are, and both
have been wrong - a number that only a person updates drifts the moment someone
forgets, and two rounds were spent finding stale counts by hand. This checks them
instead of trusting them.

The collection runs in a subprocess because it has to count the suite this file is
part of, itself included. The Chinese report states the same totals and is not
checked here: it lives outside the repository, next to the build script that turns
it into the delivered document.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
REPORT = REPO_ROOT / "Document" / "Week4" / "Week4_Experiment_Report.md"

#: The suite Week 3 handed over, counted on d67de1f. A historical fact about a
#: commit rather than a property of this tree, so it is stated instead of measured.
WEEK3_TOTAL = 300


@pytest.fixture(scope="module")
def collected() -> dict[str, int]:
    """Per-file collected counts, from one collection of the whole suite."""
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            f"--basetemp={REPO_ROOT / '.pytest-doccheck'}",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    counts: dict[str, int] = {}
    for line in done.stdout.splitlines():
        found = re.match(r"(tests/[\w/]+\.py)::", line)
        if found:
            counts[found.group(1)] = counts.get(found.group(1), 0) + 1
    assert counts, f"the collection produced nothing:\n{done.stdout[-2000:]}"
    return counts


def test_the_readme_states_the_size_of_the_suite_it_describes(
    collected: dict[str, int],
) -> None:
    total = sum(collected.values())
    claimed = set(re.findall(r"(\d+) tests, ruff clean", README.read_text(encoding="utf-8")))

    assert claimed, "the README no longer states a test count"
    assert claimed == {str(total)}, f"the README says {sorted(claimed)}; the suite has {total}"


def test_the_report_states_the_size_of_the_suite_it_describes(
    collected: dict[str, int],
) -> None:
    total = sum(collected.values())
    claimed = set(re.findall(r"\*\*(\d+) passed\*\*", REPORT.read_text(encoding="utf-8")))

    assert claimed, "the report no longer states a test count"
    assert claimed == {str(total)}, f"the report says {sorted(claimed)}; the suite has {total}"

    # The deliverables list states the same fact a second way, as a count of what
    # the week added. Nothing checked it, and it had drifted by more than a
    # hundred tests while every other number in the report stayed current.
    delivered = re.search(r"^- (\d+) new tests\.$", REPORT.read_text(encoding="utf-8"), re.MULTILINE)
    assert delivered, "the deliverables list no longer says how many tests Week 4 added"
    assert int(delivered.group(1)) == total - WEEK3_TOTAL, (
        f"the deliverables list says {delivered.group(1)} new tests; "
        f"the suite grew by {total - WEEK3_TOTAL} from the {WEEK3_TOTAL} Week 3 handed over"
    )


def test_the_per_file_table_names_the_counts_those_files_have(
    collected: dict[str, int],
) -> None:
    """Both the rows and the sentence above them, because the two disagreed once.

    The table had drifted file by file while only the total was kept up to date,
    and the sentence miscounted the files for a while.
    """
    report = REPORT.read_text(encoding="utf-8")
    rows = re.findall(r"^\| `(test_[a-z0-9_]+\.py)` \| (\d+) cases", report, re.MULTILINE)

    assert len(rows) == 14, f"the table lists {len(rows)} files, not the fourteen it claims"
    for name, claimed in rows:
        actual = collected.get(f"tests/{name}", 0)
        assert int(claimed) == actual, f"{name}: the table says {claimed}, the file has {actual}"

    prose = re.search(r"Week 4 adds (\d+): (\d+) in the\s+fourteen", report)
    assert prose, "the sentence stating the totals has moved or changed shape"
    assert sum(int(count) for _, count in rows) == int(prose.group(2)), (
        "the table does not add up to the number the sentence gives for it"
    )


def test_the_other_suites_the_sentence_lists_have_the_counts_it_gives(
    collected: dict[str, int],
) -> None:
    """The same sentence's second half, which nothing checked and which went stale.

    The sentence states both a total for the suites it does not tabulate and a
    per-file delta for each of them. Only the ten tabulated files were guarded, so
    the total could drift away from its own list without the suite noticing - and
    it did, in the Chinese translation of this report, where the row total read 52
    while the eight numbers beside it added up to 57.

    The baselines are Week 3's counts on d67de1f. They are a historical fact about
    a commit rather than a property of this tree, so they are stated here instead
    of measured.
    """
    # Week 3 ended at WEEK3_TOTAL; these are the eight files it already had.
    week3 = {
        "test_control_safety.py": 20,
        "test_model_mock.py": 10,
        "test_config.py": 0,
        "test_plan_parser.py": 18,
        "test_model_config.py": 20,
        "test_documented_counts.py": 0,
        "test_ocr.py": 18,
        "test_recording.py": 14,
    }
    report = REPORT.read_text(encoding="utf-8")
    listed = re.findall(r"`(test_[a-z0-9_]+\.py)` (\d+)(?! \|)", report)

    assert len(listed) == len(week3), (
        f"the sentence lists {len(listed)} files, not the {len(week3)} it used to"
    )
    for name, claimed in listed:
        assert name in week3, f"{name} is listed but has no Week 3 baseline here"
        delta = collected.get(f"tests/{name}", 0) - week3[name]
        assert int(claimed) == delta, (
            f"{name}: the sentence says {claimed} new tests, "
            f"the file has {collected.get(f'tests/{name}', 0)} against a baseline of {week3[name]}"
        )

    prose = re.search(
        r"Week 4 adds (\d+): (\d+) in the\s+fourteen\s+files below, and (\d+) spread", report
    )
    assert prose, "the sentence stating the totals has moved or changed shape"
    total_added, tabulated, spread = (int(group) for group in prose.groups())
    assert sum(int(count) for _, count in listed) == spread, (
        "the other-suites total does not match the numbers listed beside it"
    )
    assert tabulated + spread == total_added, (
        "the two halves of the sentence do not add up to the total it states"
    )
