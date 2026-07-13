"""The length bands, checked without a cluster.

A banding bug does not raise. It silently drops rows into `NULL`, or counts them
twice, and the profile table still looks perfectly reasonable — a tidy little
histogram that is quietly wrong. So the bands are checked as an interval cover:
no gaps, no overlaps, nothing unclassifiable.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

TASK = Path(__file__).resolve().parents[1] / "pipelines/rnacentral/ingestion/ingest_rna"

# Loaded by PATH, under a unique name — not by `sys.path.insert` + `import
# transformations`.
#
# Every example in this repo has a file called `transformations.py`. Under a single
# pytest process the first one imported wins the name `transformations` in
# sys.modules, and every later test silently gets THAT module instead of its own.
# Which one wins depends on collection order, so the failure moves around when you
# add an example. (It is the same trap the engine's task runner evicts sys.modules
# to avoid.)
_spec = importlib.util.spec_from_file_location(
    "ingest_rna_transformations", TASK / "transformations.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

LENGTH_BANDS = _mod.LENGTH_BANDS


def band_for(length: int) -> str | None:
    """The same rule the task applies: [low, high)."""
    for low, high, label in LENGTH_BANDS:
        if low <= length < high:
            return label
    return None


def test_no_gaps():
    """Each band starts exactly where the previous one ended.

    A gap means real sequences land in no band at all and vanish from the profile
    into a NULL row that nobody reads.
    """
    for (_low, high, _l), (next_low, _nh, _nl) in zip(LENGTH_BANDS, LENGTH_BANDS[1:]):
        assert high == next_low, f"gap or overlap between {high} and {next_low}"


def test_every_plausible_length_lands_in_exactly_one_band():
    for length in [0, 1, 49, 50, 199, 200, 999, 1000, 9999, 10_000, 15_466, 1_000_000]:
        hits = [label for low, high, label in LENGTH_BANDS if low <= length < high]
        assert len(hits) == 1, f"length {length} matched {len(hits)} bands: {hits}"


def test_the_bands_match_what_the_live_database_returned():
    """Real values from the live run, in the band they must fall in.

    These came from the 200k-row slice actually pulled from RNAcentral: the shortest
    sequence is 10 bases, the longest in the slice is 15,466. If somebody retunes the
    bands and these move, the README's numbers are stale too.
    """
    assert band_for(10) == "tiny (<50)"
    assert band_for(27) == "tiny (<50)"
    assert band_for(115) == "small (50-200)"
    assert band_for(508) == "medium (200-1k)"
    assert band_for(1434) == "long (1k-10k)"
    assert band_for(15_466) == "very long (>10k)"


def test_zero_and_negative_lengths_are_not_silently_counted_as_tiny():
    """A negative length is corrupt data, not a very small RNA.

    The first band starts at 0, so a negative value falls outside every band and shows
    up as NULL in the profile — visible, which is what you want. If someone ever
    "fixes" that by opening the first band to -inf, corruption starts being reported
    as biology.
    """
    assert band_for(-1) is None
    assert band_for(0) == "tiny (<50)"
