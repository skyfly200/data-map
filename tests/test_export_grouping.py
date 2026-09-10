"""How the export decides to split the dataset, and what it calls the pieces.

These pin a bug that was quiet and expensive. The taxonomy stage adds `genus`
and the other rank columns whether or not it resolves anything, so a run where
resolution did not happen left an all-empty `genus` column. The export chose it
anyway — existence was the only test — and dropped all 48,233 records into one
"Unknown" group: a 47 MB near-duplicate of the whole dataset, listed in the UI
as if it were a single species, doubling the build's compression work.

Nothing about that looked like an error. It just quietly produced a second copy
of everything under a name that made it look deliberate.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from export_geojson import _label_for_group, pick_group_column  # noqa: E402


def feat(**props):
    return {"type": "Feature", "properties": props}


# ── Choosing what to group by ────────────────────────────────────────────────

def test_a_populated_column_is_used():
    df = pd.DataFrame({
        'species': ['Amanita muscaria', 'Boletus edulis'],
        'genus': ['Amanita', 'Boletus'],
    })
    assert pick_group_column(df, 'genus') == 'genus'


def test_an_empty_genus_column_falls_back_to_species():
    # The actual bug. The column is there, so the old check passed it, and every
    # record landed in one group.
    df = pd.DataFrame({
        'species': ['Amanita muscaria', 'Boletus edulis'],
        'genus': [None, None],
    })
    assert pick_group_column(df, 'genus') == 'species'


def test_blank_strings_count_as_empty():
    # A column of empty strings is as useless as one of NaNs, and pandas does
    # not treat them the same way.
    df = pd.DataFrame({
        'species': ['Amanita muscaria', 'Boletus edulis'],
        'genus': ['', '   '],
    })
    assert pick_group_column(df, 'genus') == 'species'


def test_a_column_that_places_most_records_is_used():
    df = pd.DataFrame({
        'species': ['Amanita muscaria', 'Boletus edulis', 'Boletus edulis'],
        'genus': ['Amanita', 'Boletus', None],
    })
    assert pick_group_column(df, 'genus') == 'genus'


def test_a_barely_populated_column_is_rejected():
    # The real shape of the shipped store: genus resolved for 373 rows of
    # 19,462, two percent. Grouping by it puts the other ninety-eight into one
    # "Unknown" bucket, which is the 47 MB duplicate this exists to prevent.
    df = pd.DataFrame({
        'species': [f'Species {i}' for i in range(100)],
        'genus': ['Amanita'] + [None] * 99,
    })
    assert pick_group_column(df, 'genus') == 'species'


def test_the_bar_is_a_majority_of_records():
    half = pd.DataFrame({
        'species': ['a', 'b', 'c', 'd'],
        'genus': ['A', 'B', None, None],
    })
    assert pick_group_column(half, 'genus') == 'genus'
    under = pd.DataFrame({
        'species': ['a', 'b', 'c', 'd'],
        'genus': ['A', None, None, None],
    })
    assert pick_group_column(under, 'genus') == 'species'


def test_a_thin_column_still_beats_no_grouping_when_it_is_all_there_is():
    # No species at all and a little genus: one file per known genus plus an
    # Unknown is more use than one undivided file.
    df = pd.DataFrame({'genus': ['Amanita'] + [None] * 9})
    assert pick_group_column(df, 'genus') == 'genus'


def test_a_missing_column_falls_back():
    df = pd.DataFrame({'species': ['Amanita muscaria']})
    assert pick_group_column(df, 'genus') == 'species'


def test_nothing_usable_returns_none():
    # The caller writes one undivided file rather than inventing a grouping.
    assert pick_group_column(pd.DataFrame({'date': ['2026-01-01']}), 'genus') is None
    assert pick_group_column(pd.DataFrame({'species': [None]}), 'genus') is None
    assert pick_group_column(pd.DataFrame(), 'genus') is None


# ── Naming a group ───────────────────────────────────────────────────────────

def test_one_species_is_named_for_it():
    feats = [feat(species='Amanita muscaria'), feat(species='Amanita muscaria')]
    assert _label_for_group(feats, 'amanita-muscaria') == 'Amanita muscaria'


def test_a_genus_group_is_named_for_the_genus():
    # Not for whichever species happened to be first, which is how a bucket of
    # 48,221 records came to be labelled "Agaricus abruptibulbus".
    feats = [
        feat(species='Amanita muscaria', genus='Amanita'),
        feat(species='Amanita pantherina', genus='Amanita'),
    ]
    assert _label_for_group(feats, 'amanita') == 'Amanita'


def test_a_mixed_group_does_not_claim_to_be_a_taxon():
    feats = [
        feat(species='Amanita muscaria', genus='Amanita'),
        feat(species='Boletus edulis', genus='Boletus'),
    ]
    label = _label_for_group(feats, 'unknown')
    assert label == 'Unknown'
    # The important part: it must not name itself after one of its members.
    assert 'Amanita' not in label and 'Boletus' not in label


def test_a_group_with_no_names_falls_back_to_its_slug():
    feats = [feat(date='2026-01-01'), feat(date='2026-01-02')]
    assert _label_for_group(feats, 'front-range') == 'Front range'


def test_one_member_missing_a_value_breaks_the_shared_name():
    # Two of three share a genus and the third has none. "Amanita" would be a
    # claim about a record that does not support it.
    feats = [
        feat(species='Amanita muscaria', genus='Amanita'),
        feat(species='Amanita pantherina', genus='Amanita'),
        feat(species='Something else'),
    ]
    assert _label_for_group(feats, 'mixed') == 'Mixed'


@pytest.mark.parametrize('slug,expected', [
    ('unknown', 'Unknown'),
    ('front-range', 'Front range'),
    ('a_b', 'A b'),
    ('', ''),
])
def test_slug_labels_are_readable(slug, expected):
    assert _label_for_group([feat(date='x')], slug) == expected
