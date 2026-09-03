"""Resolving an observation's full taxonomic ancestry.

The old code split a binomial on the space to get a genus. That is right only
when the identification happens to be at species level, and gives nothing above
it — so these tests are mostly about the cases that broke: identifications at
family or genus level, subspecies, and ancestry that only partly resolved.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import taxonomy  # noqa: E402


# A real-shaped slice of the fungal tree. Ids are iNaturalist's.
FUNGI = {
    47170: {'id': 47170, 'name': 'Fungi', 'rank': 'kingdom'},
    47169: {'id': 47169, 'name': 'Basidiomycota', 'rank': 'phylum'},
    47168: {'id': 47168, 'name': 'Agaricomycetes', 'rank': 'class'},
    47167: {'id': 47167, 'name': 'Agaricales', 'rank': 'order'},
    50814: {'id': 50814, 'name': 'Amanitaceae', 'rank': 'family'},
    50815: {'id': 50815, 'name': 'Amanita', 'rank': 'genus'},
    48701: {'id': 48701, 'name': 'Amanita muscaria', 'rank': 'species'},
}
CHAIN = [47170, 47169, 47168, 47167, 50814, 50815]


def taxon(taxon_id, ancestors=CHAIN, **extra):
    """An observation's inline taxon: its own record plus its ancestor ids."""
    return {**FUNGI[taxon_id], 'ancestor_ids': ancestors, **extra}


def test_species_identification_fills_every_rank():
    ranks = taxonomy.ranks_from_ancestry(taxon(48701), FUNGI)
    assert ranks == {
        'kingdom': 'Fungi',
        'phylum': 'Basidiomycota',
        'class': 'Agaricomycetes',
        'order': 'Agaricales',
        'family': 'Amanitaceae',
        'genus': 'Amanita',
        'species': 'Amanita muscaria',
    }


def test_genus_identification_leaves_species_empty():
    # The old code gave this a species of "Amanita", which is not a species.
    ranks = taxonomy.ranks_from_ancestry(taxon(50815, ancestors=CHAIN[:-1]), FUNGI)
    assert ranks['genus'] == 'Amanita'
    assert ranks['species'] is None
    assert ranks['family'] == 'Amanitaceae'


def test_family_identification_leaves_genus_and_species_empty():
    ranks = taxonomy.ranks_from_ancestry(taxon(50814, ancestors=CHAIN[:-2]), FUNGI)
    assert ranks['family'] == 'Amanitaceae'
    assert ranks['genus'] is None
    assert ranks['species'] is None
    assert ranks['order'] == 'Agaricales'


def test_ancestry_string_is_read_when_ancestor_ids_are_absent():
    # A nested taxon carries the chain as a slash-separated string instead.
    nested = {'id': 48701, 'name': 'Amanita muscaria', 'rank': 'species',
              'ancestry': '47170/47169/47168/47167/50814/50815'}
    assert taxonomy.ranks_from_ancestry(nested, FUNGI)['family'] == 'Amanitaceae'


def test_a_subspecies_rolls_up_into_its_species():
    # Giving subspecies its own column would split one species across several
    # rows in every chart, so it collapses into the species it belongs to.
    by_id = {**FUNGI, 9: {'id': 9, 'name': 'Amanita muscaria flavivolvata',
                          'rank': 'variety'}}
    ranks = taxonomy.ranks_from_ancestry(
        {'id': 9, 'name': 'Amanita muscaria flavivolvata', 'rank': 'variety',
         'ancestor_ids': CHAIN + [48701]}, by_id)
    assert ranks['species'] == 'Amanita muscaria'
    assert ranks['genus'] == 'Amanita'


def test_a_subspecies_with_no_parent_species_in_the_chain_still_yields_one():
    ranks = taxonomy.ranks_from_ancestry(
        {'name': 'Amanita muscaria flavivolvata', 'rank': 'variety', 'ancestor_ids': CHAIN}, FUNGI)
    assert ranks['species'] == 'Amanita muscaria'


def test_partial_resolution_keeps_what_it_did_resolve():
    # One failed lookup must not cost the other six ranks.
    partial = {k: v for k, v in FUNGI.items() if k != 47167}   # order missing
    ranks = taxonomy.ranks_from_ancestry(taxon(48701), partial)
    assert ranks['order'] is None
    assert ranks['kingdom'] == 'Fungi'
    assert ranks['family'] == 'Amanitaceae'
    assert ranks['species'] == 'Amanita muscaria'


def test_genus_is_recovered_from_the_binomial_when_the_chain_lacks_it():
    no_genus = {k: v for k, v in FUNGI.items() if k != 50815}
    assert taxonomy.ranks_from_ancestry(taxon(48701), no_genus)['genus'] == 'Amanita'


def test_no_taxon_yields_empty_ranks_rather_than_raising():
    assert taxonomy.ranks_from_ancestry(None) == {r: None for r in taxonomy.RANKS}


def test_taxon_ids_collects_the_union_of_taxa_and_ancestors():
    obs = [{'taxon': taxon(48701)}, {'taxon': taxon(50815, ancestors=CHAIN[:-1])}]
    assert taxonomy.taxon_ids_in(obs) == set(CHAIN) | {48701, 50815}


def test_taxon_ids_ignores_records_with_no_taxon():
    assert taxonomy.taxon_ids_in([{'taxon': None}, {}, {'taxon': taxon(48701)}]) \
        == set(CHAIN) | {48701}


def test_taxonomy_for_carries_the_identification_rank_and_common_name():
    out = taxonomy.taxonomy_for(taxon(48701, preferred_common_name='Fly Agaric'), FUNGI)
    assert out['taxon_rank'] == 'species'
    assert out['taxon_id'] == 48701
    assert out['common_name'] == 'Fly Agaric'
    assert out['taxon_name'] == 'Amanita muscaria'


def test_unresolvable_ancestry_falls_back_to_the_iconic_kingdom():
    # Nothing looked up, but the observation still knows it is a fungus.
    out = taxonomy.taxonomy_for(
        {'id': 1, 'name': 'Amanita muscaria', 'rank': 'species', 'ancestor_ids': CHAIN,
         'iconic_taxon_name': 'Fungi'}, {})
    assert out['kingdom'] == 'Fungi'
    assert out['species'] == 'Amanita muscaria'
    assert out['genus'] == 'Amanita'


def test_a_row_with_nothing_at_all_still_names_what_was_asked_for():
    out = taxonomy.taxonomy_for(None, {}, fallback_name='morchella')
    assert out['species'] == 'morchella'
    assert out['taxon_name'] == 'morchella'


def test_taxonomy_reaches_beyond_fungi():
    # The point of resolving ancestry rather than splitting a name is that it
    # works for any kingdom, which is what makes importing plants or insects
    # into the same dataset possible.
    plants = {
        47126: {'id': 47126, 'name': 'Plantae', 'rank': 'kingdom'},
        211194: {'id': 211194, 'name': 'Tracheophyta', 'rank': 'phylum'},
        47125: {'id': 47125, 'name': 'Magnoliopsida', 'rank': 'class'},
        47605: {'id': 47605, 'name': 'Fagales', 'rank': 'order'},
        50681: {'id': 50681, 'name': 'Fagaceae', 'rank': 'family'},
        47851: {'id': 47851, 'name': 'Quercus', 'rank': 'genus'},
        49009: {'id': 49009, 'name': 'Quercus gambelii', 'rank': 'species'},
    }
    chain = [47126, 211194, 47125, 47605, 50681, 47851]
    ranks = taxonomy.ranks_from_ancestry(
        {**plants[49009], 'ancestor_ids': chain}, plants)
    assert ranks['kingdom'] == 'Plantae'
    assert ranks['order'] == 'Fagales'
    assert ranks['species'] == 'Quercus gambelii'


class TestBatching:
    """The lookup half: batching, and surviving a failed batch."""

    def _fetcher(self, calls, fail_on=()):
        def fetch(ids):
            calls.append(list(ids))
            if tuple(ids) in fail_on:
                raise RuntimeError('boom')
            return {'results': [FUNGI[i] for i in ids if i in FUNGI]}
        return fetch

    def test_ids_are_requested_in_batches(self):
        import iNat
        calls = []
        ids = list(range(1, 71))
        iNat.fetch_taxa(ids, batch_size=30, fetcher=self._fetcher(calls))
        assert [len(c) for c in calls] == [30, 30, 10]
        # Sorted and de-duplicated, so the same id is never asked for twice.
        assert calls[0] == list(range(1, 31))

    def test_duplicate_ids_are_asked_for_once(self):
        import iNat
        calls = []
        iNat.fetch_taxa([5, 5, 5, 3, 3], batch_size=30, fetcher=self._fetcher(calls))
        assert calls == [[3, 5]]

    def test_a_failed_batch_costs_only_that_batch(self):
        import iNat
        calls = []
        # Ids are sorted before batching, so the first batch is the low pair.
        fetch = self._fetcher(calls, fail_on=((47169, 47170),))
        out = iNat.fetch_taxa([47170, 47169, 50814, 50815], batch_size=2, fetcher=fetch)
        # The second batch still landed.
        assert set(out) == {50814, 50815}

    def test_resolve_taxonomy_fills_rows_and_drops_the_raw_taxon(self):
        import iNat
        rows = [
            {'_taxon': taxon(48701), '_asked_for': 'amanita', 'species': 'Amanita muscaria'},
            {'_taxon': taxon(50814, ancestors=CHAIN[:-2]), '_asked_for': 'amanita',
             'species': 'Amanitaceae'},
        ]
        iNat.resolve_taxonomy(rows, fetcher=self._fetcher([]))
        assert '_taxon' not in rows[0] and '_asked_for' not in rows[0]
        assert rows[0]['family'] == 'Amanitaceae'
        assert rows[0]['genus'] == 'Amanita'
        assert rows[1]['family'] == 'Amanitaceae'
        assert rows[1]['genus'] is None

    def test_a_total_lookup_failure_leaves_the_existing_species_alone(self):
        # Resolution failing must never blank a column that already had a value.
        import iNat

        def boom(_ids):
            raise RuntimeError('network down')

        rows = [{'_taxon': taxon(48701), '_asked_for': 'amanita',
                 'species': 'Amanita muscaria'}]
        iNat.resolve_taxonomy(rows, fetcher=boom)
        assert rows[0]['species'] == 'Amanita muscaria'
        # The inline taxon still knows its own rank and name.
        assert rows[0]['taxon_rank'] == 'species'
        assert rows[0]['genus'] == 'Amanita'


@pytest.mark.parametrize('rank', taxonomy.RANKS)
def test_every_rank_is_a_column_the_app_can_group_by(rank):
    out = taxonomy.taxonomy_for(taxon(48701), FUNGI)
    assert rank in out
