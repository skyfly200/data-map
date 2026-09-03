"""Full taxonomic ranks for an observation.

An iNaturalist observation names one taxon and lists the ids of its ancestors,
but not their names. So the pipeline recorded a binomial and split it on the
space to get a genus, which works only when the identification happens to be at
species level and gives nothing above it. A record identified to *Amanita* had a
species of "Amanita" and a genus of "Amanita"; a record identified to family had
neither.

This resolves the ancestry properly: collect every taxon id in the fetched
observations, look up the ancestors once each, and read the ranks off the chain.
That makes kingdom, phylum, class, order, family, genus and species available to
filter, group, colour and analyse by — at whatever level a record was actually
identified to, rather than at the one level the name string happened to encode.

The lookup is separated from the parsing so the parsing can be tested without a
network: `ranks_from_ancestry` takes an already-resolved id → taxon map.
"""

from __future__ import annotations

# The ranks worth carrying as columns. iNaturalist emits many more — subgenus,
# section, tribe, subspecies, variety — but these seven are the ones people
# actually navigate by, and a column per intermediate rank would be mostly
# empty. Ordered coarse to fine, which is the order they are offered in the app.
RANKS = ('kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species')

# Ranks below species collapse into it. A record identified to a variety is a
# record of that species for every question this app asks, and giving
# subspecies its own column would split one species across several rows in
# every chart.
SUBSPECIFIC = ('subspecies', 'variety', 'form', 'hybrid', 'infrahybrid')


def _name_of(taxon):
    """The scientific name on a taxon record, whatever shape it arrived in."""
    if taxon is None:
        return None
    if isinstance(taxon, dict):
        name = taxon.get('name')
    else:
        name = getattr(taxon, 'name', None)
    name = (name or '').strip()
    return name or None


def _rank_of(taxon):
    if taxon is None:
        return None
    if isinstance(taxon, dict):
        rank = taxon.get('rank')
    else:
        rank = getattr(taxon, 'rank', None)
    return (rank or '').strip().lower() or None


def _ancestor_ids(taxon):
    """Ancestor ids, oldest first, from whichever field carries them.

    The v1 API gives `ancestor_ids` on a full taxon record and `ancestry` — the
    same chain as a slash-separated string — on a nested one. Reading both means
    an observation's own inline taxon is as usable as a looked-up one.
    """
    if taxon is None:
        return []
    get = taxon.get if isinstance(taxon, dict) else lambda k, d=None: getattr(taxon, k, d)

    ids = get('ancestor_ids') or []
    if not ids:
        ancestry = get('ancestry')
        if ancestry:
            ids = [part for part in str(ancestry).split('/') if part]

    out = []
    for value in ids:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return out


def taxon_ids_in(observations):
    """Every taxon id that has to be looked up to resolve these observations.

    The union of each observation's own taxon and its ancestors. A dataset of
    48,000 records typically spans a few hundred taxa, so resolving the union
    once is a few requests rather than one per record.
    """
    wanted = set()
    for obs in observations:
        taxon = obs.get('taxon') if isinstance(obs, dict) else None
        if taxon is None:
            continue
        taxon_id = taxon.get('id') if isinstance(taxon, dict) else getattr(taxon, 'id', None)
        if taxon_id is not None:
            try:
                wanted.add(int(taxon_id))
            except (TypeError, ValueError):
                pass
        wanted.update(_ancestor_ids(taxon))
    return wanted


def ranks_from_ancestry(taxon, by_id=None):
    """The seven ranks for one observation's taxon.

    `by_id` maps taxon id → taxon record (dict or model), and supplies the names
    the observation itself does not carry. Missing entries are simply absent
    from the result rather than raising: a partially-resolved chain still yields
    every rank it did resolve, which is better than dropping the lot because one
    lookup failed.

    The taxon's own name is placed at its own rank, so a record identified only
    to family fills `family` and leaves `genus` and `species` empty — the honest
    answer, where splitting the name string invented a genus.
    """
    by_id = by_id or {}
    out = {rank: None for rank in RANKS}
    if taxon is None:
        return out

    chain = [by_id.get(tid) for tid in _ancestor_ids(taxon)]
    chain.append(taxon)

    for node in chain:
        if node is None:
            continue
        rank = _rank_of(node)
        name = _name_of(node)
        if not rank or not name:
            continue
        if rank in SUBSPECIFIC:
            # A subspecies name is "Genus species subspecies"; the species is
            # its first two words. Its parent species is normally in the chain
            # anyway, so only fill in what the chain left empty.
            if not out['species']:
                parts = name.split()
                out['species'] = ' '.join(parts[:2]) if len(parts) >= 2 else name
            continue
        if rank in out:
            out[rank] = name

    # A species binomial always starts with its genus, so a chain that resolved
    # the species but not the genus can still fill it in.
    if out['species'] and not out['genus']:
        first = out['species'].split()
        if len(first) >= 2:
            out['genus'] = first[0]

    return out


def taxonomy_for(taxon, by_id=None, fallback_name=None):
    """Every taxonomy column for one observation.

    Adds the identification's own rank and id alongside the seven names, because
    "identified to family" is itself worth filtering on — it is the difference
    between a record that could not be pinned down and one that was.
    """
    ranks = ranks_from_ancestry(taxon, by_id)
    rank = _rank_of(taxon)
    name = _name_of(taxon)

    taxon_id = None
    common = None
    if taxon is not None:
        get = taxon.get if isinstance(taxon, dict) else lambda k, d=None: getattr(taxon, k, d)
        taxon_id = get('id')
        common = (get('preferred_common_name') or '').strip() or None
        if not ranks['kingdom']:
            # iconic_taxon_name is the coarse group iNaturalist files a record
            # under and is present even on a nested taxon, so it stands in when
            # the ancestry could not be resolved at all.
            iconic = (get('iconic_taxon_name') or '').strip()
            if iconic in ('Fungi', 'Plantae', 'Animalia', 'Protozoa', 'Chromista', 'Bacteria',
                          'Archaea', 'Viruses'):
                ranks['kingdom'] = iconic

    # Nothing resolved and nothing to resolve from: fall back to the name the
    # fetch asked for, so a row is still attributable rather than blank.
    if not any(ranks.values()) and (name or fallback_name):
        ranks['species'] = name or fallback_name

    return {
        **ranks,
        'taxon_id': taxon_id,
        'taxon_rank': rank,
        'taxon_name': name or fallback_name,
        'common_name': common,
    }
