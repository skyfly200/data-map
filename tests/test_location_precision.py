import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from iNat import COARSE_ACCURACY_M, _coerce_accuracy, classify_location_precision


def test_obscured_wins_over_a_small_accuracy_number():
    # An obscured record can still carry the observer's own GPS accuracy. That
    # number describes where they stood, not the randomised point that was
    # published, so it must not be read as precision.
    assert classify_location_precision(True, None, None, 15) == 'obscured'
    assert classify_location_precision(True, 'open', None, 5) == 'obscured'


def test_geoprivacy_from_either_source_counts():
    # The observer can ask for obscuring, and iNaturalist applies it on its own
    # for threatened taxa. Either one means the point moved.
    assert classify_location_precision(False, 'obscured', None, 20) == 'obscured'
    assert classify_location_precision(False, None, 'obscured', 20) == 'obscured'


def test_private_is_treated_as_obscured():
    # A private location publishes no usable coordinate at all; whatever point
    # comes through is not to be trusted either.
    assert classify_location_precision(False, 'private', None, None) == 'obscured'
    assert classify_location_precision(False, None, 'private', None) == 'obscured'


def test_accuracy_splits_coarse_from_precise():
    assert classify_location_precision(False, 'open', None, COARSE_ACCURACY_M + 1) == 'coarse'
    assert classify_location_precision(False, 'open', None, COARSE_ACCURACY_M) == 'precise'
    assert classify_location_precision(False, None, None, 15) == 'precise'
    # An obscuring cell is ~20km, far above the threshold either way.
    assert classify_location_precision(False, 'open', None, 22000) == 'coarse'


def test_missing_accuracy_is_unknown_not_precise():
    # iNaturalist leaves accuracy unset often. Absence of evidence is not
    # evidence of a good fix, and a filter should be able to tell them apart.
    assert classify_location_precision(False, None, None, None) == 'unknown'
    assert classify_location_precision(False, 'open', 'open', None) == 'unknown'


def test_coerce_accuracy_rejects_junk_but_keeps_zero():
    assert _coerce_accuracy(None) is None
    assert _coerce_accuracy('') is None
    assert _coerce_accuracy('not a number') is None
    assert _coerce_accuracy(-5) is None
    # Zero is a real reading — a coordinate taken from a map pin — not missing.
    assert _coerce_accuracy(0) == 0
    assert _coerce_accuracy('25') == 25
    assert _coerce_accuracy(25.7) == 25
