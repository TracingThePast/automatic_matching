"""Forename-variant matching (the ForenameVariants dictionary).

The matching engine folds *orthographic/phonetic* variants well (Müller↔Mueller,
Levy↔Levi) but cannot know *cultural* nicknames (Friedrich↔Fritz), which aren't
phonetically similar. This module bridges that gap WITHOUT touching the engine:
it runs the engine normally, and only when the forenames failed to match
phonetically does it check a variant dictionary and, if linked, credit a
**discounted** forename score.

The functions here are pure: they depend only on the engine, so they unit-test
without any database. The caller supplies an already-built variant *index* — a
mapping from a normalised name to the set of variant-group ids it belongs to —
so building that index (e.g. from a database table) stays the caller's concern
and this module remains stack-neutral. See ``variant_key`` for how to normalise
names into index keys.
"""

from .automatic_matching_functions import (
    FORENAME_MAX_SCORE_CONTRIBUTION,
    TTP_MATCHING_DEFAULT_DISREGARD_VALUES,
    get_matching_score,
    normalize_string,
)

__all__ = ['variant_key', 'forenames_variant_linked', 'score_pair']


def variant_key(name):
    """Normalise a name to a dictionary key, using the same folding as the
    matcher (so casing/diacritics are irrelevant)."""
    return normalize_string(name or '')


def forenames_variant_linked(index, local_forenames, external_forenames):
    """Is any local↔external forename pair linked in the variant ``index``?

    ``index`` maps a normalised name to the set of variant-group ids it belongs
    to; two names are linked if their groups intersect. Returns
    ``(True, (local, external))`` for the first hit, else ``(False, None)``.
    """
    for local in local_forenames:
        local_groups = index.get(variant_key(local))
        if not local_groups:
            continue
        for external in external_forenames:
            external_groups = index.get(variant_key(external))
            if external_groups and (local_groups & external_groups):
                return True, (local, external)
    return False, None


def score_pair(local, external, *, variant_index=None, forename_variant_score=0.0,
               disregard=TTP_MATCHING_DEFAULT_DISREGARD_VALUES, with_edit_distances=False):
    """``get_matching_score`` plus an optional variant-dictionary boost.

    If the forenames matched only weakly but are dictionary-linked, the forename
    contribution is lifted to the discounted variant score and the totals are
    recomputed. The engine result is otherwise returned unchanged.

    ``variant_index`` is the mapping described in ``forenames_variant_linked``;
    ``forename_variant_score`` (0..1) is the discounted forename sub-score a
    dictionary-only match is credited (< 1.0 so it counts for less than a real
    or phonetic match). With either omitted, this is exactly ``get_matching_score``.
    """
    result = get_matching_score(local, external, disregard, with_edit_distances=with_edit_distances)

    if not variant_index or forename_variant_score <= 0:
        return result

    local_forenames = local.get('forenames') or []
    external_forenames = external.get('forenames') or []
    if not (local_forenames and external_forenames):
        return result

    forename_result = result.get('forename') or {}
    raw_absolute = forename_result.get('absolute_score', 0) or 0
    variant_absolute = forename_variant_score * FORENAME_MAX_SCORE_CONTRIBUTION

    # A variant match never downgrades a stronger real/phonetic match.
    if variant_absolute <= raw_absolute:
        return result

    linked, pair = forenames_variant_linked(variant_index, local_forenames, external_forenames)
    if not linked:
        return result

    result['absolute_score'] = result.get('absolute_score', 0) + (variant_absolute - raw_absolute)
    forename_result['absolute_score'] = variant_absolute
    forename_result['score'] = forename_variant_score
    forename_result['variant_matched'] = True
    forename_result['variant_pair'] = pair
    result['forename'] = forename_result
    result['variant_matched'] = True

    max_reachable = result.get('max_score_reachable', 0)
    result['relative_score'] = (result['absolute_score'] / max_reachable) if max_reachable else 0
    return result
