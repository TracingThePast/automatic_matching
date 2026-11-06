"""Pure pair-scoring and match-quality gates layered on the engine.

These functions operate ONLY on the engine's plain data-set dicts (see the
README for the contract) and on the engine result dict — no database, no ES, no
framework. They are the SR matching layers lifted out of the application so the
whole algorithm, not just the raw engine, is reusable from any stack.

Three things sit on top of the engine's identity arithmetic here:

* the forename-variant credit (via :mod:`.name_variants`),
* the confirmed-individual birth-date penalty, and
* the near-miss corroborating-signal boost (via the engine's
  ``corroborating_boost``),

plus the gates that stop runaway single-linkage clusters (surname / place
conflicts) and the sparse-record fallbacks (distinctive name, imprecise date).

Everything configurable is carried on a :class:`MatchingConfig`; its defaults
reproduce the tuning the algorithm shipped with, and a caller overrides any of
them (e.g. an application passing its own settings) without touching the code.
"""

import math
import re
from dataclasses import dataclass

from .automatic_matching_functions import (
    CORROBORATING_SIGNALS,
    TTP_MATCHING_DEFAULT_DISREGARD_VALUES,
    corroborating_boost,
    get_names_as_dict,
    normalize_string,
)
from .name_variants import score_pair

__all__ = [
    'MatchingConfig', 'DEFAULT_CONFIG',
    'scored_pair',
    'field_conflict', 'surname_conflict', 'place_conflict', 'places_disjoint',
    'place_tokens', 'identity_conflict', 'pair_is_match', 'pair_is_cohesive',
    'cluster_is_cohesive',
    'birth_date_conflict', 'apply_confirmed_birth_penalty',
    'single_confident_date', 'has_imprecise_birth_date', 'distinct_precise_dates',
    'strong_name', 'date_blocks_distinctive_attach', 'distinctive_name_target',
    'nearest_pair',
]


@dataclass(frozen=True)
class MatchingConfig:
    """Tunable thresholds for the pair-scoring layers. Defaults reproduce the
    algorithm's shipped tuning; override any field to retune without a code change
    (an application typically builds one of these from its own configuration)."""

    # Forename sub-score (0..1) credited when two forenames match ONLY via the
    # variant dictionary (e.g. Friedrich↔Fritz) rather than phonetically. < 1.0 so a
    # known-variant match counts for less than a real/phonetic one.
    variant_forename_match_score: float = 0.6

    # Surname gate (engine surname sub-score, 0..1): when BOTH records carry a
    # surname they must agree at least this much to count as the same person.
    surname_match_floor: float = 0.3

    # Place vetoes (engine birth/death-place sub-score, in [-1, 1]). A place
    # disagreement vetoes a pair only when the sub-score is below this value AND the
    # two places share no common token (see ``places_disjoint``).
    birth_place_conflict_score: float = -0.7
    death_place_conflict_score: float = -0.7

    # Confirmed-individual birth-date gate. Two prior-vetted "confirmed individuals"
    # that EACH assert a single full birth date may still be one person only if the
    # dates reconcile within transcription range; a birth-date sub-score below this
    # floor is a genuine disagreement and incurs an explicit, published penalty.
    confirmed_individual_birth_date_min_score: float = -0.75
    confirmed_individual_birth_date_penalty: float = 100.0
    # The data-set ``record_type`` value that marks a confirmed individual. The gate
    # applies only when BOTH records carry this value.
    confirmed_individual_record_type: int = 1

    # Near-miss band for corroborating signals: a signal boost applies only when the
    # score sits in [min_review_score, min_automatch_score).
    min_review_score: float = 35
    min_automatch_score: float = 60

    # Corroborating-signal configuration (see the engine's ``corroborating_boost``).
    corroborating_signals: tuple = CORROBORATING_SIGNALS
    corroborating_place_match_min: float = 0.5
    corroborating_boost_cap: float = 10

    # Distinctive-name attach: a sparse record whose rare name unambiguously matches
    # ONE existing cluster may attach to it. ``*_min`` are the "strong name" sub-score
    # floors; ``min_cluster_margin`` is how far the leading cluster must beat the
    # runner-up; ``date_conflict_floor`` is the birth-date sub-score below which a
    # contradicting date blocks the attach.
    distinctive_name_autoattach: bool = True
    distinctive_name_min_cluster_margin: float = 10
    distinctive_name_forename_min: float = 0.85
    distinctive_name_surname_min: float = 0.85
    distinctive_name_date_conflict_floor: float = 0.0

    # Cluster-cohesion relative floor (0..1). A record may join a cluster past a
    # member it cannot out-score on the ABSOLUTE scale — because that member carries
    # little data, so the reachable score is small — as long as the pair agrees on
    # (near) ALL the data they share: ``relative_score`` (achieved / max reachable)
    # at or above this, with no conflicting field. Stops a low-information member from
    # walling off a richer record it fully agrees with. See ``pair_is_cohesive``.
    cohesion_relative_min: float = 0.9

    # Cluster-cohesion quorum (0..1): the fraction of a cluster's members a record
    # must POSITIVELY cohere with to join it (see ``cluster_is_cohesive``). Below 1.0
    # this is a strong-majority rule rather than strict complete-linkage, so a single
    # indeterminate member — no conflict, just under the bar (a sparse record, or a
    # transcription variant the cluster itself also holds) — no longer walls off a
    # record that agrees with the rest. A real conflict with ANY member still vetoes.
    cohesion_quorum: float = 0.6


DEFAULT_CONFIG = MatchingConfig()


# -- scoring ----------------------------------------------------------------

def scored_pair(local, external, *, variant_index=None, with_edit_distances=False,
                config=DEFAULT_CONFIG):
    """The single scoring entry point: engine score + forename-variant credit, the
    confirmed-individual birth-date penalty, and the near-miss corroborating boost —
    so every caller sees the same published score. ``with_edit_distances`` surfaces
    the (icu-costly) edit-distance readout on the result."""
    result = score_pair(
        local, external,
        variant_index=variant_index,
        forename_variant_score=config.variant_forename_match_score,
        with_edit_distances=with_edit_distances,
    )
    result = apply_confirmed_birth_penalty(result, local, external, config=config)
    # Corroboration runs last, on the fully-adjusted score, so its near-miss gate
    # sees the true published score. Positive-only; a no-op for confident matches,
    # clear non-matches, and records without the extra data.
    return corroborating_boost(
        result, local, external,
        near_miss_min=config.min_review_score,
        near_miss_max=config.min_automatch_score,
        signals=config.corroborating_signals,
        place_match_min=config.corroborating_place_match_min,
        cap=config.corroborating_boost_cap,
    )


# -- match-quality gates (prevent runaway single-linkage clusters) ----------

_FULL_DATE_RE = re.compile(r'^\s*\d{4}-\d{2}-\d{2}\s*$')


def field_conflict(result, field, floor):
    """True when BOTH records carry ``field`` and its engine sub-score is below
    ``floor`` — a positive disagreement. The engine only emits a ``score`` for a
    field when both sides have a value, so a value missing on either side is never a
    conflict."""
    part = result.get(field) or {}
    return 'score' in part and part['score'] < floor


def single_confident_date(values):
    """The record asserts EXACTLY ONE unambiguous date — a single complete
    ``YYYY-MM-DD`` with no alternatives. Multiple dates, a span (``<``/``>``) or a
    wildcard (``**``) all make the record less certain."""
    vals = [v for v in (values or []) if v and v.strip()]
    return len(vals) == 1 and bool(_FULL_DATE_RE.match(vals[0]))


def birth_date_conflict(result, local, external, *, config=DEFAULT_CONFIG):
    """Whether the confirmed-individual birth-date penalty applies to a pair: both
    records are confirmed individuals, each asserts a single full birth date, and the
    engine birth-date sub-score is a genuine disagreement (below the floor)."""
    if not (local and external):
        return False
    confirmed = config.confirmed_individual_record_type
    if local.get('record_type') != confirmed or external.get('record_type') != confirmed:
        return False
    if not (single_confident_date(local.get('birth_date'))
            and single_confident_date(external.get('birth_date'))):
        return False
    return field_conflict(result, 'birth_date', config.confirmed_individual_birth_date_min_score)


def apply_confirmed_birth_penalty(result, local, external, *, config=DEFAULT_CONFIG):
    """When ``birth_date_conflict`` holds, subtract an explicit, published penalty
    from the pair's score (rather than vetoing it out of band): a confirmed-individual
    birth-date disagreement is visible IN the score. The engine's own graded
    arithmetic is untouched."""
    if not birth_date_conflict(result, local, external, config=config):
        return result
    penalty = config.confirmed_individual_birth_date_penalty
    before = result.get('absolute_score', 0) or 0
    result['absolute_score'] = max(0.0, before - penalty)
    result['birth_date_penalty'] = -penalty
    result['birth_date_conflict'] = True
    max_reachable = result.get('max_score_reachable', 0)
    result['relative_score'] = (result['absolute_score'] / max_reachable) if max_reachable else 0
    return result


def surname_conflict(result, *, config=DEFAULT_CONFIG):
    """The surname case of ``field_conflict``."""
    return field_conflict(result, 'surname', config.surname_match_floor)


def date_blocks_distinctive_attach(result, *, config=DEFAULT_CONFIG):
    """Whether a birth-date conflict should block a distinctive-name attach. A
    positively-contradicting date normally does — UNLESS a corroborating signal fired
    for the pair (an already OCR/edit-gated distinctive identifier), which lifts the
    date block. Surname/place conflicts are handled separately and still stand."""
    return (field_conflict(result, 'birth_date', config.distinctive_name_date_conflict_floor)
            and not result.get('corroborating_signals'))


# The engine's tokeniser splits on ", " / "/ " (separator + space) but not a BARE
# "," or "/", so a comma-joined place like "Bechhofen,Mittelfranken" would stay one
# token — and a correct match that shares "Bechhofen" would be wrongly vetoed as a
# place conflict. Pre-split on the bare separators first.
_PLACE_SEP_RE = re.compile(r'[,;/.\-]')


def place_tokens(places):
    """Normalised place tokens (diacritics folded, split on separators), minus the
    engine's generic place stopwords (e.g. 'Deutsches Reich')."""
    expanded = [part for value in (places or []) for part in _PLACE_SEP_RE.split(value or '')]
    tokens = set(get_names_as_dict(expanded, False).keys())
    drop = {normalize_string(v) for v in TTP_MATCHING_DEFAULT_DISREGARD_VALUES.get('birth_place', [])}
    return tokens - drop


def places_disjoint(places_a, places_b):
    """The two place fields share no common normalised token. A shared token means
    they are compatible — one a more/less specific or differently-framed form of the
    other — so their low engine score is NOT a veto."""
    return not (place_tokens(places_a) & place_tokens(places_b))


def place_conflict(result, field, floor, places_a, places_b):
    """A PLACE veto: both records carry the place, the engine scores it strongly
    negative (< ``floor``, in [-1, 1]), AND the two places share no common token."""
    part = result.get(field) or {}
    if 'score' not in part or part['score'] >= floor:
        return False
    return places_disjoint(places_a, places_b)


def identity_conflict(result, local=None, external=None, *, config=DEFAULT_CONFIG):
    """A distinctive field the two records BOTH carry disagrees outright — strong
    evidence they are different people even when the additive total is high. Vetoes
    the pair. Surname: any sub-score below the floor. Birth/death PLACE: only a strong
    disagreement with NO shared token (needs the two records' place fields, so
    ``local``/``external`` are required for the place checks). Dates are never vetoed
    here (the confirmed-individual disagreement is an explicit scored penalty instead)."""
    if field_conflict(result, 'surname', config.surname_match_floor):
        return True
    if local is None or external is None:
        return False
    return (
        place_conflict(result, 'birth_place', config.birth_place_conflict_score,
                       local.get('birth_place'), external.get('birth_place'))
        or place_conflict(result, 'death_place', config.death_place_conflict_score,
                          local.get('death_place'), external.get('death_place'))
    )


def pair_is_match(result, min_score, local=None, external=None, *, config=DEFAULT_CONFIG):
    """A pair is the same person only if it clears ``min_score`` AND no distinctive
    field (surname / birth place / death place) actively conflicts."""
    return ((result.get('absolute_score', 0) or 0) >= min_score
            and not identity_conflict(result, local, external, config=config))


def pair_is_cohesive(result, min_score, local=None, external=None, *,
                     min_relative=None, config=DEFAULT_CONFIG):
    """Complete-linkage COHESION test — whether a record may sit in the same cluster
    as one existing member. Like :func:`pair_is_match` it vetoes any outright identity
    conflict (surname / birth place / death place).

    But cohesion must not let a member that merely carries LITTLE data wall off a
    richer record it fully agrees with: when the member is sparse the maximum
    reachable score is small, so even a perfect agreement can fall below ``min_score``
    through ABSENCE of data, not disagreement. So the pair also coheres when it agrees
    on (near) all the data the two records share — ``relative_score`` (the achieved
    score over the maximum reachable given the shared fields) at or above
    ``min_relative`` (default :attr:`MatchingConfig.cohesion_relative_min`) — with no
    conflicting field.

    This is looser than ``pair_is_match`` ONLY on the absolute-vs-relative axis; a
    surname/place conflict still vetoes, and a genuine field disagreement lowers the
    relative score (the conflicting field is reachable yet scores low), so it does not
    pass either."""
    if identity_conflict(result, local, external, config=config):
        return False
    if (result.get('absolute_score', 0) or 0) >= min_score:
        return True
    if min_relative is None:
        min_relative = config.cohesion_relative_min
    return (result.get('relative_score', 0) or 0) >= min_relative


def cluster_is_cohesive(external, member_datasets, min_score, *, variant_index=None,
                        min_relative=None, quorum=None, config=DEFAULT_CONFIG):
    """Whether a record (``external`` data-set) may JOIN a cluster given its existing
    members (``member_datasets``, each an engine data-set). Quorum cohesion, scoring
    the record against each member's OWN data:

    * a hard identity conflict (surname / birth place / death place) with ANY member
      VETOES — this is the anti-chaining guard (a Veen ↔ … ↔ van Klaveren chain can
      never form), and
    * the record must POSITIVELY cohere (:func:`pair_is_cohesive`) with at least
      ``quorum`` (a fraction, default :attr:`MatchingConfig.cohesion_quorum`) of the
      members.

    Members that neither conflict nor positively cohere — merely INDETERMINATE: no
    conflict, but just under the bar (a sparse record, or a transcription variant the
    cluster itself also holds) — are tolerated as the minority, so one of them cannot
    wall off a record that clearly belongs to the rest. ``ceil`` keeps a 1- or
    2-member cluster at full agreement. An empty cluster coheres trivially."""
    members = list(member_datasets)
    if not members:
        return True
    if quorum is None:
        quorum = config.cohesion_quorum
    cohere = 0
    for member in members:
        result = scored_pair(member, external, variant_index=variant_index, config=config)
        if identity_conflict(result, member, external, config=config):
            return False
        if pair_is_cohesive(result, min_score, member, external,
                            min_relative=min_relative, config=config):
            cohere += 1
    return cohere >= math.ceil(len(members) * quorum)


def strong_name(forename_score, surname_score, *, config=DEFAULT_CONFIG):
    """Forename AND surname both present and agreeing strongly — a real, near-exact
    name match. ``None`` on either side (a field missing on one record) is not strong."""
    return (
        forename_score is not None and surname_score is not None
        and forename_score >= config.distinctive_name_forename_min
        and surname_score >= config.distinctive_name_surname_min
    )


def has_imprecise_birth_date(birth_dates):
    """True when the record carries birth date(s) but NONE is an exact day — every
    value is a ``**``-wildcard or an open ``<``/``>`` bound. (No date at all is a
    different case, handled by the caller.)"""
    values = [str(v).strip() for v in (birth_dates or []) if str(v).strip()]
    if not values:
        return False
    return all('*' in v or v[:1] in ('<', '>') for v in values)


def distinct_precise_dates(birth_date_arrays):
    """The distinct exact-day (``YYYY-MM-DD``) dates across several birth_date arrays."""
    dates = set()
    for birth_dates in birth_date_arrays:
        for value in (birth_dates or []):
            value = (value or '').strip()
            if _FULL_DATE_RE.match(value):
                dates.add(value)
    return dates


def distinctive_name_target(scored_raw, cluster_map, *, config=DEFAULT_CONFIG):
    """The single existing cluster a sparse record may attach to on a strong,
    unambiguous name alone — or ``None``. Gated: the rule is enabled, the top
    already-formed cluster carries a strong-name match, and it leads the runner-up
    cluster's best score by ``distinctive_name_min_cluster_margin``.

    ``scored_raw`` is an iterable of dicts with ``record_id``, ``score``,
    ``forename_score`` and ``surname_score``; ``cluster_map`` maps a record id to its
    cluster id (or ``None`` when unaggregated). Unaggregated candidates are ignored:
    this only ever grows an existing cluster, never forms one from a bare name."""
    if not config.distinctive_name_autoattach:
        return None
    cluster_best = {}
    strong_clusters = set()
    for s in scored_raw:
        cluster_id = cluster_map.get(s['record_id'])
        if cluster_id is None:
            continue
        if s['score'] > cluster_best.get(cluster_id, float('-inf')):
            cluster_best[cluster_id] = s['score']
        if strong_name(s.get('forename_score'), s.get('surname_score'), config=config):
            strong_clusters.add(cluster_id)
    if not cluster_best:
        return None
    ranked = sorted(cluster_best.items(), key=lambda kv: kv[1], reverse=True)
    top_cluster, top_score = ranked[0]
    # The leading cluster must be the one that strongly matches the name…
    if top_cluster not in strong_clusters:
        return None
    # …and clear the runner-up cluster by a decisive margin (no runner-up ⇒ attaches).
    runner_up_score = ranked[1][1] if len(ranked) > 1 else float('-inf')
    if top_score - runner_up_score < config.distinctive_name_min_cluster_margin:
        return None
    return top_cluster


# -- proximity / next-closest ------------------------------------------------

def nearest_pair(local, candidate_datasets, *, exclude_id, config=DEFAULT_CONFIG):
    """The best-matching candidate for ``local``, as ``(best_id, best_score)`` — or
    ``(None, None)`` when there is no candidate. ``candidate_datasets`` is an iterable
    of ``(id, data_set)`` pairs; each is scored with ``scored_pair`` (the same score
    the matcher uses) and the single highest is returned. The next-closest/proximity
    score used for cluster comparison is this function applied to a cluster's
    neighbours. Pure — retrieval of the candidates stays the caller's concern."""
    best_id, best_score = None, None
    for cid, data_set in candidate_datasets:
        if cid == exclude_id:
            continue
        score = scored_pair(data_set, local, config=config)['absolute_score']
        if best_score is None or score > best_score:
            best_id, best_score = cid, score
    return best_id, best_score
