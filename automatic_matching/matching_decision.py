"""The pure cluster-resolution decision (ROADMAP Phase 3, Option A).

Given a new record's scored candidates — each a record with its similarity score
and the aggregated person it currently belongs to (if any) — decide what should
happen to the *new* record. This module is deliberately free of Django, ES and
I/O so the genuinely tricky logic can be unit-tested in isolation; the
orchestrator in ``matching.py`` does the DB/ES work around it.

The rule (see ROADMAP):

1. strong matches all in one aggregated person X  -> attach the new record to X
2. strong matches, none aggregated yet            -> create a new person spanning
                                                      the new record + those
3. no strong match                                -> the record stands alone
                                                      (its own single-record person)
4. strong matches span >= 2 aggregated persons    -> attach to the best, but FLAG
                                                      a potential merge; never
                                                      auto-merge the persons
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Action(str, Enum):
    ATTACH = 'attach'        # join an existing aggregated person
    CREATE = 'create'        # form a new person from the new record + matches
    STANDALONE = 'standalone'  # new person, just the record (no strong match)


# Why a record was flagged for human review.
REVIEW_NEAR_THRESHOLD = 'near_threshold'   # near misses, below the auto threshold
REVIEW_POTENTIAL_MERGE = 'potential_merge'  # strong matches across several persons
# An imprecise date (a wildcard / open bound) whose range spans several distinct
# persons: too ambiguous to auto-join any of them — left for a human to place.
REVIEW_AMBIGUOUS_IMPRECISE_DATE = 'ambiguous_imprecise_date'


@dataclass
class ScoredCandidate:
    record_id: int
    cluster_id: Optional[int]   # aggregated person id, or None if unaggregated
    score: float                # engine absolute score, 0–100
    variant_matched: bool = False  # the score relied on a name-variant link


@dataclass
class MatchDecision:
    action: Action
    target_cluster_id: Optional[int] = None     # existing person to attach to
    attach_record_ids: List[int] = field(default_factory=list)  # other unaggregated records to pull in
    top_score: float = 0.0
    flagged_for_review: bool = False
    review_reason: Optional[str] = None
    competing_cluster_ids: List[int] = field(default_factory=list)
    # Confidence of the match: gap to the runner-up candidate, and how many other
    # candidates there were. `score_margin` is None when there was no runner-up.
    score_margin: Optional[float] = None
    alternate_count: int = 0
    # The attach was granted by the distinctive-name rule (a rare, unambiguous name
    # standing in for the score a sparse record can't reach). The applier then
    # checks cohesion on the name rather than the numeric score.
    distinctive_name: bool = False
    # The attach was granted because the record fell just short of the auto bar but
    # ALL its candidates pointed to ONE existing person (no competing cluster). The
    # applier then checks cohesion at the lower single-cluster bar.
    near_threshold_single: bool = False


def decide(scored: List[ScoredCandidate], *, auto_threshold: float, review_threshold: float,
           single_cluster_threshold: Optional[float] = None) -> MatchDecision:
    strong = [c for c in scored if c.score >= auto_threshold]
    near = [c for c in scored if review_threshold <= c.score < auto_threshold]
    top_overall = max((c.score for c in scored), default=0.0)

    decision = _resolve(scored, strong, near, top_overall, single_cluster_threshold)

    # Annotate confidence: gap from the best candidate to the runner-up.
    ranked = sorted((c.score for c in scored), reverse=True)
    if len(ranked) >= 2:
        decision.score_margin = ranked[0] - ranked[1]
    decision.alternate_count = max(len(scored) - 1, 0)
    return decision


def _single_cluster_near_target(near, threshold):
    """The one existing person a near-miss record may attach to — or ``None``. The
    60 auto bar mainly guards against choosing between RIVAL persons; when every
    near candidate (review-band) points to the SAME cluster there is no rival, so a
    record that reaches ``threshold`` attaches to it. ``None`` when more than one
    cluster is in contention, none is aggregated, or the best falls short."""
    if threshold is None:
        return None
    clustered = [(c.cluster_id, c.score) for c in near if c.cluster_id is not None]
    if not clustered:
        return None
    clusters = {cid for cid, _ in clustered}
    if len(clusters) != 1:
        return None  # a competing cluster near the bar → ambiguous, leave for review
    if max(score for _, score in clustered) < threshold:
        return None
    return next(iter(clusters))


def _resolve(scored, strong, near, top_overall, single_cluster_threshold=None) -> MatchDecision:
    # 3. No confident match → stand alone (flag if there were near misses). But a
    #    record that fell just short AND whose candidates ALL point to ONE existing
    #    person is unambiguous — attach it rather than orphaning it.
    if not strong:
        target = _single_cluster_near_target(near, single_cluster_threshold)
        if target is not None:
            return MatchDecision(
                action=Action.ATTACH,
                target_cluster_id=target,
                top_score=top_overall,
                near_threshold_single=True,
            )
        return MatchDecision(
            action=Action.STANDALONE,
            top_score=top_overall,
            flagged_for_review=bool(near),
            review_reason=REVIEW_NEAR_THRESHOLD if near else None,
        )

    clusters = sorted({c.cluster_id for c in strong if c.cluster_id is not None})
    unaggregated = [c.record_id for c in strong if c.cluster_id is None]
    top_strong = max(c.score for c in strong)

    # 2. Strong matches but none aggregated yet → new person spanning them.
    if not clusters:
        return MatchDecision(
            action=Action.CREATE,
            attach_record_ids=unaggregated,
            top_score=top_strong,
        )

    # 1. All strong matches in a single existing person → attach to it (and pull
    #    in any unaggregated records that also matched strongly).
    if len(clusters) == 1:
        return MatchDecision(
            action=Action.ATTACH,
            target_cluster_id=clusters[0],
            attach_record_ids=unaggregated,
            top_score=top_strong,
        )

    # 4. Strong matches across several persons → attach to the best-scoring one
    #    and flag a potential merge; do NOT merge the persons automatically.
    return MatchDecision(
        action=Action.ATTACH,
        target_cluster_id=_best_cluster(strong),
        attach_record_ids=unaggregated,
        top_score=top_strong,
        flagged_for_review=True,
        review_reason=REVIEW_POTENTIAL_MERGE,
        competing_cluster_ids=clusters,
    )


def _best_cluster(strong: List[ScoredCandidate]) -> int:
    """The aggregated person whose strongest matching record scored highest."""
    best_by_cluster = {}
    for candidate in strong:
        if candidate.cluster_id is None:
            continue
        current = best_by_cluster.get(candidate.cluster_id)
        if current is None or candidate.score > current:
            best_by_cluster[candidate.cluster_id] = candidate.score
    # Highest score wins; ties broken by lowest cluster id for determinism.
    return min(best_by_cluster, key=lambda cid: (-best_by_cluster[cid], cid))
