"""The TracingThePast automatic person-matching algorithm.

A pure-Python library: plain data-set dicts in, scores and decisions out — no
database, no search engine, no web framework. See the README for the data-set
contract and the layers that sit on top of the raw engine.

Public surface:

* the engine — ``get_matching_score`` and its scoring constants / helpers
  (``normalize_string``, ``split_string_values``, ``get_names_as_dict``,
  ``convert_dates``, ``match_date_against_local_date``, ``corroborating_boost``,
  the ``*_MAX_SCORE_CONTRIBUTION`` constants, …), from
  :mod:`.automatic_matching_functions`;
* the pair-scoring layers and gates (``scored_pair``, ``identity_conflict``,
  ``pair_is_match``, ``distinctive_name_target``, ``nearest_pair``, …) and
  ``MatchingConfig``, from :mod:`.pair_scoring`;
* the forename-variant credit (``score_pair``, ``variant_key``,
  ``forenames_variant_linked``), from :mod:`.name_variants`;
* the cluster-resolution decision (``decide``, ``MatchDecision``,
  ``ScoredCandidate``, ``Action``, the ``REVIEW_*`` reasons), from
  :mod:`.matching_decision`.
"""

from .automatic_matching_functions import *  # noqa: F401,F403
from . import matching_decision, name_variants, pair_scoring  # noqa: F401
from .name_variants import *  # noqa: F401,F403
from .pair_scoring import *  # noqa: F401,F403
from .matching_decision import (  # noqa: F401
    Action,
    MatchDecision,
    REVIEW_AMBIGUOUS_IMPRECISE_DATE,
    REVIEW_NEAR_THRESHOLD,
    REVIEW_POTENTIAL_MERGE,
    ScoredCandidate,
    decide,
)
