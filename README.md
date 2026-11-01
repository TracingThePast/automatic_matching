# automatic_matching

A pure-Python algorithm for linking references to the **same person** across
multiple independent databases, from a handful of identity fields plus optional
corroborating signals. Plain data in, scores and decisions out — no database, no
search engine, no web framework. It is the person-matching core of
[Tracing the Past](https://www.tracingthepast.org/)'s *Shared Remembrance*,
extracted so it can be reused, inspected and tested on any stack.

- **Engine** — scores one record against another across six identity fields,
  with transliteration/phonetic folding, fuzzy and imprecise date handling, and
  a graded 0–100 score.
- **Layers on top of the engine** — a forename-variant credit (cultural
  nicknames the phonetics miss), a confirmed-individual birth-date penalty, and a
  near-miss corroborating-signal boost (a shared prisoner number, a residence).
- **Gates** — surname/place conflict vetoes that stop runaway single-linkage
  clusters, plus sparse-record fallbacks (distinctive name, imprecise date).
- **Decision** — a pure cluster-resolution step (attach / create / stand-alone,
  with review flags).

MIT licensed.

## Install

```bash
pip install git+https://github.com/TracingThePast/automatic_matching@v3.0.0
```

PyICU needs the system ICU development libraries at build time (e.g.
`apt-get install libicu-dev pkg-config` on Debian/Ubuntu). The other
dependencies (`numpy`, `python-Levenshtein`, `pyxdameraulevenshtein`,
`rapidfuzz`, `doublemetaphone`, `python-dateutil`) are pure wheels.

## The data-set contract

Every function takes one or two **data-set dicts**. This documented shape is what
makes the algorithm stack-neutral: build it from a Django model, a CSV row, an
API payload — anything. All fields are **lists of strings** and all are optional
(a missing field simply doesn't contribute and is never counted as a conflict).

```python
data_set = {
    # The six identity fields (lists, so a person can carry several spellings,
    # a maiden + married surname, a birth place and its region, …):
    'forenames':   ['Anna', 'Anne'],
    'surnames':    ['Musterfrau', 'Levy'],
    'birth_place': ['München', 'Bayern'],
    'birth_date':  ['1900-01-01'],
    'death_place': ['Dachau'],
    'death_date':  ['1944-05-02'],
}
```

**Dates** are ISO `YYYY-MM-DD`. Two kinds of imprecision are understood:

- a wildcard month/day — `'1912-**-**'` (year known, rest unknown);
- an open bound — `'<1910-01-01'` or `'>1926-01-01'` (only the year is used).

**Corroborating keys** are distinctive non-identity data points the six fields
never see. They live on the same dict under the key each signal is configured
with (see `MatchingConfig.corroborating_signals`); the defaults read:

```python
data_set['prisoner_number']       = ['12345']        # OCR-tolerant digit compare
data_set['place_of_imprisonment'] = ['Auschwitz']    # a guard for prisoner_number
data_set['residence_place']       = ['Wien']         # place-style compare
```

A seventh key, `record_type`, is **not scored** — it only marks "confirmed
individuals" for the birth-date penalty (its value is
`MatchingConfig.confirmed_individual_record_type`, default `1`).

## Quick start

### Score a pair

```python
from automatic_matching import get_matching_score

local =    {'forenames': ['Anna'], 'surnames': ['Levy'], 'birth_date': ['1900-01-01']}
external = {'forenames': ['Anna'], 'surnames': ['Levi'], 'birth_date': ['1900-01-01']}

result = get_matching_score(local, external)
result['absolute_score']    # 0–100, the engine's graded similarity
result['relative_score']    # absolute / max reachable given the fields present
result['surname']['score']  # per-field sub-score, −1…1
```

### Score with the full algorithm

`scored_pair` is the single entry point that layers the forename-variant credit,
the confirmed-individual birth-date penalty and the near-miss corroborating boost
on top of the engine — the same published score every caller should use:

```python
from automatic_matching import scored_pair, MatchingConfig

# A variant index maps a normalised name to the set of variant-group ids it is in
# (build it however you like — a DB table, a JSON file). Friedrich ↔ Fritz here:
from automatic_matching import variant_key
index = {variant_key('Friedrich'): frozenset({1}), variant_key('Fritz'): frozenset({1})}

result = scored_pair(local, external, variant_index=index)
```

### Configuration is a parameter, not a global

Every threshold lives on an immutable `MatchingConfig`; the defaults reproduce the
algorithm's shipped tuning. Override any field to retune — no code change, no
environment coupling:

```python
cfg = MatchingConfig(
    min_automatch_score=65,      # the confident-match bar
    surname_match_floor=0.4,     # how much two surnames must agree
    variant_forename_match_score=0.5,
)
result = scored_pair(local, external, config=cfg)
```

An application typically builds one `MatchingConfig` from its own settings and
threads it through.

### Resolve a cluster

Given a new record's scored candidates — each the record's similarity score and
the aggregated person it currently belongs to — `decide` returns what should
happen to the new record (pure; the caller does the database work around it):

```python
from automatic_matching import ScoredCandidate, decide

scored = [
    ScoredCandidate(record_id=1, cluster_id=10, score=75.0),
    ScoredCandidate(record_id=2, cluster_id=10, score=70.0),
]
decision = decide(scored, auto_threshold=60, review_threshold=35)
decision.action            # Action.ATTACH / CREATE / STANDALONE
decision.target_cluster_id # 10
```

## Modules

| Module | What it holds |
| --- | --- |
| `automatic_matching_functions` | the engine: `get_matching_score`, `corroborating_boost`, normalisation/date helpers, the `*_MAX_SCORE_CONTRIBUTION` constants |
| `pair_scoring` | `MatchingConfig`, `scored_pair`, the conflict gates (`identity_conflict`, `pair_is_match`, `surname_conflict`, `place_conflict`, …), the sparse-record fallbacks (`distinctive_name_target`, `has_imprecise_birth_date`), and the proximity selector `nearest_pair` |
| `name_variants` | the forename-variant credit: `variant_key`, `forenames_variant_linked`, `score_pair` (takes an already-built variant index) |
| `matching_decision` | the pure cluster-resolution `decide` and its `MatchDecision` / `ScoredCandidate` / `Action` types |

The algorithm's scoring behaviour is versioned independently of the package
release in `AUTOMATIC_MATCHING_ALGORITHM_VERSION_STRING`.

## Tests

```bash
python -m unittest discover -s automatic_matching/tests
```

The suite pins the engine's score contract and the behaviour of every lifted
layer, so the algorithm cannot drift silently.
