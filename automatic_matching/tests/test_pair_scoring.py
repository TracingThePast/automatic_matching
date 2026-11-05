"""Match-quality gates and the scoring entry point (pure, no DB/ES/framework).

Ported from Shared Remembrance's gate tests: ``override_settings(...)`` becomes a
``MatchingConfig(...)`` passed as ``config=``, and record objects become plain
data-set dicts / value lists.

    python -m unittest automatic_matching.tests.test_pair_scoring
"""

import unittest

from automatic_matching.pair_scoring import (
    DEFAULT_CONFIG,
    MatchingConfig,
    apply_confirmed_birth_penalty,
    date_blocks_distinctive_attach,
    distinct_precise_dates,
    distinctive_name_target,
    field_conflict,
    has_imprecise_birth_date,
    identity_conflict,
    nearest_pair,
    pair_is_match,
    pair_is_cohesive,
    place_tokens,
    places_disjoint,
    scored_pair,
    single_confident_date,
    strong_name,
    surname_conflict,
)
from automatic_matching import get_matching_score


def _ds(**places):
    """A minimal engine data-set carrying just the place fields under test."""
    return {'birth_place': places.get('birth', []), 'death_place': places.get('death', [])}


class SurnameConflictTests(unittest.TestCase):
    def test_both_surnames_disagree_is_a_conflict(self):
        self.assertTrue(surname_conflict({'surname': {'score': 0.0}}))
        self.assertTrue(surname_conflict({'surname': {'score': 0.1}}))

    def test_agreeing_surnames_are_not_a_conflict(self):
        self.assertFalse(surname_conflict({'surname': {'score': 0.9}}))
        self.assertFalse(surname_conflict({'surname': {'score': 0.5}}))

    def test_missing_surname_is_never_a_conflict(self):
        self.assertFalse(surname_conflict({'surname': {}}))
        self.assertFalse(surname_conflict({}))

    def test_floor_is_configurable(self):
        cfg = MatchingConfig(surname_match_floor=0.5)
        self.assertTrue(surname_conflict({'surname': {'score': 0.4}}, config=cfg))
        self.assertFalse(surname_conflict({'surname': {'score': 0.6}}, config=cfg))


class IdentityConflictTests(unittest.TestCase):
    # score in [-1, 1]; the place veto needs BOTH a strong negative AND no shared token.

    def test_strong_disagreement_no_shared_token_is_a_conflict(self):
        self.assertTrue(identity_conflict(
            {'birth_place': {'score': -1.0}},
            _ds(birth=['München']), _ds(birth=['Fulda', '-', 'Hessen-Nassau']),
        ))

    def test_shared_token_is_never_a_conflict_even_when_strongly_negative(self):
        self.assertFalse(identity_conflict(
            {'birth_place': {'score': -0.79}},
            _ds(birth=['Markt', 'Bechhofen']),
            _ds(birth=['Bechhofen,Mittelfranken', 'Bayern']),
        ))

    def test_granularity_difference_sharing_a_token_is_not_a_conflict(self):
        self.assertFalse(identity_conflict(
            {'birth_place': {'score': -0.5}},
            _ds(birth=['Berlin']), _ds(birth=['Schöneberg', 'Stadt Berlin']),
        ))

    def test_mild_disagreement_is_left_to_the_graded_total(self):
        self.assertFalse(identity_conflict(
            {'birth_place': {'score': -0.38}},
            _ds(birth=['Custrin', 'Berlin']), _ds(birth=['Küstrin', 'Königsberg']),
        ))

    def test_missing_place_is_never_a_conflict(self):
        self.assertFalse(identity_conflict({'birth_place': {}}, _ds(), _ds(birth=['Berlin'])))

    def test_surname_conflict_covered_without_datasets(self):
        self.assertTrue(identity_conflict({'surname': {'score': 0.0}}))

    def test_floor_is_configurable(self):
        result = {'birth_place': {'score': -0.5}}
        a, b = _ds(birth=['München']), _ds(birth=['Fulda'])
        self.assertTrue(identity_conflict(result, a, b, config=MatchingConfig(birth_place_conflict_score=-0.4)))
        self.assertFalse(identity_conflict(result, a, b, config=MatchingConfig(birth_place_conflict_score=-0.9)))


class PlaceTokenTests(unittest.TestCase):
    def test_shared_token_not_disjoint(self):
        self.assertFalse(places_disjoint(['Markt', 'Bechhofen'], ['Bechhofen,Mittelfranken']))

    def test_disjoint_places(self):
        self.assertTrue(places_disjoint(['München'], ['Fulda']))

    def test_place_tokens_fold_and_split(self):
        # The comma-joined place splits into two phonetically-folded tokens, and the
        # 'Bechhofen' token matches across both framings (doubled consonants collapse).
        tokens = place_tokens(['Bechhofen,Mittelfranken'])
        self.assertEqual(len(tokens), 2)
        self.assertTrue(tokens & place_tokens(['Markt', 'Bechhofen']))


class PairIsMatchTests(unittest.TestCase):
    def test_strong_score_and_agreeing_surname_matches(self):
        self.assertTrue(pair_is_match({'absolute_score': 70, 'surname': {'score': 0.9}}, 60))

    def test_surname_conflict_blocks_even_a_high_score(self):
        self.assertFalse(pair_is_match({'absolute_score': 75, 'surname': {'score': 0.0}}, 60))

    def test_birth_place_conflict_blocks_even_a_high_score(self):
        self.assertFalse(pair_is_match(
            {'absolute_score': 90, 'surname': {'score': 1.0}, 'birth_place': {'score': -1.0}}, 60,
            _ds(birth=['Külsheim', 'Tauberbischofsheim', 'Baden']),
            _ds(birth=['Witzenhausen', 'Witzenhausen', 'Hessen-Nassau']),
        ))

    def test_shared_place_token_still_matches_on_a_strong_score(self):
        self.assertTrue(pair_is_match(
            {'absolute_score': 85, 'surname': {'score': 1.0}, 'birth_place': {'score': -0.79}}, 60,
            _ds(birth=['Markt', 'Bechhofen']), _ds(birth=['Bechhofen,Mittelfranken', 'Bayern']),
        ))

    def test_below_threshold_is_not_a_match(self):
        self.assertFalse(pair_is_match({'absolute_score': 50, 'surname': {'score': 0.9}}, 60))

    def test_missing_surname_and_place_still_matches_on_score(self):
        self.assertTrue(pair_is_match({'absolute_score': 65}, 60))


class PairIsCohesiveTests(unittest.TestCase):
    def test_clears_the_absolute_bar_like_pair_is_match(self):
        self.assertTrue(pair_is_cohesive(
            {'absolute_score': 70, 'relative_score': 0.7, 'surname': {'score': 0.9}}, 60))

    def test_sparse_member_with_total_relative_agreement_coheres(self):
        # Absolute 50 (< 60) only because the member is sparse (max reachable 50), but
        # relative_score 1.0 — perfect agreement on the data they share. Coheres.
        self.assertTrue(pair_is_cohesive(
            {'absolute_score': 50, 'relative_score': 1.0, 'surname': {'score': 1.0}}, 60))

    def test_low_relative_agreement_below_the_bar_does_not_cohere(self):
        # Low absolute AND only partial agreement on shared data → not cohesive.
        self.assertFalse(pair_is_cohesive(
            {'absolute_score': 40, 'relative_score': 0.5, 'surname': {'score': 0.9}}, 60))

    def test_surname_conflict_blocks_even_at_full_relative_agreement(self):
        # A real conflict vetoes regardless of relative_score.
        self.assertFalse(pair_is_cohesive(
            {'absolute_score': 50, 'relative_score': 1.0, 'surname': {'score': 0.0}}, 60))

    def test_relative_floor_is_configurable(self):
        res = {'absolute_score': 50, 'relative_score': 0.85, 'surname': {'score': 1.0}}
        self.assertFalse(pair_is_cohesive(res, 60))                       # default 0.9
        self.assertTrue(pair_is_cohesive(res, 60, min_relative=0.8))      # loosened


def _cids(birth, rt=1):
    """A confirmed-individual (rt=1) engine data-set carrying a birth date."""
    return {'birth_place': [], 'death_place': [], 'birth_date': birth, 'record_type': rt}


def _result(birth_score, absolute=60, max_reachable=100):
    return {'absolute_score': absolute, 'max_score_reachable': max_reachable,
            'birth_date': {'score': birth_score}}


class ConfirmedIndividualBirthDatePenaltyTests(unittest.TestCase):
    def test_strong_full_date_mismatch_is_penalised(self):
        r = apply_confirmed_birth_penalty(_result(-1.0, absolute=60),
                                          _cids(['1920-12-19']), _cids(['1923-01-17']))
        self.assertEqual(r['absolute_score'], 0.0)
        self.assertEqual(r['birth_date_penalty'], -100.0)
        self.assertTrue(r['birth_date_conflict'])

    def test_transcription_range_is_not_penalised(self):
        for score, a, b in ((0.86, ['1922-11-12'], ['1922-12-11']),
                            (0.5, ['1920-10-14'], ['1920-10-11'])):
            r = apply_confirmed_birth_penalty(_result(score, absolute=60), _cids(a), _cids(b))
            self.assertEqual(r['absolute_score'], 60)
            self.assertNotIn('birth_date_penalty', r)

    def test_only_confirmed_vs_confirmed(self):
        r = apply_confirmed_birth_penalty(_result(-1.0, absolute=60),
                                          _cids(['1920-12-19']), _cids(['1923-01-17'], rt=0))
        self.assertEqual(r['absolute_score'], 60)

    def test_both_sides_need_a_single_full_date(self):
        for a, b in ((['1920-**-**'], ['1923-01-17']),
                     (['>1919'], ['1923-01-17']),
                     (['1920-12-19', '1921-05-03'], ['1923-01-17'])):
            r = apply_confirmed_birth_penalty(_result(-1.0, absolute=60), _cids(a), _cids(b))
            self.assertEqual(r['absolute_score'], 60)

    def test_missing_score_is_never_penalised(self):
        r = apply_confirmed_birth_penalty({'absolute_score': 60}, _cids(['1920-12-19']), _cids(['1923-01-17']))
        self.assertEqual(r['absolute_score'], 60)

    def test_penalty_is_configurable(self):
        cfg = MatchingConfig(confirmed_individual_birth_date_penalty=30)
        r = apply_confirmed_birth_penalty(_result(-1.0, absolute=60),
                                          _cids(['1920-12-19']), _cids(['1923-01-17']), config=cfg)
        self.assertEqual(r['absolute_score'], 30)

    def test_floor_is_configurable(self):
        cfg = MatchingConfig(confirmed_individual_birth_date_min_score=0.6)
        r = apply_confirmed_birth_penalty(_result(0.5, absolute=60),
                                          _cids(['1920-10-14']), _cids(['1920-10-11']), config=cfg)
        self.assertTrue(r.get('birth_date_conflict'))


class SingleConfidentDateTests(unittest.TestCase):
    def test_single_full_date(self):
        self.assertTrue(single_confident_date(['1920-05-05']))

    def test_rejects_fuzzy_multi_or_empty(self):
        self.assertFalse(single_confident_date(['1920-**-**']))
        self.assertFalse(single_confident_date(['1920-05-05', '1921-01-01']))
        self.assertFalse(single_confident_date([]))
        self.assertFalse(single_confident_date(['>1919']))


class ImpreciseBirthDateTests(unittest.TestCase):
    def test_wildcard_only_is_imprecise(self):
        self.assertTrue(has_imprecise_birth_date(['1926-**-**']))

    def test_open_bounds_are_imprecise(self):
        self.assertTrue(has_imprecise_birth_date(['>1926-01-01']))
        self.assertTrue(has_imprecise_birth_date(['<1930-01-01']))

    def test_full_date_is_precise(self):
        self.assertFalse(has_imprecise_birth_date(['1926-12-15']))

    def test_any_full_date_makes_it_precise(self):
        self.assertFalse(has_imprecise_birth_date(['1926-**-**', '1926-12-15']))

    def test_no_date_is_not_flagged(self):
        self.assertFalse(has_imprecise_birth_date([]))


class ContendingPreciseDatesTests(unittest.TestCase):
    def test_one_shared_precise_date_softens(self):
        dates = distinct_precise_dates([['1902-05-10'], ['1902-05-10', '1902-**-**']])
        self.assertEqual(dates, {'1902-05-10'})

    def test_conflicting_precise_dates_keep_guard(self):
        self.assertEqual(len(distinct_precise_dates([['1902-05-10'], ['1910-03-02']])), 2)

    def test_no_precise_date_keeps_guard(self):
        self.assertEqual(distinct_precise_dates([['1902-**-**'], ['<1945-05-08']]), set())

    def test_only_full_days_counted(self):
        self.assertEqual(distinct_precise_dates([['1902', '1902-05-10', '>1900-01-01']]), {'1902-05-10'})


class DistinctiveNameDateConflictTests(unittest.TestCase):
    FLOOR = DEFAULT_CONFIG.distinctive_name_date_conflict_floor  # 0.0

    def test_contradicting_date_blocks(self):
        self.assertTrue(field_conflict({'birth_date': {'score': -1.0}}, 'birth_date', self.FLOOR))
        self.assertTrue(field_conflict({'birth_date': {'score': -0.3}}, 'birth_date', self.FLOOR))

    def test_agreeing_or_partial_date_allows(self):
        self.assertFalse(field_conflict({'birth_date': {'score': 0.5}}, 'birth_date', self.FLOOR))
        self.assertFalse(field_conflict({'birth_date': {'score': 1.0}}, 'birth_date', self.FLOOR))
        self.assertFalse(field_conflict({'birth_date': {'score': 0.0}}, 'birth_date', self.FLOOR))

    def test_missing_date_allows(self):
        self.assertFalse(field_conflict({}, 'birth_date', self.FLOOR))
        self.assertFalse(field_conflict({'birth_date': {}}, 'birth_date', self.FLOOR))

    def test_corroborating_signal_overrides_the_date_block(self):
        conflict = {'birth_date': {'score': -0.3}}
        self.assertTrue(date_blocks_distinctive_attach(conflict))
        conflict_corroborated = {
            'birth_date': {'score': -0.3},
            'corroborating_signals': [{'field': 'prisoner_number', 'boost': 10}],
        }
        self.assertFalse(date_blocks_distinctive_attach(conflict_corroborated))

    def test_no_conflict_never_blocks(self):
        self.assertFalse(date_blocks_distinctive_attach({'birth_date': {'score': 0.5}}))
        self.assertFalse(date_blocks_distinctive_attach({}))


class StrongNameTests(unittest.TestCase):
    def test_both_strong(self):
        self.assertTrue(strong_name(0.95, 0.9))

    def test_weak_or_missing_not_strong(self):
        self.assertFalse(strong_name(0.95, 0.2))
        self.assertFalse(strong_name(None, 0.95))
        self.assertFalse(strong_name(0.95, None))

    def test_floors_configurable(self):
        cfg = MatchingConfig(distinctive_name_forename_min=0.5, distinctive_name_surname_min=0.5)
        self.assertTrue(strong_name(0.6, 0.6, config=cfg))


def _cand(record_id, *, score, strong=True):
    sub = 0.95 if strong else 0.2
    return {'record_id': record_id, 'score': score, 'forename_score': sub, 'surname_score': sub}


class DistinctiveNameTargetTests(unittest.TestCase):
    def test_lone_clear_winner_attaches(self):
        scored = [_cand(1, score=50), _cand(2, score=1.9, strong=False)]
        self.assertEqual(distinctive_name_target(scored, {1: 'A', 2: 'B'}), 'A')

    def test_unopposed_cluster_attaches(self):
        self.assertEqual(distinctive_name_target([_cand(1, score=50)], {1: 'A'}), 'A')

    def test_unclustered_strong_candidates_are_ignored(self):
        scored = [_cand(1, score=50), _cand(2, score=50)]
        self.assertEqual(distinctive_name_target(scored, {1: 'A', 2: None}), 'A')

    def test_near_tie_between_two_clusters_is_left_for_review(self):
        scored = [_cand(1, score=50), _cand(2, score=48)]
        self.assertIsNone(distinctive_name_target(scored, {1: 'A', 2: 'B'}))

    def test_leader_must_be_a_strong_name(self):
        scored = [_cand(1, score=55, strong=False), _cand(2, score=20)]
        self.assertIsNone(distinctive_name_target(scored, {1: 'A', 2: 'B'}))

    def test_margin_is_configurable(self):
        scored = [_cand(1, score=50), _cand(2, score=30)]  # gap 20
        cmap = {1: 'A', 2: 'B'}
        self.assertEqual(distinctive_name_target(scored, cmap, config=MatchingConfig(distinctive_name_min_cluster_margin=15)), 'A')
        self.assertIsNone(distinctive_name_target(scored, cmap, config=MatchingConfig(distinctive_name_min_cluster_margin=25)))

    def test_disabled_returns_none(self):
        self.assertIsNone(distinctive_name_target([_cand(1, score=50)], {1: 'A'},
                                                  config=MatchingConfig(distinctive_name_autoattach=False)))

    def test_no_formed_clusters_returns_none(self):
        scored = [_cand(1, score=50), _cand(2, score=50)]
        self.assertIsNone(distinctive_name_target(scored, {1: None, 2: None}))


class ScoredPairCorroborationTests(unittest.TestCase):
    def _ds(self, **extra):
        base = {'forenames': ['Anna'], 'surnames': ['Nowak']}  # names-only → absolute 50 (near miss)
        base.update(extra)
        return base

    def test_shared_prisoner_number_lifts_near_miss(self):
        local = self._ds(prisoner_number='12345', place_of_imprisonment='Buchenwald')
        external = self._ds(prisoner_number='12345', place_of_imprisonment='Buchenwald')
        baseline = scored_pair(self._ds(), self._ds())['absolute_score']
        boosted = scored_pair(local, external)
        self.assertEqual(boosted['absolute_score'], baseline + 5)
        self.assertEqual(boosted['corroborating_boost'], 5)

    def test_no_shared_data_no_boost(self):
        self.assertNotIn('corroborating_boost', scored_pair(self._ds(), self._ds()))

    def test_confident_pair_is_not_boosted(self):
        complete = {
            'forenames': ['Anna'], 'surnames': ['Nowak'],
            'birth_date': ['1920-04-15'], 'birth_place': ['Warschau'],
            'prisoner_number': '12345', 'place_of_imprisonment': 'Buchenwald',
        }
        r = scored_pair(dict(complete), dict(complete))
        self.assertGreaterEqual(r['absolute_score'], DEFAULT_CONFIG.min_automatch_score)
        self.assertNotIn('corroborating_boost', r)

    def test_plain_pair_equals_engine_when_no_adjustment(self):
        # No variant index, no corroborating data, not confirmed individuals → scored_pair
        # must equal the raw engine score.
        a = {'forenames': ['Anna'], 'surnames': ['Nowak'], 'birth_date': ['1920-04-15']}
        b = {'forenames': ['Anna'], 'surnames': ['Nowak'], 'birth_date': ['1920-04-15']}
        self.assertEqual(scored_pair(dict(a), dict(b))['absolute_score'],
                         get_matching_score(dict(a), dict(b))['absolute_score'])


class NearestPairTests(unittest.TestCase):
    """The cross-cluster neighbour selector: pick the highest-scoring OTHER person,
    never the subject itself. Pure — scores via ``scored_pair``."""

    LOCAL = {'forenames': ['Anna'], 'surnames': ['Nowak'], 'birth_date': ['1920-04-15'],
             'birth_place': ['Warschau'], 'death_date': [], 'death_place': []}

    def _cand(self, forename, surname, birth=None):
        return {'forenames': [forename], 'surnames': [surname],
                'birth_date': [birth] if birth else [], 'birth_place': [],
                'death_date': [], 'death_place': []}

    def test_picks_highest_scoring_other(self):
        cands = [
            (1, self._cand('Jan', 'Kowalski')),                 # different person
            (2, self._cand('Anna', 'Nowak', '1920-04-15')),     # strong duplicate
            (99, self._cand('Anna', 'Nowak', '1920-04-15')),    # identical, but SELF
        ]
        best_id, score = nearest_pair(self.LOCAL, cands, exclude_id=99)
        self.assertEqual(best_id, 2)
        self.assertGreater(score, 50)

    def test_self_is_excluded(self):
        cands = [(7, self._cand('Anna', 'Nowak', '1920-04-15'))]
        self.assertEqual(nearest_pair(self.LOCAL, cands, exclude_id=7), (None, None))

    def test_no_candidates(self):
        self.assertEqual(nearest_pair(self.LOCAL, [], exclude_id=1), (None, None))


if __name__ == '__main__':
    unittest.main()
