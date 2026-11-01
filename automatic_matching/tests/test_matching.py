"""Regression tests that lock the vendored matching engine's contract.

The engine (algorithm v2.9) is the heart of the product; these tests pin its
behaviour so it cannot drift silently. They are plain ``unittest`` cases (no
Django required) and run either via::

    python -m unittest automatic_matching.tests.test_matching

or under the Django test runner::

    python manage.py test automatic_matching

Assertions favour structural / deterministic / equivalence properties over
hand-computed fuzzy scores, so they stay meaningful without being brittle.
"""

import unittest

from automatic_matching import (
    AUTOMATIC_MATCHING_ALGORITHM_VERSION_STRING,
    BIRTH_DATE_MAX_SCORE_CONTRIBUTION,
    BIRTH_PLACE_MAX_SCORE_CONTRIBUTION,
    DEATH_DATE_MAX_SCORE_CONTRIBUTION,
    DEATH_PLACE_MAX_SCORE_CONTRIBUTION,
    FORENAME_MAX_SCORE_CONTRIBUTION,
    MIN_REQUIRED_SCORE_FOR_AUTO_MATCHING,
    SURNAME_MAX_SCORE_CONTRIBUTION,
    corroborating_boost,
    corroborating_signal_fields,
    date_edit_distance,
    get_matching_score,
    get_names_as_dict,
    name_edit_distance,
    normalize_string,
    split_string_values,
)

# Sum of every field's maximum contribution — a perfect, complete match.
TOTAL_MAX = (
    FORENAME_MAX_SCORE_CONTRIBUTION
    + SURNAME_MAX_SCORE_CONTRIBUTION
    + BIRTH_PLACE_MAX_SCORE_CONTRIBUTION
    + BIRTH_DATE_MAX_SCORE_CONTRIBUTION
    + DEATH_PLACE_MAX_SCORE_CONTRIBUTION
    + DEATH_DATE_MAX_SCORE_CONTRIBUTION
)

COMPLETE_PERSON = {
    'forenames': ['Anna', 'Maria'],
    'surnames': ['Musterfrau'],
    'birth_place': ['München'],
    'birth_date': ['1900-01-01'],
    'death_place': ['Dachau'],
    'death_date': ['1945-03-15'],
}


class ScoreContractTests(unittest.TestCase):
    def test_result_has_expected_shape(self):
        result = get_matching_score(COMPLETE_PERSON, COMPLETE_PERSON)
        for key in (
            'absolute_score',
            'relative_score',
            'total_relative_score',
            'max_score_reachable',
            'automatically_matched',
            'matching_algorithm_version',
        ):
            self.assertIn(key, result)
        for field in ('forename', 'surname', 'birth_place', 'birth_date', 'death_place', 'death_date'):
            self.assertIn(field, result)

    def test_algorithm_version_is_reported(self):
        result = get_matching_score(COMPLETE_PERSON, COMPLETE_PERSON)
        self.assertEqual(result['matching_algorithm_version'], '2.9')
        self.assertEqual(AUTOMATIC_MATCHING_ALGORITHM_VERSION_STRING, '2.9')

    def test_identical_complete_record_is_a_perfect_match(self):
        result = get_matching_score(COMPLETE_PERSON, COMPLETE_PERSON)
        self.assertAlmostEqual(result['absolute_score'], TOTAL_MAX)
        self.assertAlmostEqual(result['relative_score'], 1.0)
        self.assertAlmostEqual(result['total_relative_score'], 1.0)
        self.assertTrue(result['automatically_matched'])

    def test_clearly_different_people_do_not_auto_match(self):
        local = {
            'forenames': ['Hans'],
            'surnames': ['Müller'],
            'birth_place': ['München'],
            'birth_date': ['1900-01-01'],
        }
        external = {
            'forenames': ['Sarah'],
            'surnames': ['Cohen'],
            'birth_place': ['Berlin'],
            'birth_date': ['1880-12-31'],
        }
        result = get_matching_score(local, external)
        self.assertLess(result['absolute_score'], MIN_REQUIRED_SCORE_FOR_AUTO_MATCHING)
        self.assertFalse(result['automatically_matched'])


class DiacriticInsensitivityTests(unittest.TestCase):
    """Transliteration must make umlaut / accent spellings equivalent."""

    def test_normalize_folds_diacritics(self):
        self.assertEqual(normalize_string('Müller', True), normalize_string('Muller', True))
        self.assertEqual(normalize_string('Anna'), normalize_string('anna'))

    def test_variant_spellings_group_under_one_key(self):
        grouped = get_names_as_dict(['Müller', 'Muller'], is_surname=True)
        self.assertEqual(len(grouped), 1)

    def test_diacritic_only_difference_still_scores_high(self):
        local = {'forenames': ['Anna'], 'surnames': ['Müller']}
        external = {'forenames': ['Anna'], 'surnames': ['Muller']}
        result = get_matching_score(local, external)
        # Normalisation makes the keys identical, but the engine still factors
        # in a sliver of residual difference between the raw spellings, so the
        # score lands just under a perfect 1.0 rather than exactly on it.
        self.assertGreater(result['relative_score'], 0.99)
        self.assertLessEqual(result['relative_score'], 1.0)


class PartialDataTests(unittest.TestCase):
    """A field only counts toward `max_score_reachable` when both sides have
    it; toward `max_total_score_reachable` when either side does. This is what
    keeps `relative_score` and `total_relative_score` distinct."""

    def test_names_only_identical_auto_matches_via_perfect_relative_rule(self):
        names_only = {'forenames': ['Anna'], 'surnames': ['Musterfrau']}
        result = get_matching_score(names_only, names_only)
        self.assertAlmostEqual(
            result['absolute_score'],
            FORENAME_MAX_SCORE_CONTRIBUTION + SURNAME_MAX_SCORE_CONTRIBUTION,
        )
        # total_relative_score == 1 and absolute >= 50 → auto-match even though
        # absolute (50) is below the 60 hard threshold.
        self.assertAlmostEqual(result['total_relative_score'], 1.0)
        self.assertTrue(result['automatically_matched'])

    def test_missing_fields_on_one_side_split_relative_from_total(self):
        # Names match perfectly, but the external record carries nothing else.
        result = get_matching_score(COMPLETE_PERSON, {'forenames': ['Anna', 'Maria'], 'surnames': ['Musterfrau']})
        # Only forename+surname are reachable (present on both sides).
        self.assertAlmostEqual(result['relative_score'], 1.0)
        # …but the local record's other fields still inflate the total ceiling,
        # so the total relative score is dragged below 1 and it must not
        # auto-match on names alone.
        self.assertLess(result['total_relative_score'], 1.0)
        self.assertFalse(result['automatically_matched'])


class HelperFunctionTests(unittest.TestCase):
    def test_split_string_values_handles_separators(self):
        self.assertEqual(split_string_values('Anna, Anne'), ['Anna', 'Anne'])
        self.assertEqual(split_string_values('Anna-Maria'), ['Anna', 'Maria'])

    def test_split_string_values_strips_brackets(self):
        self.assertEqual(split_string_values('(Levy)'), ['Levy'])

    def test_get_names_as_dict_drops_single_letter_acronyms(self):
        # remove_acronyms=True by default → single letters / "X." are skipped.
        grouped = get_names_as_dict(['Anna', 'A', 'M.'])
        self.assertEqual(len(grouped), 1)


class DateSymmetryTests(unittest.TestCase):
    """An imprecise date (a **-wildcard, a </> bound, or a range) that overlaps the
    other side scores a MODERATE partial the SAME in both directions — never a
    strong match one way and a conflict the other (the old order-dependent bug)."""

    @staticmethod
    def _score(a, b):
        from automatic_matching.automatic_matching_functions import match_date_against_local_date
        return match_date_against_local_date([a] if a else [], [b] if b else [])['score']

    def test_wildcard_vs_full_is_symmetric_and_partial(self):
        from automatic_matching.automatic_matching_functions import DATE_IMPRECISE_PARTIAL_SCORE
        self.assertAlmostEqual(self._score('1926-**-**', '1926-12-15'), DATE_IMPRECISE_PARTIAL_SCORE)
        self.assertAlmostEqual(self._score('1926-12-15', '1926-**-**'), DATE_IMPRECISE_PARTIAL_SCORE)

    def test_open_bound_vs_full_is_symmetric_and_partial(self):
        from automatic_matching.automatic_matching_functions import DATE_IMPRECISE_PARTIAL_SCORE
        self.assertAlmostEqual(self._score('>1926-01-01', '1930-05-10'), DATE_IMPRECISE_PARTIAL_SCORE)
        self.assertAlmostEqual(self._score('1930-05-10', '>1926-01-01'), DATE_IMPRECISE_PARTIAL_SCORE)

    def test_month_wildcard_overlap_is_symmetric_partial(self):
        from automatic_matching.automatic_matching_functions import DATE_IMPRECISE_PARTIAL_SCORE
        self.assertAlmostEqual(self._score('1926-05-**', '1926-05-15'), DATE_IMPRECISE_PARTIAL_SCORE)
        self.assertAlmostEqual(self._score('1926-05-15', '1926-05-**'), DATE_IMPRECISE_PARTIAL_SCORE)

    def test_exact_full_date_outscores_a_wildcard(self):
        from automatic_matching.automatic_matching_functions import DATE_IMPRECISE_PARTIAL_SCORE
        exact = self._score('1926-12-15', '1926-12-15')
        self.assertAlmostEqual(exact, 1.0)
        self.assertGreater(exact, DATE_IMPRECISE_PARTIAL_SCORE)

    def test_non_overlapping_imprecise_dates_conflict(self):
        # different years / out of range → still a conflict, both directions
        self.assertLess(self._score('1926-**-**', '1925-08-13'), 0)
        self.assertLess(self._score('1925-08-13', '1926-**-**'), 0)
        self.assertLess(self._score('>1926-01-01', '1920-05-10'), 0)
        self.assertLess(self._score('1926-05-**', '1926-08-15'), 0)

    def test_adjacent_ranges_that_only_touch_do_not_match(self):
        # Exclusive upper bound: 1912-**-** ends where 1913-**-** begins → conflict,
        # not a partial. (Regression: the overlap test must be strict.)
        self.assertLess(self._score('1912-**-**', '1913-**-**'), 0)
        self.assertLess(self._score('1913-**-**', '1912-**-**'), 0)
        self.assertLess(self._score('1912-11-**', '1912-12-15'), 0)  # adjacent months
        # a full date on the wildcard's exclusive boundary is NOT inside it
        self.assertLess(self._score('1912-**-**', '1913-01-01'), 0)


PRISONER = {
    'field': 'prisoner_number', 'compare': 'number', 'boost': 5,
    'require_match': ('place_of_imprisonment',),
}
RESIDENCE = {'field': 'residence_place', 'compare': 'place', 'boost': 5}


class CorroboratingSignalsTests(unittest.TestCase):
    """The near-miss corroborating boost (prisoner number / residence place),
    positive-only and gated on the near-miss band. Pure — operates on a result dict."""

    AUTO = 60
    REVIEW = 40

    def near_miss(self, score=50):
        return {'absolute_score': score, 'max_score_reachable': 100, 'relative_score': score / 100}

    def boost(self, result, local, external, signals=(PRISONER, RESIDENCE), **kw):
        return corroborating_boost(result, local, external,
                                   near_miss_min=self.REVIEW, near_miss_max=self.AUTO,
                                   signals=signals, **kw)

    # -- banding -----------------------------------------------------------
    def test_near_miss_prisoner_match_adds_boost(self):
        local = {'prisoner_number': '12345', 'place_of_imprisonment': 'Buchenwald'}
        external = {'prisoner_number': '12345', 'place_of_imprisonment': 'Buchenwald'}
        r = self.boost(self.near_miss(52), local, external)
        self.assertEqual(r['absolute_score'], 57)
        self.assertEqual(r['corroborating_boost'], 5)
        self.assertEqual([s['field'] for s in r['corroborating_signals']], ['prisoner_number'])

    def test_above_auto_is_untouched(self):
        local = external = {'prisoner_number': '12345', 'place_of_imprisonment': 'Buchenwald'}
        r = self.boost(self.near_miss(72), local, external)
        self.assertEqual(r['absolute_score'], 72)
        self.assertNotIn('corroborating_boost', r)

    def test_below_review_is_untouched(self):
        local = external = {'prisoner_number': '12345', 'place_of_imprisonment': 'Buchenwald'}
        r = self.boost(self.near_miss(30), local, external)
        self.assertEqual(r['absolute_score'], 30)
        self.assertNotIn('corroborating_boost', r)

    # -- positive-only -----------------------------------------------------
    def test_number_mismatch_is_a_no_op(self):
        local = {'prisoner_number': '12345', 'place_of_imprisonment': 'Buchenwald'}
        external = {'prisoner_number': '99999', 'place_of_imprisonment': 'Buchenwald'}
        r = self.boost(self.near_miss(52), local, external)
        self.assertEqual(r['absolute_score'], 52)
        self.assertNotIn('corroborating_boost', r)

    def test_missing_signal_is_a_no_op(self):
        r = self.boost(self.near_miss(52), {}, {})
        self.assertEqual(r['absolute_score'], 52)
        self.assertNotIn('corroborating_boost', r)

    # -- guards ------------------------------------------------------------
    def test_guard_conflict_withholds_boost(self):
        # same number, DIFFERENT camp → per-camp reuse, boost withheld (not a penalty)
        local = {'prisoner_number': '12345', 'place_of_imprisonment': 'Buchenwald'}
        external = {'prisoner_number': '12345', 'place_of_imprisonment': 'Dachau'}
        r = self.boost(self.near_miss(52), local, external)
        self.assertEqual(r['absolute_score'], 52)
        self.assertNotIn('corroborating_boost', r)

    def test_guard_absent_is_inconclusive_and_boost_applies(self):
        # same number, no camp recorded on one side → guard inconclusive → still boosts
        local = {'prisoner_number': '12345', 'place_of_imprisonment': 'Buchenwald'}
        external = {'prisoner_number': '12345'}
        r = self.boost(self.near_miss(52), local, external)
        self.assertEqual(r['corroborating_boost'], 5)
        self.assertEqual(r['corroborating_signals'][0]['guards'], {'place_of_imprisonment': 'absent'})

    def test_guard_match_is_recorded(self):
        local = external = {'prisoner_number': '12345', 'place_of_imprisonment': 'Buchenwald'}
        r = self.boost(self.near_miss(52), local, external)
        self.assertEqual(r['corroborating_signals'][0]['guards'], {'place_of_imprisonment': 'match'})

    # -- comparison semantics ---------------------------------------------
    def test_ocr_digit_tolerance_on_numbers(self):
        # 7↔1 is the engine's common OCR digit confusion — treated as the same number
        local = {'prisoner_number': '70304718', 'place_of_imprisonment': 'Buchenwald'}
        external = {'prisoner_number': '10304118', 'place_of_imprisonment': 'Buchenwald'}
        r = self.boost(self.near_miss(52), local, external)
        self.assertEqual(r['corroborating_boost'], 5)

    def test_residence_place_matches_phonetically(self):
        local = {'residence_place': 'Muenchen'}
        external = {'residence_place': 'München'}
        r = self.boost(self.near_miss(52), local, external, signals=(RESIDENCE,))
        self.assertEqual(r['corroborating_boost'], 5)

    def test_residence_place_mismatch_is_a_no_op(self):
        local = {'residence_place': 'Berlin'}
        external = {'residence_place': 'Warschau'}
        r = self.boost(self.near_miss(52), local, external, signals=(RESIDENCE,))
        self.assertNotIn('corroborating_boost', r)

    # -- cap + accumulation ------------------------------------------------
    def test_multiple_signals_accumulate_and_cap(self):
        local = external = {
            'prisoner_number': '12345', 'place_of_imprisonment': 'Buchenwald',
            'residence_place': 'Berlin',
        }
        uncapped = self.boost(self.near_miss(52), local, external)
        self.assertEqual(uncapped['corroborating_boost'], 10)  # 5 + 5
        capped = self.boost(self.near_miss(52), local, external, cap=7)
        self.assertEqual(capped['corroborating_boost'], 7)
        self.assertEqual(capped['absolute_score'], 59)

    def test_relative_score_tracks_the_boost(self):
        local = external = {'prisoner_number': '12345', 'place_of_imprisonment': 'Buchenwald'}
        r = self.boost(self.near_miss(52), local, external)
        self.assertAlmostEqual(r['relative_score'], 0.57)

    # -- config helper -----------------------------------------------------
    def test_signal_fields_lists_fields_and_guards(self):
        fields = corroborating_signal_fields((PRISONER, RESIDENCE))
        self.assertEqual(fields, ('prisoner_number', 'place_of_imprisonment', 'residence_place'))

    def test_empty_signals_is_a_no_op(self):
        r = self.boost(self.near_miss(52), {'prisoner_number': '1'}, {'prisoner_number': '1'}, signals=())
        self.assertNotIn('corroborating_boost', r)


class EditDistanceReadoutTests(unittest.TestCase):
    """The edit-distance readout: how many spelling/OCR edits apart two names/dates
    are, after the engine's own normalisation. Wildcards/open bounds don't count."""

    def test_name_distance_after_normalisation(self):
        self.assertEqual(name_edit_distance(['Anna'], ['Hanna']), 1)
        self.assertEqual(name_edit_distance(['Nowak'], ['Nowak'], True), 0)
        self.assertGreaterEqual(name_edit_distance(['Nowak'], ['Kowalski'], True), 3)

    def test_name_distance_none_when_missing(self):
        self.assertIsNone(name_edit_distance([], ['Anna']))

    def test_name_distance_is_min_over_tokens(self):
        # best-aligned token wins (Maria↔Maria = 0 despite the extra forename)
        self.assertEqual(name_edit_distance(['Anna', 'Maria'], ['Maria']), 0)

    def test_date_digit_distance(self):
        self.assertEqual(date_edit_distance(['1914-10-09'], ['1904-10-09']), 1)  # one-digit year
        self.assertEqual(date_edit_distance(['1920-04-15'], ['1920-11-15']), 2)  # month swap
        self.assertEqual(date_edit_distance(['1920-01-01'], ['1955-12-31']), 5)

    def test_date_distance_ignores_wildcards_and_bounds(self):
        self.assertEqual(date_edit_distance(['1920-**-**'], ['1920-04-15']), 0)  # year agrees, rest unknown
        self.assertEqual(date_edit_distance(['<1920-01-01'], ['1920-06-06']), 0)
        self.assertIsNone(date_edit_distance(['**-**-**'], ['1920-04-15']))      # nothing comparable

    def test_date_distance_none_when_missing(self):
        self.assertIsNone(date_edit_distance([], ['1920-04-15']))

    def test_readout_is_opt_in(self):
        a = {'forenames': ['Anna'], 'surnames': ['Nowak'], 'birth_date': ['1920-04-15']}
        b = {'forenames': ['Hanna'], 'surnames': ['Nowak'], 'birth_date': ['1920-11-15']}
        # off by default (hot path pays nothing)
        self.assertNotIn('edit_distances', get_matching_score(a, b))
        # present when explicitly requested
        r = get_matching_score(a, b, with_edit_distances=True)
        self.assertEqual(r['edit_distances'], {'forename': 1, 'surname': 0, 'birth_date': 2})


class CorroborationEditGateTests(unittest.TestCase):
    """``require_close``: a reused identifier only corroborates when the rest of the
    identity is a small edit apart. A missing readout is inconclusive (doesn't block)."""

    SIG = {'field': 'prisoner_number', 'compare': 'number', 'boost': 10,
           'require_close': {'surname': 2, 'forename': 2, 'birth_date': 3}}

    def near(self, edits):
        return {'absolute_score': 52, 'max_score_reachable': 100, 'edit_distances': edits}

    def boost(self, edits):
        return corroborating_boost(self.near(edits),
                                   {'prisoner_number': '123'}, {'prisoner_number': '123'},
                                   near_miss_min=40, near_miss_max=60, signals=(self.SIG,))

    def test_small_edits_pass(self):
        r = self.boost({'forename': 1, 'surname': 0, 'birth_date': 2})  # Anna/Hanna, month swap
        self.assertEqual(r['corroborating_boost'], 10)
        self.assertEqual(r['corroborating_signals'][0]['edits'], {'forename': 1, 'surname': 0, 'birth_date': 2})

    def test_distant_surname_withholds(self):
        r = self.boost({'forename': 0, 'surname': 4, 'birth_date': 0})  # Nowak vs Kowalski
        self.assertNotIn('corroborating_boost', r)

    def test_distant_date_withholds(self):
        r = self.boost({'forename': 0, 'surname': 0, 'birth_date': 5})  # wholly different date
        self.assertNotIn('corroborating_boost', r)

    def test_missing_readout_is_inconclusive(self):
        r = self.boost({'forename': 0, 'surname': 0, 'birth_date': None})  # names-only, no date
        self.assertEqual(r['corroborating_boost'], 10)


class TieredBoostTests(unittest.TestCase):
    """Tiered prisoner-number boost: +5 when names-only (no date to verify), the full
    +10 when a birth date also corroborates (present on both, within tolerance)."""

    SIG = {'field': 'prisoner_number', 'compare': 'number',
           'boost': 5, 'boost_corroborated': 10, 'corroborated_by': 'birth_date',
           'require_close': {'surname': 2, 'forename': 2, 'birth_date': 2}}

    def boost(self, edits):
        result = {'absolute_score': 50, 'max_score_reachable': 100, 'edit_distances': edits}
        return corroborating_boost(result, {'prisoner_number': '123'}, {'prisoner_number': '123'},
                                   near_miss_min=40, near_miss_max=60, signals=(self.SIG,))

    def test_names_only_gets_base_boost(self):
        r = self.boost({'forename': 0, 'surname': 0, 'birth_date': None})
        self.assertEqual(r['corroborating_boost'], 5)
        self.assertEqual(r['absolute_score'], 55)   # stays below the 60 auto-bar
        self.assertFalse(r['corroborating_signals'][0]['corroborated'])

    def test_matching_date_upgrades_to_full_boost(self):
        r = self.boost({'forename': 0, 'surname': 0, 'birth_date': 0})
        self.assertEqual(r['corroborating_boost'], 10)
        self.assertEqual(r['absolute_score'], 60)   # crosses the auto-bar
        self.assertTrue(r['corroborating_signals'][0]['corroborated'])

    def test_slight_name_diff_with_matching_dob_still_full(self):
        # the caveat: Anna/Hanna (edit 1) + matching DOB → full +10
        r = self.boost({'forename': 1, 'surname': 0, 'birth_date': 0})
        self.assertEqual(r['corroborating_boost'], 10)

    def test_month_swap_date_still_corroborates(self):
        # e26b0b11: month 04↔11 = edit 2, within tolerance → full +10
        r = self.boost({'forename': 0, 'surname': 0, 'birth_date': 2})
        self.assertEqual(r['corroborating_boost'], 10)

    def test_diverging_date_beyond_tolerance_withholds(self):
        r = self.boost({'forename': 0, 'surname': 0, 'birth_date': 4})
        self.assertNotIn('corroborating_boost', r)


if __name__ == '__main__':
    unittest.main()
