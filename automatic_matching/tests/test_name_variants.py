"""Tests for forename-variant matching (engine-backed, no DB/ES).

Run locally with:

    python -m unittest automatic_matching.tests.test_name_variants
"""

import unittest

from automatic_matching.name_variants import forenames_variant_linked, score_pair, variant_key

# A tiny hand-built index: Friedrich/Fritz in group 1, Wilhelm/Willi in group 2.
INDEX = {
    variant_key('Friedrich'): frozenset({1}),
    variant_key('Fritz'): frozenset({1}),
    variant_key('Wilhelm'): frozenset({2}),
    variant_key('Willi'): frozenset({2}),
}

DISCOUNT = 0.6


def person(forename, surname, birth_date, birth_place):
    return {
        'forenames': [forename], 'surnames': [surname],
        'birth_date': [birth_date], 'birth_place': [birth_place],
        'death_date': [], 'death_place': [],
    }


class LinkingTests(unittest.TestCase):
    def test_links_known_nickname(self):
        linked, pair = forenames_variant_linked(INDEX, ['Friedrich'], ['Fritz'])
        self.assertTrue(linked)
        self.assertEqual(pair, ('Friedrich', 'Fritz'))

    def test_unrelated_names_not_linked(self):
        linked, _ = forenames_variant_linked(INDEX, ['Friedrich'], ['Willi'])
        self.assertFalse(linked)

    def test_unknown_names_not_linked(self):
        linked, _ = forenames_variant_linked(INDEX, ['Xaver'], ['Quirin'])
        self.assertFalse(linked)


class ScoreBoostTests(unittest.TestCase):
    def test_nickname_raises_score_when_other_fields_match(self):
        a = person('Friedrich', 'Schmidt', '1910-10-01', 'Berlin')
        b = person('Fritz', 'Schmidt', '1910-10-01', 'Berlin')

        baseline = score_pair(a, b)  # no variant index
        boosted = score_pair(a, b, variant_index=INDEX, forename_variant_score=DISCOUNT)

        self.assertGreater(boosted['absolute_score'], baseline['absolute_score'])
        self.assertTrue(boosted.get('variant_matched'))
        # The credited forename sub-score is exactly the discounted value.
        self.assertAlmostEqual(boosted['forename']['score'], DISCOUNT)

    def test_no_boost_without_dictionary_link(self):
        a = person('Xaver', 'Schmidt', '1910-10-01', 'Berlin')
        b = person('Quirin', 'Schmidt', '1910-10-01', 'Berlin')
        baseline = score_pair(a, b)
        result = score_pair(a, b, variant_index=INDEX, forename_variant_score=DISCOUNT)
        self.assertEqual(result['absolute_score'], baseline['absolute_score'])
        self.assertFalse(result.get('variant_matched'))

    def test_variant_never_downgrades_a_real_match(self):
        # Identical forenames already score full; the 0.6 variant must not lower it.
        a = person('Friedrich', 'Schmidt', '1910-10-01', 'Berlin')
        result = score_pair(a, a, variant_index=INDEX, forename_variant_score=DISCOUNT)
        self.assertAlmostEqual(result['forename']['score'], 1.0)
        self.assertFalse(result.get('variant_matched'))


if __name__ == '__main__':
    unittest.main()
