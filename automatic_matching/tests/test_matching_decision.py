"""Tests for the pure cluster-resolution decision (no DB / ES / Django needed).

Run locally with:

    python -m unittest automatic_matching.tests.test_matching_decision
"""

import unittest

from automatic_matching.matching_decision import (
    Action,
    MatchDecision,
    REVIEW_NEAR_THRESHOLD,
    REVIEW_POTENTIAL_MERGE,
    ScoredCandidate,
    decide,
)

AUTO = 60.0
REVIEW = 40.0


def run(candidates):
    return decide(candidates, auto_threshold=AUTO, review_threshold=REVIEW)


class StandaloneTests(unittest.TestCase):
    def test_no_candidates_stands_alone_unflagged(self):
        d = run([])
        self.assertEqual(d.action, Action.STANDALONE)
        self.assertFalse(d.flagged_for_review)
        self.assertEqual(d.top_score, 0.0)

    def test_only_weak_candidates_stands_alone_unflagged(self):
        d = run([ScoredCandidate(1, None, 12.0), ScoredCandidate(2, 5, 30.0)])
        self.assertEqual(d.action, Action.STANDALONE)
        self.assertFalse(d.flagged_for_review)

    def test_near_miss_stands_alone_but_flags_for_review(self):
        d = run([ScoredCandidate(1, 5, 50.0)])  # between REVIEW and AUTO
        self.assertEqual(d.action, Action.STANDALONE)
        self.assertTrue(d.flagged_for_review)
        self.assertEqual(d.review_reason, REVIEW_NEAR_THRESHOLD)
        self.assertEqual(d.top_score, 50.0)


class AttachTests(unittest.TestCase):
    def test_single_cluster_attaches(self):
        d = run([ScoredCandidate(1, 5, 88.0), ScoredCandidate(2, 5, 72.0)])
        self.assertEqual(d.action, Action.ATTACH)
        self.assertEqual(d.target_cluster_id, 5)
        self.assertFalse(d.flagged_for_review)
        self.assertEqual(d.top_score, 88.0)

    def test_unaggregated_strong_matches_are_pulled_in(self):
        d = run([ScoredCandidate(1, 5, 80.0), ScoredCandidate(2, None, 75.0)])
        self.assertEqual(d.action, Action.ATTACH)
        self.assertEqual(d.target_cluster_id, 5)
        self.assertEqual(d.attach_record_ids, [2])

    def test_near_miss_in_other_cluster_does_not_trigger_merge(self):
        # Only strong matches count toward cluster spanning.
        d = run([ScoredCandidate(1, 5, 90.0), ScoredCandidate(2, 7, 45.0)])
        self.assertEqual(d.action, Action.ATTACH)
        self.assertEqual(d.target_cluster_id, 5)
        self.assertFalse(d.flagged_for_review)


class CreateTests(unittest.TestCase):
    def test_strong_unaggregated_matches_create_new_person(self):
        d = run([ScoredCandidate(1, None, 85.0), ScoredCandidate(2, None, 70.0)])
        self.assertEqual(d.action, Action.CREATE)
        self.assertIsNone(d.target_cluster_id)
        self.assertEqual(sorted(d.attach_record_ids), [1, 2])


class MergeReviewTests(unittest.TestCase):
    def test_strong_matches_across_clusters_flag_potential_merge(self):
        d = run([ScoredCandidate(1, 5, 70.0), ScoredCandidate(2, 7, 92.0)])
        self.assertEqual(d.action, Action.ATTACH)
        self.assertTrue(d.flagged_for_review)
        self.assertEqual(d.review_reason, REVIEW_POTENTIAL_MERGE)
        self.assertEqual(d.competing_cluster_ids, [5, 7])
        # Attaches to the best-scoring cluster (7 scored 92 vs 5's 70).
        self.assertEqual(d.target_cluster_id, 7)
        # Never auto-merges: it only flags.

    def test_best_cluster_uses_max_score_per_cluster(self):
        d = run([
            ScoredCandidate(1, 5, 95.0),
            ScoredCandidate(2, 7, 80.0),
            ScoredCandidate(3, 7, 99.0),
        ])
        self.assertEqual(d.target_cluster_id, 7)
        self.assertEqual(d.competing_cluster_ids, [5, 7])


class MarginTests(unittest.TestCase):
    def test_margin_and_alternate_count(self):
        d = run([ScoredCandidate(1, 5, 90.0), ScoredCandidate(2, 5, 84.0), ScoredCandidate(3, None, 40.0)])
        self.assertAlmostEqual(d.score_margin, 6.0)   # 90 − 84
        self.assertEqual(d.alternate_count, 2)

    def test_no_runner_up(self):
        d = run([ScoredCandidate(1, 5, 90.0)])
        self.assertIsNone(d.score_margin)
        self.assertEqual(d.alternate_count, 0)


class SingleClusterNearAttachTests(unittest.TestCase):
    """A near-miss record whose candidates ALL point to one existing person attaches
    to it (the auto bar guards against choosing between rivals; there is none)."""

    SC = 55.0

    def run_sc(self, candidates):
        return decide(candidates, auto_threshold=AUTO, review_threshold=REVIEW,
                      single_cluster_threshold=self.SC)

    def test_sole_cluster_near_bar_attaches(self):
        # the Wilh. Oppel case: several members of ONE person, all at 59.69
        d = self.run_sc([ScoredCandidate(i, 934, 59.69) for i in range(1, 6)])
        self.assertEqual(d.action, Action.ATTACH)
        self.assertEqual(d.target_cluster_id, 934)
        self.assertTrue(d.near_threshold_single)

    def test_below_single_cluster_threshold_stands_alone(self):
        d = self.run_sc([ScoredCandidate(1, 934, 52.0)])  # < 55
        self.assertEqual(d.action, Action.STANDALONE)
        self.assertFalse(d.near_threshold_single)

    def test_two_competing_clusters_stay_for_review(self):
        d = self.run_sc([ScoredCandidate(1, 934, 59.0), ScoredCandidate(2, 999, 58.0)])
        self.assertEqual(d.action, Action.STANDALONE)
        self.assertTrue(d.flagged_for_review)

    def test_unaggregated_near_candidate_does_not_attach(self):
        d = self.run_sc([ScoredCandidate(1, None, 59.0)])  # no existing person
        self.assertEqual(d.action, Action.STANDALONE)

    def test_strong_match_is_a_normal_attach(self):
        d = self.run_sc([ScoredCandidate(1, 934, 65.0)])  # >= AUTO → ordinary attach
        self.assertEqual(d.action, Action.ATTACH)
        self.assertFalse(d.near_threshold_single)

    def test_disabled_without_threshold(self):
        d = decide([ScoredCandidate(1, 934, 59.69)], auto_threshold=AUTO, review_threshold=REVIEW)
        self.assertEqual(d.action, Action.STANDALONE)


if __name__ == '__main__':
    unittest.main()
