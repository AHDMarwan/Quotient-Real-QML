import unittest

import numpy as np

from statistical_study import _bootstrap_paired, _holm_adjust, _paired_summary


class TestStatisticalStudy(unittest.TestCase):
    def test_bootstrap_detects_positive_paired_difference(self):
        fixed = np.array([1.0, 1.2, 0.8, 1.1, 0.9])
        augmented = fixed + 0.4
        summary = _bootstrap_paired(
            fixed,
            augmented,
            bootstrap_samples=1000,
            seed=7,
        )
        self.assertGreater(summary["mean_difference_ci95_low"], 0.0)
        self.assertGreater(summary["mean_ratio_ci95_low"], 1.0)

    def test_paired_summary_reports_fixed_advantage_direction(self):
        fixed = np.linspace(0.8, 1.2, 20)
        augmented = fixed + np.linspace(0.1, 0.3, 20)
        summary = _paired_summary(
            fixed,
            augmented,
            bootstrap_samples=1000,
            seed=11,
        )
        self.assertGreater(summary["mean_difference_augmented_minus_fixed"], 0.0)
        self.assertGreater(summary["paired_cohen_dz"], 0.0)
        self.assertGreater(summary["fraction_augmented_worse"], 0.95)

    def test_holm_adjustment_is_monotone_in_sorted_order(self):
        raw = [0.001, 0.02, 0.04, 0.8]
        adjusted = _holm_adjust(raw)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in adjusted))
        order = np.argsort(raw)
        sorted_adjusted = np.asarray(adjusted)[order]
        self.assertTrue(np.all(np.diff(sorted_adjusted) >= -1e-15))
        self.assertTrue(np.all(np.asarray(adjusted) >= np.asarray(raw) - 1e-15))


if __name__ == "__main__":
    unittest.main()
