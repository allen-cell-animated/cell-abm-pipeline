import unittest

from cell_abm_pipeline.cell_shape.calculate_coefficients import CalculateCoefficients


class TestCalculateCoefficients(unittest.TestCase):
    def test_get_coeff_names_no_prefix_no_suffix_creates_list(self):
        order = 1
        coeff_names = CalculateCoefficients.get_coeff_names(order=order)

        expected_names = {
            "shcoeffs_L0M0C",
            "shcoeffs_L0M1C",
            "shcoeffs_L1M0C",
            "shcoeffs_L1M1C",
            "shcoeffs_L0M0S",
            "shcoeffs_L0M1S",
            "shcoeffs_L1M0S",
            "shcoeffs_L1M1S",
        }

        self.assertSetEqual(expected_names, set(coeff_names))
