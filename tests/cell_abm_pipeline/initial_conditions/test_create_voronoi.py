import unittest

import numpy as np

from cell_abm_pipeline.initial_conditions.create_voronoi import CreateVoronoi


class TestCreateVoronoi(unittest.TestCase):
    def test_get_bounded_array_without_holes(self):
        array = np.array(
            [
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 2, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
            ]
        )

        expected_mask = np.array(
            [
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
            ]
        )

        mask = CreateVoronoi.create_boundary_mask(array, iterations=2)
        self.assertTrue(np.array_equal(expected_mask, mask))

    def test_get_bounded_array_with_holes(self):
        array = np.array(
            [
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 2, 2, 0, 0, 0, 3, 3, 0, 0],
                    [0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0],
                    [0, 0, 2, 2, 0, 0, 0, 3, 3, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ]
        )

        expected_mask = np.array(
            [
                [
                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                ],
                [
                    [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                ],
            ]
        )

        mask = CreateVoronoi.create_boundary_mask(array, iterations=2)
        self.assertTrue(np.array_equal(expected_mask, mask))

    def test_get_array_slices_within_shape(self):
        array = np.zeros((11, 11, 11))
        array[2, 5, 5] = 1
        array[7, 5, 5] = 1
        array[5, 6, 5] = 1
        array[5, 1, 5] = 1
        array[5, 5, 8] = 1
        array[5, 5, 5] = 1
        expected_slices = (slice(1, 9), slice(0, 8), slice(4, 10))

        slices = CreateVoronoi.get_array_slices(array)
        self.assertTupleEqual(expected_slices, slices)

    def test_get_array_bounds_outside_shape(self):
        array = np.zeros((3, 5, 7))
        array[0, 2, 3] = 1
        array[2, 2, 3] = 1
        array[1, 0, 3] = 1
        array[1, 4, 3] = 1
        array[1, 2, 0] = 1
        array[1, 2, 6] = 1
        expected_slices = (slice(0, 3), slice(0, 5), slice(0, 7))

        slices = CreateVoronoi.get_array_slices(array)
        self.assertTupleEqual(expected_slices, slices)

    def test_calculate_voronoi_array(self):
        array = np.array(
            [
                [
                    [2, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 3],
                ],
                [
                    [0, 0, 0, 0, 0],
                    [0, 2, 0, 3, 0],
                    [0, 0, 0, 0, 0],
                ],
            ]
        )

        expected_voronoi = np.array(
            [
                [
                    [2, 2, 2, 3, 3],
                    [2, 2, 2, 3, 3],
                    [2, 2, 2, 3, 3],
                ],
                [
                    [2, 2, 2, 3, 3],
                    [2, 2, 2, 3, 3],
                    [2, 2, 2, 3, 3],
                ],
            ]
        )

        voronoi = CreateVoronoi.calculate_voronoi_array(array)
        self.assertTrue(np.array_equal(expected_voronoi, voronoi))


if __name__ == "__main__":
    unittest.main()
