import unittest

import numpy as np
import pandas as pd

from cell_abm_pipeline.initial_conditions.__config__ import SCALE_MICRONS_XY, SCALE_MICRONS_Z
from cell_abm_pipeline.initial_conditions.process_samples import ProcessSamples


class TestProcessSamples(unittest.TestCase):
    def test_scale_coordinates_using_defaults(self):
        scale_factor = 0.75
        sample_data = np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
                [17, 18, 19, 20],
            ]
        )

        samples = pd.DataFrame(sample_data, columns=["id", "x", "y", "z"])
        x_scaled = sample_data[:, 1] * scale_factor * SCALE_MICRONS_XY
        y_scaled = sample_data[:, 2] * scale_factor * SCALE_MICRONS_XY
        z_scaled = sample_data[:, 3] * scale_factor * SCALE_MICRONS_Z

        scaled_samples = ProcessSamples.scale_coordinates(samples, scale_factor)

        for a, b in zip(x_scaled, scaled_samples["x_scaled"]):
            self.assertAlmostEqual(a, b, places=5)

        for a, b in zip(y_scaled, scaled_samples["y_scaled"]):
            self.assertAlmostEqual(a, b, places=5)

        for a, b in zip(z_scaled, scaled_samples["z_scaled"]):
            self.assertAlmostEqual(a, b, places=5)

    def test_scale_coordinates_given_all_parameters(self):
        scale_xy = 0.4
        scale_z = 1.3
        scale_factor = 0.75
        sample_data = np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
                [17, 18, 19, 20],
            ]
        )

        samples = pd.DataFrame(sample_data, columns=["id", "x", "y", "z"])
        x_scaled = sample_data[:, 1] * scale_factor * scale_xy
        y_scaled = sample_data[:, 2] * scale_factor * scale_xy
        z_scaled = sample_data[:, 3] * scale_factor * scale_z

        scaled_samples = ProcessSamples.scale_coordinates(
            samples, scale_factor, scale_xy=scale_xy, scale_z=scale_z
        )

        for a, b in zip(x_scaled, scaled_samples["x_scaled"]):
            self.assertAlmostEqual(a, b, places=5)

        for a, b in zip(y_scaled, scaled_samples["y_scaled"]):
            self.assertAlmostEqual(a, b, places=5)

        for a, b in zip(z_scaled, scaled_samples["z_scaled"]):
            self.assertAlmostEqual(a, b, places=5)

    def test_include_cells(self):
        cell_1_data = [[1, 1, 2, 3]]
        cell_2_data = [[2, 1, 2, 1]]
        cell_3_data = [[3, 1, 2, 2]]
        sample_data = cell_1_data + cell_2_data + cell_3_data
        samples = pd.DataFrame(sample_data, columns=["id", "x", "y", "z"])

        include = [2, 3]
        expected_data = cell_2_data + cell_3_data
        expected_samples = pd.DataFrame(expected_data, columns=["id", "x", "y", "z"])

        selected_samples = ProcessSamples.include_cells(samples, include)

        self.assertTrue(expected_samples.equals(selected_samples))

    def test_remove_edge_cells_rect_grid(self):
        edge_threshold = 1
        cell_1_data = [[1, 0, 2, 1], [1, 2, 0, 2], [1, 2, 4, 3], [1, 4, 2, 4]]
        cell_2_data = [[2, 1, 1, 1], [2, 2, 1, 2]]
        cell_3_data = [[3, 0, 3, 1], [3, 1, 3, 2]]
        cell_4_data = [[4, 3, 3, 1], [4, 3, 4, 2]]

        sample_data = cell_1_data + cell_2_data + cell_3_data + cell_4_data
        samples = pd.DataFrame(sample_data, columns=["id", "x", "y", "z"])

        expected_data = cell_2_data + cell_3_data + cell_4_data
        expected_samples = pd.DataFrame(expected_data, columns=["id", "x", "y", "z"])

        filtered_samples = ProcessSamples.remove_edge_cells(
            samples, grid="rect", edge_threshold=edge_threshold
        )
        self.assertTrue(expected_samples.equals(filtered_samples))

    def test_remove_edge_cells_hex_grid(self):
        edge_threshold = 1
        cell_1_data = [[1, 0, 2, 1], [1, 4, 0, 2], [1, 4, 4, 3], [1, 8, 2, 4]]
        cell_2_data = [[2, 3, 1, 1], [2, 5, 1, 2]]
        cell_3_data = [[3, 1, 3, 1], [3, 3, 3, 2]]
        cell_4_data = [[4, 6, 2, 1], [4, 6, 4, 2]]

        sample_data = cell_1_data + cell_2_data + cell_3_data + cell_4_data
        samples = pd.DataFrame(sample_data, columns=["id", "x", "y", "z"])

        expected_data = cell_4_data
        expected_samples = pd.DataFrame(expected_data, columns=["id", "x", "y", "z"])

        filtered_samples = ProcessSamples.remove_edge_cells(
            samples, grid="hex", edge_threshold=edge_threshold
        )
        self.assertTrue(expected_samples.equals(filtered_samples))

    def test_remove_unconnected_regions_rect_grid(self):
        connected_threshold = 0
        samples = pd.DataFrame(
            [
                [2, 6, 3, 10],
                [2, 6, 6, 10],
                [2, 6, 3, 20],
                [2, 4, 9, 20],
                [2, 2, 9, 20],
                [1, 2, 3, 10],
                [1, 2, 6, 10],
                [1, 4, 3, 10],
                [1, 4, 6, 10],
                [1, 6, 9, 10],
            ],
            columns=["id", "x", "y", "z"],
        )

        expected = pd.DataFrame(
            [
                [1, 2, 3, 10],
                [1, 2, 6, 10],
                [1, 4, 3, 10],
                [1, 4, 6, 10],
                [2, 6, 3, 10],
                [2, 6, 3, 20],
                [2, 6, 6, 10],
            ],
            columns=["id", "x", "y", "z"],
        )

        filtered_samples = ProcessSamples.remove_unconnected_regions(
            samples, grid="rect", connected_threshold=connected_threshold
        )
        self.assertTrue(expected.equals(filtered_samples))

    def test_remove_unconnected_regions_hex_grid(self):
        connected_threshold = 1.5
        samples = pd.DataFrame(
            [
                [2, 0, 0, 0],
                [2, 1, 1, 1],
                [2, 1, 2, 2],
                [1, 4, 4, 4],
                [1, 7, 5, 4],
                [1, 4, 3, 4],
            ],
            columns=["id", "x", "y", "z"],
        )

        expected = pd.DataFrame(
            [
                [1, 4, 3, 4],
                [1, 4, 4, 4],
                [2, 1, 1, 1],
                [2, 1, 2, 2],
            ],
            columns=["id", "x", "y", "z"],
        )

        filtered_samples = ProcessSamples.remove_unconnected_regions(
            samples, grid="hex", connected_threshold=connected_threshold
        )
        self.assertTrue(expected.equals(filtered_samples))

    def test_remove_unconnected_regions_invalid_grid_throws_exception(self):
        with self.assertRaises(ValueError):
            samples = pd.DataFrame()
            grid = "invalid_grid"
            ProcessSamples.remove_unconnected_regions(samples, grid)

    def test_get_step_sizes(self):
        samples = pd.DataFrame(
            [
                [1, 0, 3, 10],
                [1, 2, 15, 40],
                [1, 6, 6, 20],
                [1, 4, 12, 50],
                [1, 8, 9, 30],
                [1, 10, 18, 60],
            ],
            columns=["id", "x", "y", "z"],
        )
        expected_step_sizes = (2, 3, 10)
        step_sizes = ProcessSamples.get_step_sizes(samples)
        self.assertTupleEqual(expected_step_sizes, step_sizes)

    def test_get_step_size_equal_step_sizes(self):
        array = [2, 4, 8, 6, 12, 10, 4, 6, 10]
        step_size = ProcessSamples.get_step_size(array)
        self.assertEqual(2, step_size)

    def test_get_step_size_unequal_step_sizes(self):
        array = [2, 3, 4, 8, 6, 12, 10, 4, 6, 10]
        step_size = ProcessSamples.get_step_size(array)
        self.assertEqual(2, step_size)

    def test_get_sample_minimums(self):
        samples = pd.DataFrame(
            [
                [1, 0, 3, 10],
                [1, 2, 15, 40],
                [1, 6, 6, 20],
                [1, 4, 12, 50],
                [1, 8, 9, 30],
                [1, 10, 18, 60],
            ],
            columns=["id", "x", "y", "z"],
        )
        expected_minimums = (0, 3, 10)
        minimums = ProcessSamples.get_sample_minimums(samples)
        self.assertTupleEqual(expected_minimums, minimums)

    def test_get_sample_maximums(self):
        samples = pd.DataFrame(
            [
                [1, 0, 3, 10],
                [1, 2, 15, 40],
                [1, 6, 6, 20],
                [1, 4, 12, 50],
                [1, 8, 9, 30],
                [1, 10, 18, 60],
            ],
            columns=["id", "x", "y", "z"],
        )
        expected_maximums = (10, 18, 60)
        maximums = ProcessSamples.get_sample_maximums(samples)
        self.assertTupleEqual(expected_maximums, maximums)

    def test_find_edge_ids_no_padding_low_threshold(self):
        padding = 0
        threshold = 0
        samples = pd.DataFrame(
            [
                [1, 0, 1, 1],
                [1, 1, 1, 2],
                [1, 2, 1, 3],
                [1, 2, 0, 4],
                [1, 3, 0, 5],
                [1, 4, 0, 5],
                [2, 0, 2, 1],
                [2, 1, 2, 2],
                [3, 1, 3, 1],
                [3, 2, 3, 2],
                [3, 2, 4, 3],
            ],
            columns=["id", "x", "y", "z"],
        )
        expected_edge_ids = [1, 2]

        edge_ids = ProcessSamples.find_edge_ids("x", samples, padding, threshold)
        self.assertListEqual(expected_edge_ids, edge_ids)

    def test_find_edge_ids_no_padding_high_threshold(self):
        padding = 0
        threshold = 3
        samples = pd.DataFrame(
            [
                [1, 0, 1, 1],
                [1, 1, 1, 2],
                [1, 2, 1, 3],
                [1, 2, 0, 4],
                [1, 3, 0, 5],
                [1, 4, 0, 5],
                [2, 0, 2, 1],
                [2, 1, 2, 2],
                [3, 1, 3, 1],
                [3, 2, 3, 2],
                [3, 2, 4, 3],
            ],
            columns=["id", "x", "y", "z"],
        )
        expected_edge_ids = []

        edge_ids = ProcessSamples.find_edge_ids("x", samples, padding, threshold)
        self.assertListEqual(expected_edge_ids, edge_ids)

    def test_find_edge_ids_with_padding_low_threshold(self):
        padding = 1
        threshold = 0
        samples = pd.DataFrame(
            [
                [1, 0, 1, 1],
                [1, 1, 1, 2],
                [1, 2, 1, 3],
                [1, 2, 0, 4],
                [1, 3, 0, 5],
                [1, 4, 0, 5],
                [2, 0, 2, 1],
                [2, 1, 2, 2],
                [3, 1, 3, 1],
                [3, 2, 3, 2],
                [3, 2, 4, 3],
            ],
            columns=["id", "x", "y", "z"],
        )
        expected_edge_ids = [1, 2, 3]

        edge_ids = ProcessSamples.find_edge_ids("x", samples, padding, threshold)
        self.assertListEqual(expected_edge_ids, edge_ids)

    def test_find_edge_ids_with_padding_high_threshold(self):
        padding = 1
        threshold = 3
        samples = pd.DataFrame(
            [
                [1, 0, 1, 1],
                [1, 1, 1, 2],
                [1, 2, 1, 3],
                [1, 2, 0, 4],
                [1, 3, 0, 5],
                [1, 4, 0, 5],
                [2, 0, 2, 1],
                [2, 1, 2, 2],
                [3, 1, 3, 1],
                [3, 2, 3, 2],
                [3, 2, 4, 3],
            ],
            columns=["id", "x", "y", "z"],
        )
        expected_edge_ids = [1]

        edge_ids = ProcessSamples.find_edge_ids("x", samples, padding, threshold)
        self.assertListEqual(expected_edge_ids, edge_ids)

    def test_convert_to_integer_array(self):
        samples = pd.DataFrame(
            [
                [1, 0, 3, 10],
                [2, 2, 6, 10],
                [3, 4, 6, 10],
                [4, 6, 9, 10],
                [5, 8, 9, 10],
                [6, 10, 9, 20],
            ],
            columns=["id", "x", "y", "z"],
        )
        steps = (2, 3, 10)
        minimums = (0, 3, 10)
        maximums = (10, 9, 20)

        expected_array = np.array(
            [
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 2, 3, 0, 0, 0],
                    [0, 0, 0, 4, 5, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 6],
                ],
            ]
        )

        array = ProcessSamples.convert_to_integer_array(samples, steps, minimums, maximums)
        self.assertTrue(np.array_equal(expected_array, array))

    def test_convert_to_dataframe(self):
        array = np.array(
            [
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 2, 3, 0, 0, 0],
                    [0, 0, 0, 4, 5, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 6],
                ],
            ]
        )
        steps = (2, 3, 10)
        minimums = (0, 3, 10)

        expected_dataframe = pd.DataFrame(
            [
                [1, 0, 3, 10],
                [2, 2, 6, 10],
                [3, 4, 6, 10],
                [4, 6, 9, 10],
                [5, 8, 9, 10],
                [6, 10, 9, 20],
            ],
            columns=["id", "x", "y", "z"],
        )

        dataframe = ProcessSamples.convert_to_dataframe(array, steps, minimums)
        self.assertTrue(expected_dataframe.equals(dataframe))

    def test_get_minimum_distance(self):
        source = np.array([[0, 0, 0]])
        targets = np.array(
            [
                [3, 2, 1],
                [1, 2, 3],
                [2, 1, 2],
                [3, 1, 2],
            ]
        )

        minimum_distance = ProcessSamples.get_minimum_distance(source, targets)
        self.assertAlmostEqual(3, minimum_distance)


if __name__ == "__main__":
    unittest.main()
