import unittest
from unittest import mock

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

    def test_select_cells(self):
        cell_1_data = [[1, 1, 2, 3]]
        cell_2_data = [[2, 1, 2, 1]]
        cell_3_data = [[3, 1, 2, 2]]
        sample_data = cell_1_data + cell_2_data + cell_3_data
        samples = pd.DataFrame(sample_data, columns=["id", "x", "y", "z"])

        select = [2, 3]
        expected_data = cell_2_data + cell_3_data
        expected_samples = pd.DataFrame(expected_data, columns=["id", "x", "y", "z"])

        selected_samples = ProcessSamples.select_cells(samples, select)

        self.assertTrue(expected_samples.equals(selected_samples))

    def test_remove_edge_cells_rect_grid(self):
        cell_1_data = [[1, 0, 2, 1], [1, 2, 0, 2], [1, 2, 4, 3], [1, 4, 2, 4]]
        cell_2_data = [[2, 1, 1, 1], [2, 2, 1, 2]]
        cell_3_data = [[3, 0, 3, 1], [3, 1, 3, 2]]
        cell_4_data = [[4, 3, 3, 1], [4, 3, 4, 2]]

        sample_data = cell_1_data + cell_2_data + cell_3_data + cell_4_data
        samples = pd.DataFrame(sample_data, columns=["id", "x", "y", "z"])

        expected_data = cell_2_data + cell_3_data + cell_4_data
        expected_samples = pd.DataFrame(expected_data, columns=["id", "x", "y", "z"])

        filtered_samples = ProcessSamples.remove_edge_cells(samples, grid="rect")
        self.assertTrue(expected_samples.equals(filtered_samples))

    def test_get_step_sizes(self):
        sample_data = [
            [1, 0, 3, 10],
            [1, 2, 15, 40],
            [1, 6, 6, 20],
            [1, 4, 12, 50],
            [1, 8, 9, 30],
            [1, 10, 18, 60],
        ]
        samples = pd.DataFrame(sample_data, columns=["id", "x", "y", "z"])
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

    def test_find_edge_ids_no_padding_low_threshold(self):
        padding = 0
        threshold = 0
        sample_data = [
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
        ]
        samples = pd.DataFrame(sample_data, columns=["id", "x", "y", "z"])
        expected_edge_ids = [1, 2]

        edge_ids = ProcessSamples.find_edge_ids("x", samples, padding, threshold)
        self.assertListEqual(expected_edge_ids, edge_ids)

    def test_find_edge_ids_no_padding_high_threshold(self):
        padding = 0
        threshold = 3
        sample_data = [
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
        ]
        samples = pd.DataFrame(sample_data, columns=["id", "x", "y", "z"])
        expected_edge_ids = []

        edge_ids = ProcessSamples.find_edge_ids("x", samples, padding, threshold)
        self.assertListEqual(expected_edge_ids, edge_ids)

    def test_find_edge_ids_with_padding_low_threshold(self):
        padding = 1
        threshold = 0
        sample_data = [
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
        ]
        samples = pd.DataFrame(sample_data, columns=["id", "x", "y", "z"])
        expected_edge_ids = [1, 2, 3]

        edge_ids = ProcessSamples.find_edge_ids("x", samples, padding, threshold)
        self.assertListEqual(expected_edge_ids, edge_ids)

    def test_find_edge_ids_with_padding_high_threshold(self):
        padding = 1
        threshold = 3
        sample_data = [
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
        ]
        samples = pd.DataFrame(sample_data, columns=["id", "x", "y", "z"])
        expected_edge_ids = [1]

        edge_ids = ProcessSamples.find_edge_ids("x", samples, padding, threshold)
        self.assertListEqual(expected_edge_ids, edge_ids)


if __name__ == "__main__":
    unittest.main()