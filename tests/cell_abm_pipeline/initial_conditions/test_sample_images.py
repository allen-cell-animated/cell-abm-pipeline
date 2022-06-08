import unittest
from unittest import mock
from math import sqrt

import numpy as np
from aicsimageio import AICSImage

from cell_abm_pipeline.initial_conditions.sample_images import SampleImages


class TestSampleImages(unittest.TestCase):
    def test_get_image_bounds_returns_bounds(self):
        image_mock = mock.MagicMock(spec=AICSImage)
        image_mock.shape = (1, 2, 3, 4, 5)
        expected_bounds = (5, 4, 3)

        bounds = SampleImages.get_image_bounds(image_mock)

        self.assertEqual(expected_bounds, bounds)

    @mock.patch("cell_abm_pipeline.initial_conditions.sample_images.SCALE_MICRONS_XY", 0.5)
    @mock.patch("cell_abm_pipeline.initial_conditions.sample_images.SCALE_MICRONS_Z", 0.25)
    def test_get_sample_indices_rect_grid_gets_indices(self):
        resolution = 1.0
        bounds = (4, 6, 5)

        expected_indices = [(x, y, z) for x in [0, 2] for y in [0, 2, 4] for z in [0, 4]]

        indices = SampleImages.get_sample_indices(bounds, "rect", resolution)

        self.assertSetEqual(set(expected_indices), set(indices))

    @mock.patch("cell_abm_pipeline.initial_conditions.sample_images.SCALE_MICRONS_XY", 0.5)
    @mock.patch("cell_abm_pipeline.initial_conditions.sample_images.SCALE_MICRONS_Z", 0.25)
    def test_get_sample_indices_hex_grid_gets_indices(self):
        resolution = 1.0
        bounds = (4, 6, 13)
        delta = sqrt(3)

        base_hex_indices = [
            (0, 0),
            (2, 0),
            (1, delta),
            (3, delta),
            (0, 2 * delta),
            (2, 2 * delta),
            (1, 3 * delta),
            (3, 3 * delta),
        ]
        all_hex_indices = [
            [(x, y, 0) for x, y in base_hex_indices],
            [(x + 1, y + delta / 3, 4) for x, y in base_hex_indices],
            [(x, y + 2 * delta / 3, 8) for x, y in base_hex_indices],
            [(x, y, 12) for x, y in base_hex_indices],
        ]
        expected_indices = [
            (round(x), round(y), z)
            for hex_indices in all_hex_indices
            for x, y, z in hex_indices
            if round(x) < 4 and round(y) < 6
        ]

        indices = SampleImages.get_sample_indices(bounds, "hex", resolution)

        self.assertSetEqual(set(expected_indices), set(indices))

    def test_get_sample_indices_invalid_grid_throws_exception(self):
        with self.assertRaises(ValueError):
            bounds = (0, 0, 0)
            grid = "invalid_grid"
            SampleImages.get_sample_indices(bounds, grid)

    def test_get_sample_images_extract_given_samples(self):
        channel = 1
        array = np.array([
            [
                [0, 1, 2],
                [3, 4, 5],
            ],
            [
                [6, 7, 8],
                [9, 10, 11],
            ],
        ])
        sample_indices = [
            (0, 1, 0),
            (1, 1, 2),
            (1, 0, 1),
        ]

        image_mock = mock.MagicMock(spec=AICSImage)
        image_mock.get_image_data.return_value = array
        
        expected_samples = {
            (3, 0, 1, 0),
            (11, 1, 1, 2),
            (7, 1, 0, 1),
        }
        
        samples = SampleImages.get_image_samples(image_mock, sample_indices, channel)
        
        image_mock.get_image_data.assert_called_with("XYZ", T=0, C=channel)
        self.assertSetEqual(expected_samples, set(samples))

if __name__ == "__main__":
    unittest.main()
