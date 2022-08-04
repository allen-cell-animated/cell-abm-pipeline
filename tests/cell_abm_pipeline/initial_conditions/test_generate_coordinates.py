import unittest
from math import sqrt

from cell_abm_pipeline.initial_conditions.generate_coordinates import GenerateCoordinates


class TestGenerateCoordinates(unittest.TestCase):
    def test_get_box_coordinates_rect_grid(self):
        bounds = (4, 6, 5)
        expected_coordinates = [(x, y, z) for x in range(4) for y in range(6) for z in range(5)]

        coordinates = GenerateCoordinates.get_box_coordinates(bounds, "rect")

        self.assertSetEqual(set(expected_coordinates), set(coordinates))

    def test_get_box_coordinates_hex_grid(self):
        bounds = (2, 3, 4)
        delta = sqrt(3)

        base_hex_indices = [
            (0, 0),
            (1, 0),
            (0.5, 0.5 * delta),
            (1.5, 0.5 * delta),
            (0, delta),
            (1, delta),
            (0.5, 1.5 * delta),
            (1.5, 1.5 * delta),
            (0, 2 * delta),
            (1, 2 * delta),
        ]
        all_hex_indices = [
            [(x, y, 0) for x, y in base_hex_indices],
            [(x + 0.5, y + 0.5 * delta / 3, 1) for x, y in base_hex_indices],
            [(x, y + delta / 3, 2) for x, y in base_hex_indices],
            [(x, y, 3) for x, y in base_hex_indices],
        ]
        expected_coordinates = [
            (x, y, z)
            for hex_indices in all_hex_indices
            for x, y, z in hex_indices
            if round(x) < 2 and round(y) < 3
        ]

        coordinates = GenerateCoordinates.get_box_coordinates(bounds, "hex")

        self.assertSetEqual(set(expected_coordinates), set(coordinates))

    def test_get_box_coordinates_invalid_grid_throws_exception(self):
        with self.assertRaises(ValueError):
            bounds = (0, 0, 0)
            grid = "invalid_grid"
            GenerateCoordinates.get_box_coordinates(bounds, grid)

    def test_make_rect_coordinates(self):
        bounds = (4, 6, 5)
        xy_increment = 2
        z_increment = 4

        expected_coordinates = [(x, y, z) for x in [0, 2] for y in [0, 2, 4] for z in [0, 4]]

        coordinates = GenerateCoordinates.make_rect_coordinates(bounds, xy_increment, z_increment)

        self.assertSetEqual(set(expected_coordinates), set(coordinates))

    def test_make_hex_coordinates(self):
        bounds = (4, 6, 13)
        xy_increment = 2
        z_increment = 4
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
        expected_coordinates = [
            (x, y, z)
            for hex_indices in all_hex_indices
            for x, y, z in hex_indices
            if round(x) < 4 and round(y) < 6
        ]

        coordinates = GenerateCoordinates.make_hex_coordinates(bounds, xy_increment, z_increment)

        self.assertSetEqual(set(expected_coordinates), set(coordinates))

    def test_transform_cell_coordinates(self):
        coordinates = [(x, y, z) for x in range(5, 8) for y in range(2, 5) for z in range(2, 4)]
        radius = 1
        x_offset, y_offset = (10, 20)

        xy_coordinates = [
            (-1 + x_offset, 0 + y_offset),
            (0 + x_offset, -1 + y_offset),
            (0 + x_offset, 0 + y_offset),
            (0 + x_offset, 1 + y_offset),
            (1 + x_offset, 0 + y_offset),
        ]
        expected_coordinates = [(x, y, z) for x, y in xy_coordinates for z in range(2, 4)]

        coordinates = GenerateCoordinates.transform_cell_coordinates(
            coordinates, radius, (x_offset, y_offset)
        )

        self.assertSetEqual(set(expected_coordinates), set(coordinates))


if __name__ == "__main__":
    unittest.main()
