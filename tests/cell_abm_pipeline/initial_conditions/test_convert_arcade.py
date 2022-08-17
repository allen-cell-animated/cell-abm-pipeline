import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from cell_abm_pipeline.initial_conditions.__config__ import (
    POTTS_TERMS,
    VOLUME_AVGS,
    VOLUME_STDS,
    CRITICAL_VOLUME_AVGS,
    CRITICAL_VOLUME_STDS,
    HEIGHT_AVGS,
    HEIGHT_STDS,
    CRITICAL_HEIGHT_AVGS,
    CRITICAL_HEIGHT_STDS,
    CELL_STATE_THRESHOLD_FRACTIONS,
)
from cell_abm_pipeline.initial_conditions.convert_arcade import ConvertARCADE


class TestConvertARCADE(unittest.TestCase):
    def test_transform_sample_voxels_no_margin_no_reference(self):
        margins = (0, 0, 0)
        samples = pd.DataFrame(
            [
                [1, 0, 3, 10],
                [2, 2, 15, 40],
                [3, 6, 6, 20],
                [4, 4, 12, 50],
                [5, 8, 9, 30],
                [6, 10, 18, 60],
            ],
            columns=["id", "x", "y", "z"],
        )

        expected = pd.DataFrame(
            [
                [1, 1, 1, 1],
                [2, 2, 5, 4],
                [3, 4, 2, 2],
                [4, 3, 4, 5],
                [5, 5, 3, 3],
                [6, 6, 6, 6],
            ],
            columns=["id", "x", "y", "z"],
        )

        transformed = ConvertARCADE.transform_sample_voxels(samples, margins)
        self.assertTrue(expected.equals(transformed))

    def test_transform_sample_voxels_with_margin_no_reference(self):
        margins = (3, 4, 5)
        samples = pd.DataFrame(
            [
                [1, 0, 3, 10],
                [2, 2, 15, 40],
                [3, 6, 6, 20],
                [4, 4, 12, 50],
                [5, 8, 9, 30],
                [6, 10, 18, 60],
            ],
            columns=["id", "x", "y", "z"],
        )

        expected = pd.DataFrame(
            [
                [1, 4, 5, 6],
                [2, 5, 9, 9],
                [3, 7, 6, 7],
                [4, 6, 8, 10],
                [5, 8, 7, 8],
                [6, 9, 10, 11],
            ],
            columns=["id", "x", "y", "z"],
        )

        transformed = ConvertARCADE.transform_sample_voxels(samples, margins)
        self.assertTrue(expected.equals(transformed))

    def test_transform_sample_voxels_no_margin_with_reference(self):
        margins = (0, 0, 0)
        reference = pd.DataFrame(
            [
                [1, 0, 3, 10],
                [2, 2, 15, 40],
                [3, 6, 6, 20],
                [4, 4, 12, 50],
                [5, 8, 9, 30],
                [6, 10, 18, 60],
            ],
            columns=["id", "x", "y", "z"],
        )
        samples = pd.DataFrame(
            [
                [7, 4, 21, 30],
                [8, 8, 21, 20],
                [9, 12, 21, 20],
            ],
            columns=["id", "x", "y", "z"],
        )

        expected = pd.DataFrame(
            [
                [7, 3, 7, 3],
                [8, 5, 7, 2],
                [9, 7, 7, 2],
            ],
            columns=["id", "x", "y", "z"],
        )

        transformed = ConvertARCADE.transform_sample_voxels(samples, margins, reference)
        self.assertTrue(expected.equals(transformed))

    def test_transform_sample_voxels_with_margin_with_reference(self):
        margins = (5, 4, 3)
        reference = pd.DataFrame(
            [
                [1, 0, 3, 10],
                [2, 2, 15, 40],
                [3, 6, 6, 20],
                [4, 4, 12, 50],
                [5, 8, 9, 30],
                [6, 10, 18, 60],
            ],
            columns=["id", "x", "y", "z"],
        )
        samples = pd.DataFrame(
            [
                [7, 4, 21, 30],
                [8, 8, 21, 20],
                [9, 12, 21, 20],
            ],
            columns=["id", "x", "y", "z"],
        )

        expected = pd.DataFrame(
            [
                [7, 8, 11, 6],
                [8, 10, 11, 5],
                [9, 12, 11, 5],
            ],
            columns=["id", "x", "y", "z"],
        )

        transformed = ConvertARCADE.transform_sample_voxels(samples, margins, reference)
        self.assertTrue(expected.equals(transformed))

    def test_calculate_sample_voxels(self):
        margins = (10, 20, 30)
        samples = pd.DataFrame(
            [
                [1, 0, 3, 10],
                [2, 2, 3, 20],
                [3, 6, 6, 30],
                [4, 4, 6, 20],
                [5, 8, 6, 30],
            ],
            columns=["id", "x", "y", "z"],
        )
        expected_bounds = (27, 44, 65)

        bounds = ConvertARCADE.calculate_sample_bounds(samples, margins)
        self.assertTupleEqual(expected_bounds, bounds)

    def test_make_setup_file_default_parameters_no_region(self):
        init = 10
        bounds = (20, 30, 40)

        potts_terms = "".join([f'<potts.term id="{term}" />' for term in POTTS_TERMS])
        expected_setup = f"""
            <set>
                <series name="ARCADE" interval="1" start="0" end="0" dt="1" ds="1"
                    ticks="1" length="{bounds[0]}" width="{bounds[1]}" height="{bounds[2]}">
                    <potts>{potts_terms}</potts>
                    <agents>
                        <populations>
                            <population id="X" init="{init}" />
                        </populations>
                    </agents>
                </series>
            </set>
        """
        expected = ET.tostring(ET.fromstring(expected_setup))
        expected = "".join(line.strip() for line in expected.decode().split("\n"))

        setup = ConvertARCADE.make_setup_file(init, bounds)
        setup = "".join(line.strip() for line in setup.decode().split("\n"))

        self.assertEqual(expected, setup)

    def test_make_setup_file_default_parameters_with_regions(self):
        init = 10
        bounds = (20, 30, 40)
        regions = ["REGION_A", "REGION_B"]

        potts_terms = "".join([f'<potts.term id="{term}" />' for term in POTTS_TERMS])
        expected_setup = f"""
            <set>
                <series name="ARCADE" interval="1" start="0" end="0" dt="1" ds="1"
                    ticks="1" length="{bounds[0]}" width="{bounds[1]}" height="{bounds[2]}">
                    <potts>{potts_terms}</potts>
                    <agents>
                        <populations>
                            <population id="X" init="{init}">
                                <population.region id="{regions[0]}" />
                                <population.region id="{regions[1]}" />
                            </population>
                        </populations>
                    </agents>
                </series>
            </set>
        """
        expected = ET.tostring(ET.fromstring(expected_setup))
        expected = "".join(line.strip() for line in expected.decode().split("\n"))

        setup = ConvertARCADE.make_setup_file(init, bounds, regions)
        setup = "".join(line.strip() for line in setup.decode().split("\n"))

        self.assertEqual(expected, setup)

    def test_make_setup_file_given_parameters_no_region(self):
        init = 10
        bounds = (20, 30, 40)
        potts_terms = ("TERM_1", "TERM_2")

        expected_setup = f"""
            <set>
                <series name="ARCADE" interval="1" start="0" end="0" dt="1" ds="1"
                    ticks="1" length="{bounds[0]}" width="{bounds[1]}" height="{bounds[2]}">
                    <potts>
                        <potts.term id="{potts_terms[0]}" />
                        <potts.term id="{potts_terms[1]}" />
                    </potts>
                    <agents>
                        <populations>
                            <population id="X" init="{init}" />
                        </populations>
                    </agents>
                </series>
            </set>
        """
        expected = ET.tostring(ET.fromstring(expected_setup))
        expected = "".join(line.strip() for line in expected.decode().split("\n"))

        setup = ConvertARCADE.make_setup_file(init, bounds, terms=potts_terms)
        setup = "".join(line.strip() for line in setup.decode().split("\n"))

        self.assertEqual(expected, setup)

    def test_make_setup_file_given_parameters_with_regions(self):
        init = 10
        bounds = (20, 30, 40)
        regions = ["REGION_A", "REGION_B"]
        potts_terms = ("TERM_1", "TERM_2")

        expected_setup = f"""
            <set>
                <series name="ARCADE" interval="1" start="0" end="0" dt="1" ds="1"
                    ticks="1" length="{bounds[0]}" width="{bounds[1]}" height="{bounds[2]}">
                    <potts>
                        <potts.term id="{potts_terms[0]}" />
                        <potts.term id="{potts_terms[1]}" />
                    </potts>
                    <agents>
                        <populations>
                            <population id="X" init="{init}">
                                <population.region id="{regions[0]}" />
                                <population.region id="{regions[1]}" />
                            </population>
                        </populations>
                    </agents>
                </series>
            </set>
        """
        expected = ET.tostring(ET.fromstring(expected_setup))
        expected = "".join(line.strip() for line in expected.decode().split("\n"))

        setup = ConvertARCADE.make_setup_file(init, bounds, regions, terms=potts_terms)
        setup = "".join(line.strip() for line in setup.decode().split("\n"))

        self.assertEqual(expected, setup)

    def test_filter_valid_samples_no_regions_does_nothing(self):
        samples = pd.DataFrame(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
            ],
            columns=["id", "x", "y", "z"],
        )
        filtered_samples = ConvertARCADE.filter_valid_samples(samples)
        self.assertTrue(samples.equals(filtered_samples))

    def test_filter_valid_samples_with_regions(self):
        samples = pd.DataFrame(
            [
                [1, 1, 2, 3, "REGION_A"],
                [1, 4, 5, 6, "REGION_B"],
                [1, 7, 8, 9, "REGION_C"],
                [2, 10, 11, 12, "REGION_C"],
                [2, 13, 14, 15, "REGION_B"],
                [3, 16, 17, 18, "REGION_A"],
                [3, 19, 20, 21, "REGION_B"],
                [3, 22, 23, 24, "REGION_C"],
            ],
            columns=["id", "x", "y", "z", "region"],
        )

        expected = pd.DataFrame(
            [
                [1, 1, 2, 3, "REGION_A"],
                [1, 4, 5, 6, "REGION_B"],
                [1, 7, 8, 9, "REGION_C"],
                [3, 16, 17, 18, "REGION_A"],
                [3, 19, 20, 21, "REGION_B"],
                [3, 22, 23, 24, "REGION_C"],
            ],
            columns=["id", "x", "y", "z", "region"],
        )

        filtered_samples = ConvertARCADE.filter_valid_samples(samples)
        self.assertTrue(expected.equals(filtered_samples))

    def test_filter_cell_reference_invalid_id(self):
        cell_id = 3
        reference = pd.DataFrame([{"ID": 1}, {"ID": 2}])
        cell_reference = ConvertARCADE.filter_cell_reference(cell_id, reference)
        self.assertDictEqual({}, cell_reference)

    def test_filter_cell_reference_valid_id(self):
        cell_id = 2
        cell_1_data = {"ID": 1, "key1": "a", "key2": 2}
        cell_2_data = {"ID": 2, "key1": "b", "key2": 4}
        reference = pd.DataFrame([cell_1_data, cell_2_data])

        cell_reference = ConvertARCADE.filter_cell_reference(cell_id, reference)
        self.assertDictEqual(cell_2_data, cell_reference)

    @mock.patch.object(ConvertARCADE, "get_cell_critical_volume")
    @mock.patch.object(ConvertARCADE, "get_cell_critical_height")
    @mock.patch.object(ConvertARCADE, "get_cell_state")
    def test_convert_to_cell_no_reference_no_region(
        self, get_cell_state, get_cell_critical_height, get_cell_critical_volume
    ):
        cell_id = 10
        state = "STATE_PHASE"
        num_samples = 100
        z_delta = 10

        samples = pd.DataFrame(np.zeros((num_samples, 4)), columns=["id", "x", "y", "z"])
        samples["z"] = np.linspace(0, z_delta, num_samples)

        get_cell_critical_volume.side_effect = lambda v, *args: v / 2
        get_cell_critical_height.side_effect = lambda v, *args: v / 3
        get_cell_state.return_value = state

        expected_cell = {
            "id": cell_id,
            "parent": 0,
            "pop": 1,
            "age": 0,
            "divisions": 0,
            "state": "STATE",
            "phase": "STATE_PHASE",
            "voxels": num_samples,
            "criticals": [num_samples / 2, z_delta / 3],
        }

        cell = ConvertARCADE.convert_to_cell(cell_id, samples, {})
        self.assertDictEqual(expected_cell, cell)

    @mock.patch.object(ConvertARCADE, "get_cell_critical_volume")
    @mock.patch.object(ConvertARCADE, "get_cell_critical_height")
    @mock.patch.object(ConvertARCADE, "get_cell_state")
    def test_convert_to_cell_with_reference_no_region(
        self, get_cell_state, get_cell_critical_height, get_cell_critical_volume
    ):
        critical_volume = 1500
        critical_height = 20

        reference = {
            "volume": critical_volume,
            "height": critical_height,
        }

        cell_id = 10
        state = "STATE_PHASE"
        num_samples = 100
        z_delta = 10

        samples = pd.DataFrame(np.zeros((num_samples, 4)), columns=["id", "x", "y", "z"])
        samples["z"] = np.linspace(0, z_delta, num_samples)

        get_cell_critical_volume.side_effect = lambda v, *args: v / 2
        get_cell_critical_height.side_effect = lambda v, *args: v / 3
        get_cell_state.return_value = state

        expected_cell = {
            "id": cell_id,
            "parent": 0,
            "pop": 1,
            "age": 0,
            "divisions": 0,
            "state": "STATE",
            "phase": "STATE_PHASE",
            "voxels": num_samples,
            "criticals": [critical_volume / 2, critical_height / 3],
        }

        cell = ConvertARCADE.convert_to_cell(cell_id, samples, reference)
        self.assertDictEqual(expected_cell, cell)
        get_cell_state.assert_called_with(num_samples, critical_volume / 2)

    @mock.patch.object(ConvertARCADE, "get_cell_critical_volume")
    @mock.patch.object(ConvertARCADE, "get_cell_critical_height")
    @mock.patch.object(ConvertARCADE, "get_cell_state")
    def test_convert_to_cell_no_reference_with_region(
        self, get_cell_state, get_cell_critical_height, get_cell_critical_volume
    ):
        cell_id = 10
        state = "STATE_PHASE"
        region_a = "REGION_A"
        region_b = "REGION_B"
        num_samples = 100
        num_samples_region_a = 60
        num_samples_region_b = 40
        z_delta_region_a = 3
        z_delta_region_b = 5

        samples = pd.DataFrame(np.zeros((num_samples, 4)), columns=["id", "x", "y", "z"])
        samples["region"] = [region_a] * num_samples_region_a + [region_b] * num_samples_region_b
        samples["z"].loc[: num_samples_region_a - 1] = np.linspace(
            0, z_delta_region_a, num_samples_region_a
        )
        samples["z"].loc[num_samples_region_a:num_samples] = np.linspace(
            0, z_delta_region_b, num_samples_region_b
        )

        get_cell_critical_volume.side_effect = lambda v, *args: v / 2
        get_cell_critical_height.side_effect = lambda v, *args: v / 3
        get_cell_state.return_value = state

        expected_cell = {
            "id": cell_id,
            "parent": 0,
            "pop": 1,
            "age": 0,
            "divisions": 0,
            "state": "STATE",
            "phase": "STATE_PHASE",
            "voxels": num_samples,
            "criticals": [num_samples / 2, z_delta_region_b / 3],
            "regions": [
                {
                    "region": region_a,
                    "voxels": num_samples_region_a,
                    "criticals": [num_samples_region_a / 2, z_delta_region_a / 3],
                },
                {
                    "region": region_b,
                    "voxels": num_samples_region_b,
                    "criticals": [num_samples_region_b / 2, z_delta_region_b / 3],
                },
            ],
        }

        cell = ConvertARCADE.convert_to_cell(cell_id, samples, {})
        self.assertDictEqual(expected_cell, cell)

    @mock.patch.object(ConvertARCADE, "get_cell_critical_volume")
    @mock.patch.object(ConvertARCADE, "get_cell_critical_height")
    @mock.patch.object(ConvertARCADE, "get_cell_state")
    def test_convert_to_cell_with_reference_with_region(
        self, get_cell_state, get_cell_critical_height, get_cell_critical_volume
    ):
        region_a = "REGION_A"
        region_b = "REGION_B"
        critical_volume = 1000
        critical_volumes = {region_a: 500, region_b: 600}
        critical_height = 100
        critical_heights = {region_a: 50, region_b: 60}

        reference = {
            "volume": critical_volume,
            "height": critical_height,
            f"volume.{region_a}": critical_volumes[region_a],
            f"height.{region_a}": critical_heights[region_a],
            f"volume.{region_b}": critical_volumes[region_b],
            f"height.{region_b}": critical_heights[region_b],
        }

        cell_id = 10
        state = "STATE_PHASE"
        num_samples = 100
        num_samples_region_a = 60
        num_samples_region_b = 40
        z_delta_region_a = 3
        z_delta_region_b = 5

        samples = pd.DataFrame(np.zeros((num_samples, 4)), columns=["id", "x", "y", "z"])
        samples["region"] = [region_a] * num_samples_region_a + [region_b] * num_samples_region_b
        samples["z"].loc[: num_samples_region_a - 1] = np.linspace(
            0, z_delta_region_a, num_samples_region_a
        )
        samples["z"].loc[num_samples_region_a:num_samples] = np.linspace(
            0, z_delta_region_b, num_samples_region_b
        )

        get_cell_critical_volume.side_effect = lambda v, *args: v / 2
        get_cell_critical_height.side_effect = lambda v, *args: v / 3
        get_cell_state.return_value = state

        expected_cell = {
            "id": cell_id,
            "parent": 0,
            "pop": 1,
            "age": 0,
            "divisions": 0,
            "state": "STATE",
            "phase": "STATE_PHASE",
            "voxels": num_samples,
            "criticals": [critical_volume / 2, critical_height / 3],
            "regions": [
                {
                    "region": region_a,
                    "voxels": num_samples_region_a,
                    "criticals": [critical_volumes[region_a] / 2, critical_heights[region_a] / 3],
                },
                {
                    "region": region_b,
                    "voxels": num_samples_region_b,
                    "criticals": [critical_volumes[region_b] / 2, critical_heights[region_b] / 3],
                },
            ],
        }

        cell = ConvertARCADE.convert_to_cell(cell_id, samples, reference)
        self.assertDictEqual(expected_cell, cell)
        get_cell_state.assert_called_with(num_samples, critical_volume / 2)

    @mock.patch.object(ConvertARCADE, "get_location_center")
    @mock.patch.object(ConvertARCADE, "get_location_voxels")
    def test_convert_to_location_no_region(self, get_location_voxels, get_location_center):
        center = (1, 2, 3)
        voxels = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]

        get_location_center.return_value = center
        get_location_voxels.return_value = voxels

        cell_id = 10
        samples = pd.DataFrame([[0, 0, 0, 0]], columns=["id", "x", "y", "z"])

        expected_location = {
            "id": cell_id,
            "center": center,
            "location": [{"region": "UNDEFINED", "voxels": voxels}],
        }

        location = ConvertARCADE.convert_to_location(cell_id, samples)
        self.assertDictEqual(expected_location, location)

    @mock.patch.object(ConvertARCADE, "get_location_center")
    @mock.patch.object(ConvertARCADE, "get_location_voxels")
    def test_convert_to_location_with_region(self, get_location_voxels, get_location_center):
        region_a = "REGION_A"
        region_b = "REGION_B"
        center = (1, 2, 3)
        voxels_region_a = [(1, 2, 3), (4, 5, 6)]
        voxels_region_b = [(7, 8, 9), (10, 11, 12)]
        voxels = {region_a: voxels_region_a, region_b: voxels_region_b}

        get_location_center.return_value = center
        get_location_voxels.side_effect = lambda v, *args: voxels[args[0]] if args else []

        cell_id = 10
        samples = pd.DataFrame([[0, 0, 0, 0], [0, 0, 0, 0]], columns=["id", "x", "y", "z"])
        samples["region"] = [region_a, region_b]

        expected_location = {
            "id": cell_id,
            "center": center,
            "location": [
                {"region": region_a, "voxels": voxels_region_a},
                {"region": region_b, "voxels": voxels_region_b},
            ],
        }

        location = ConvertARCADE.convert_to_location(cell_id, samples)
        self.assertDictEqual(expected_location, location)

    def test_get_cell_critical_volume_default_parameters(self):
        volume = 1000
        avgs = VOLUME_AVGS["DEFAULT"]
        stds = VOLUME_STDS["DEFAULT"]
        critical_avgs = CRITICAL_VOLUME_AVGS["DEFAULT"]
        critical_stds = CRITICAL_VOLUME_STDS["DEFAULT"]
        expected_volume = ((volume - avgs) / stds) * critical_stds + critical_avgs

        critical_volume = ConvertARCADE.get_cell_critical_volume(volume)
        self.assertAlmostEqual(expected_volume, critical_volume, places=5)

    def test_get_cell_critical_volume_given_parameters(self):
        region = "REGION"
        volume = 1000
        avgs = {region: 100}
        stds = {region: 10}
        critical_avgs = {region: 200}
        critical_stds = {region: 30}
        expected_volume = ((volume - 100) / 10) * 30 + 200

        critical_volume = ConvertARCADE.get_cell_critical_volume(
            volume, region, avgs, stds, critical_avgs, critical_stds
        )
        self.assertAlmostEqual(expected_volume, critical_volume, places=5)

    def test_get_cell_critical_height_default_parameters(self):
        height = 10
        avgs = HEIGHT_AVGS["DEFAULT"]
        stds = HEIGHT_STDS["DEFAULT"]
        critical_avgs = CRITICAL_HEIGHT_AVGS["DEFAULT"]
        critical_stds = CRITICAL_HEIGHT_STDS["DEFAULT"]
        expected_height = ((height - avgs) / stds) * critical_stds + critical_avgs

        critical_height = ConvertARCADE.get_cell_critical_height(height)
        self.assertAlmostEqual(expected_height, critical_height, places=5)

    def test_get_cell_critical_height_given_parameters(self):
        region = "REGION"
        height = 10
        avgs = {region: 100}
        stds = {region: 10}
        critical_avgs = {region: 200}
        critical_stds = {region: 30}
        expected_height = ((height - 100) / 10) * 30 + 200

        critical_height = ConvertARCADE.get_cell_critical_height(
            height, region, avgs, stds, critical_avgs, critical_stds
        )
        self.assertAlmostEqual(expected_height, critical_height, places=5)

    def test_get_cell_state_default_parameters(self):
        critical_volume = 1000

        volumes = [
            0,
            CELL_STATE_THRESHOLD_FRACTIONS["APOPTOTIC_LATE"] * critical_volume - 1,
            CELL_STATE_THRESHOLD_FRACTIONS["APOPTOTIC_LATE"] * critical_volume,
            CELL_STATE_THRESHOLD_FRACTIONS["APOPTOTIC_EARLY"] * critical_volume - 1,
            CELL_STATE_THRESHOLD_FRACTIONS["APOPTOTIC_EARLY"] * critical_volume,
            CELL_STATE_THRESHOLD_FRACTIONS["PROLIFERATIVE_G1"] * critical_volume - 1,
            CELL_STATE_THRESHOLD_FRACTIONS["PROLIFERATIVE_G1"] * critical_volume,
            CELL_STATE_THRESHOLD_FRACTIONS["PROLIFERATIVE_S"] * critical_volume - 1,
            CELL_STATE_THRESHOLD_FRACTIONS["PROLIFERATIVE_S"] * critical_volume,
            CELL_STATE_THRESHOLD_FRACTIONS["PROLIFERATIVE_G2"] * critical_volume - 1,
            CELL_STATE_THRESHOLD_FRACTIONS["PROLIFERATIVE_G2"] * critical_volume,
            2 * critical_volume,
        ]
        states = [
            "APOPTOTIC_LATE",
            "APOPTOTIC_LATE",
            "APOPTOTIC_EARLY",
            "APOPTOTIC_EARLY",
            "PROLIFERATIVE_G1",
            "PROLIFERATIVE_G1",
            "PROLIFERATIVE_S",
            "PROLIFERATIVE_S",
            "PROLIFERATIVE_G2",
            "PROLIFERATIVE_G2",
            "PROLIFERATIVE_G2",
        ]

        for volume, state in zip(volumes, states):
            with self.subTest(given=volume):
                cell_state = ConvertARCADE.get_cell_state(volume, critical_volume)
                self.assertEqual(state, cell_state)

    def test_get_cell_state_given_parameters(self):
        critical_volume = 1000

        state_1_name = "STATE_1"
        state_2_name = "STATE_2"
        state_1_threshold_fraction = 0.5
        state_2_threshold_fraction = 1.3
        threshold_fractions = {
            state_1_name: state_1_threshold_fraction,
            state_2_name: state_2_threshold_fraction,
        }

        volumes = [0, 499, 500, 1299, 1300, 2000]
        states = [state_1_name, state_1_name, state_2_name, state_2_name, state_2_name]

        for volume, state in zip(volumes, states):
            with self.subTest(given=volume):
                cell_state = ConvertARCADE.get_cell_state(
                    volume, critical_volume, threshold_fractions
                )
                self.assertEqual(state, cell_state)

    def test_get_location_center(self):
        samples = pd.DataFrame(
            [
                [1, 1, 2, 6],
                [1, 2, 4, 7],
                [1, 3, 6, 8],
                [1, 4, 8, 9],
                [1, 5, 10, 10],
            ],
            columns=["id", "x", "y", "z"],
        )
        expected_center = (3, 6, 8)
        center = ConvertARCADE.get_location_center(samples)
        self.assertTupleEqual(expected_center, center)

    def test_get_location_voxels_no_region(self):
        samples = pd.DataFrame(
            [
                [1, 1, 2, 6],
                [1, 2, 4, 7],
                [1, 3, 6, 8],
                [1, 4, 8, 9],
                [1, 5, 10, 10],
            ],
            columns=["id", "x", "y", "z"],
        )

        expected_voxels = [
            (1, 2, 6),
            (2, 4, 7),
            (3, 6, 8),
            (4, 8, 9),
            (5, 10, 10),
        ]

        voxels = ConvertARCADE.get_location_voxels(samples)
        self.assertListEqual(expected_voxels, voxels)

    def test_get_location_voxels_with_region(self):
        region = "REGION"
        samples = pd.DataFrame(
            [
                [1, 1, 2, 6, region],
                [1, 2, 4, 7, "NONE"],
                [1, 3, 6, 8, region],
                [1, 4, 8, 9, "NONE"],
                [1, 5, 10, 10, region],
            ],
            columns=["id", "x", "y", "z", "region"],
        )

        expected_voxels = [
            (1, 2, 6),
            (3, 6, 8),
            (5, 10, 10),
        ]

        voxels = ConvertARCADE.get_location_voxels(samples, region)
        self.assertListEqual(expected_voxels, voxels)


if __name__ == "__main__":
    unittest.main()
