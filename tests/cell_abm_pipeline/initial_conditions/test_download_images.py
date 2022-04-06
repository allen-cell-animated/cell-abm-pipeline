import unittest
from unittest import mock

import quilt3
import pandas as pd

from cell_abm_pipeline.initial_conditions.download_images import DownloadImages


class TestDownloadImages(unittest.TestCase):
    @mock.patch("cell_abm_pipeline.initial_conditions.download_images.load_dataframe")
    @mock.patch("cell_abm_pipeline.initial_conditions.download_images.os.path")
    def test_get_fov_files_manifest_exists_loads_manifest(self, path_mock, load_mock):
        working = "working/path/"
        key = "manifest_key"
        pkg = mock.MagicMock(spec=quilt3.Package)

        expected_fov_files = mock.Mock(spec=pd.DataFrame)

        path_mock.isfile.return_value = True
        load_mock.return_value = expected_fov_files

        fov_files = DownloadImages.get_fov_files(working, key, pkg)

        self.assertEqual(expected_fov_files, fov_files)
        load_mock.assert_called_with(working, key)

    @mock.patch("cell_abm_pipeline.initial_conditions.download_images.save_dataframe")
    @mock.patch("cell_abm_pipeline.initial_conditions.download_images.pd")
    @mock.patch("cell_abm_pipeline.initial_conditions.download_images.os.path")
    def test_get_fov_files_manifest_does_not_exist_saves_manifest(
        self, path_mock, pd_mock, save_mock
    ):
        working = "working/path/"
        key = "manifest_key"
        pkg = mock.MagicMock(spec=quilt3.Package)
        pkg.manifest = [
            {"version": "v0", "message": "message"},
            {
                "logical_key": "README.md",
                "physical_keys": ["s3://bucket/README.md"],
                "size": 10,
                "hash": {"type": "SHA256", "value": "readme"},
                "meta": {"user_meta": {}},
            },
            {
                "logical_key": "folder/key_c.ome.tif",
                "physical_keys": ["s3://bucket/folder/key_c.ome.tiff"],
                "size": 200,
                "hash": {"type": "SHA256", "value": "hash_c"},
                "meta": {
                    "user_meta": {
                        "associates": {
                            "crop_raw": "crop_raw/crow_raw_c.ome.tif",
                            "crop_seg": "crop_seg/crop_seg_c.ome.tif",
                            "fov_path": "fov_path/fov_path_2.ome.tiff",
                            "fov_seg_path": "fov_seg_path/fov_seg_path_2.ome.tiff",
                            "struct_seg_path": "struct_seg_path/struct_seg_path_2.tiff",
                        }
                    }
                },
            },
            {
                "logical_key": "folder/key_a.ome.tif",
                "physical_keys": ["s3://bucket/folder/key_a.ome.tiff"],
                "size": 100,
                "hash": {"type": "SHA256", "value": "hash_a"},
                "meta": {
                    "user_meta": {
                        "associates": {
                            "crop_raw": "crop_raw/crow_raw_a.ome.tif",
                            "crop_seg": "crop_seg/crop_seg_a.ome.tif",
                            "fov_path": "fov_path/fov_path_1.ome.tiff",
                            "fov_seg_path": "fov_seg_path/fov_seg_path_1.ome.tiff",
                            "struct_seg_path": "struct_seg_path/struct_seg_path_1.tiff",
                        }
                    }
                },
            },
            {
                "logical_key": "folder/key_b.ome.tif",
                "physical_keys": ["s3://bucket/folder/key_b.ome.tiff"],
                "size": 101,
                "hash": {"type": "SHA256", "value": "hash_b"},
                "meta": {
                    "user_meta": {
                        "associates": {
                            "crop_raw": "crop_raw/crow_raw_b.ome.tif",
                            "crop_seg": "crop_seg/crop_seg_b.ome.tif",
                            "fov_path": "fov_path/fov_path_1.ome.tiff",
                            "fov_seg_path": "fov_seg_path/fov_seg_path_1.ome.tiff",
                            "struct_seg_path": "struct_seg_path/struct_seg_path_1.tiff",
                        }
                    }
                },
            },
        ]

        expected_fov_files = mock.Mock(spec=pd.DataFrame)
        fov_columns = ["fov_seg_path", "fov_path", "status"]
        fov_entries = [
            ("fov_seg_path/fov_seg_path_1.ome.tiff", "fov_path/fov_path_1.ome.tiff", "available"),
            ("fov_seg_path/fov_seg_path_2.ome.tiff", "fov_path/fov_path_2.ome.tiff", "available"),
        ]

        path_mock.isfile.return_value = False
        pd_mock.DataFrame.return_value = expected_fov_files

        fov_files = DownloadImages.get_fov_files(working, key, pkg)

        self.assertEqual(expected_fov_files, fov_files)
        save_mock.assert_called_with(working, key, fov_files, index=False)
        pd_mock.DataFrame.assert_called_with(fov_entries, columns=fov_columns)


if __name__ == "__main__":
    unittest.main()
