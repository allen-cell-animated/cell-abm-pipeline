import io

import numpy as np
from skimage import measure

from cell_abm_pipeline.utilities.load import load_tar, load_tar_member
from cell_abm_pipeline.utilities.save import save_buffer
from cell_abm_pipeline.utilities.keys import make_folder_key, make_file_key, make_full_key


class ArcadeToMesh:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "input": make_folder_key(context.name, "data", "LOCATIONS", False),
            "output": make_folder_key(context.name, "converted", "MESH", False),
        }
        self.files = {
            "input": make_file_key(context.name, ["LOCATIONS", "tar", "xz"], "%s", ""),
            "output": make_file_key(context.name, ["obj"], "%s_%06d_%02d", ""),
        }

    def run(self, frames=(0, 1, 1)):
        for key in self.context.keys:
            self.arcade_to_mesh(key, frames)

    def arcade_to_mesh(self, key, frames):
        data_key = make_full_key(self.folders, self.files, "input", (key))
        data_tar = load_tar(self.context.working, data_key)

        for frame in np.arange(*frames):
            print(f"Converting frame {frame} to meshes ...")
            member_name = f"{self.context.name}_{key}_{frame:06d}.LOCATIONS.json"
            tar_member = load_tar_member(data_tar, member_name)
            meshes = self.convert_frame_meshes(tar_member)

            for cell_id, mesh in meshes:
                output_key = make_full_key(
                    self.folders, self.files, "output", (key, frame, cell_id)
                )

                with io.BytesIO() as buffer:
                    buffer.write(mesh.encode("utf-8"))
                    save_buffer(self.context.working, output_key, buffer)

    @staticmethod
    def convert_frame_meshes(locations):
        meshes = []

        for location in locations:
            voxels = [(z, y, x) for region in location["location"] for x, y, z in region["voxels"]]
            array = ArcadeToMesh.make_mesh_array(voxels)
            verts, faces, normals = ArcadeToMesh.make_array_mesh(array)
            mesh = ArcadeToMesh.make_mesh_object(verts, faces, normals)
            meshes.append((location["id"], mesh))

        return meshes

    @staticmethod
    def make_mesh_array(voxels):
        mins = np.min(voxels, axis=0)
        maxs = np.max(voxels, axis=0)
        height, width, length = np.subtract(maxs, mins) + 3
        array = np.zeros((height, width, length), dtype=np.uint8)

        voxels_transposed = [voxel - mins + 1 for voxel in voxels]
        array[tuple(np.transpose(voxels_transposed))] = 7

        # Get set of zero neighbors for all voxels.
        offsets = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        neighbors = {
            (z + i, y + j, x + k)
            for z, y, x in voxels_transposed
            for k, j, i in offsets
            if array[z + k, y + j, x + i] == 0
        }

        # Remove invalid neighbors on borders.
        neighbors = {
            (z, y, x)
            for z, y, x in neighbors
            if x != 0
            and y != 0
            and z != 0
            and x != length - 1
            and y != width - 1
            and z != height - 1
        }

        # Smooth array levels based on neighbor counts.
        for z, y, x in neighbors:
            array[z, y, x] = sum([array[z + k, y + j, x + i] == 7 for k, j, i in offsets]) + 1

        return array

    @staticmethod
    def make_array_mesh(array):
        verts, faces, normals, _ = measure.marching_cubes(array, level=3, allow_degenerate=False)
        return verts, faces, normals

    @staticmethod
    def make_mesh_object(verts, faces, normals):
        mesh = ""
        faces = faces + 1

        for item in verts:
            mesh += f"v {item[0]} {item[1]} {item[2]}\n"

        for item in normals:
            mesh += f"vn {item[0]} {item[1]} {item[2]}\n"

        for item in faces:
            mesh += f"f {item[2]}//{item[2]} {item[1]}//{item[1]} {item[0]}//{item[0]}\n"

        return mesh
