import io
import tempfile
import xml.etree.ElementTree as ET

import vtk
import trimesh
import numpy as np
import pandas as pd
from aicsshparam import shtools

from project_aics.cell_shape.__config__ import COEFF_ORDER, PCA_COMPONENTS
from project_aics.cell_shape.calculate_coefficients import CalculateCoefficients
from project_aics.utilities.load import load_pickle_from_df
from project_aics.utilities.save import save_buffer
from project_aics.utilities.keys import make_folder_key, make_file_key


class ExtractShapes:
    def __init__(self, context):
        self.context = context
        self.folders = {
            "input": make_folder_key(context.name, "analysis", "SHPCA", True),
            "output": make_folder_key(context.name, "plots", "SHPCA", True),
        }
        self.files = {
            "input": lambda r: make_file_key(context.name, ["SHPCA", r, "pkl"], "%s", ""),
            "output": make_file_key(context.name, ["SHPCA", "svg"], "%s", ""),
        }

    def run(self, delta=1, scale=1, box=(100, 100), region=None):
        for key in self.context.keys:
            self.extract_shapes(key, region, delta, scale, box)

    def extract_shapes(self, key, region, delta, scale, box):
        file_key = self.folders["input"] + self.files["input"](region) % key
        loaded = load_pickle_from_df(self.context.working, file_key)
        pca = loaded["pca"]
        data = loaded["data"]

        shape_svgs = self.extract_shape_svgs(pca, data, delta)
        svg = self.compile_shape_svg(shape_svgs, scale, box)

        output_key = self.folders["output"] + self.files["output"] % key
        save_buffer(self.context.working, output_key, svg)

    @staticmethod
    def extract_shape_svgs(pca, data, delta=1.0, components=PCA_COMPONENTS, order=COEFF_ORDER):
        # Transform data.
        coeffs = CalculateCoefficients.get_coeff_names()
        transformed_data = pca.transform(data[coeffs].values)

        # Calculate means and standard deviations of the transformed data.
        means = transformed_data.mean(axis=0)
        stds = transformed_data.std(axis=0, ddof=1)

        # Select points in transformed space.
        points = np.arange(-2, 2.5, delta)

        shape_svgs = []

        for i in range(components):
            point_vector = [0] * components
            component_shape_svgs = []

            for point in points:
                point_vector[i] = point
                vector = means + np.multiply(stds, point_vector)
                mesh = ExtractShapes.construct_mesh_from_points(pca, vector, coeffs, order)
                mesh = ExtractShapes.convert_vtk_to_trimesh(mesh)
                slices = ExtractShapes.get_mesh_slices(mesh)
                component_shape_svgs.append(slices)

            shape_svgs.append(component_shape_svgs)

        return shape_svgs

    @staticmethod
    def compile_shape_svg(svgs, scale=1, box=(100, 100)):
        width, height = box

        # Initialize output.
        root = ET.fromstring("<svg></svg>")
        group_top = ET.SubElement(root, "g", {"id": "view_top", "transform": "translate(0,0)"})
        group_side1 = ET.SubElement(root, "g", {"id": "view_side_1", "transform": "translate(0,0)"})
        group_side2 = ET.SubElement(root, "g", {"id": "view_side_2", "transform": "translate(0,0)"})

        # Add svg elements to output.
        for i, component in enumerate(svgs):
            for j, slices in enumerate(component):
                ExtractShapes.append_svg_element(
                    slices["side1"], group_side1, i, j, width, height, scale, rotate=90
                )
                ExtractShapes.append_svg_element(
                    slices["side2"], group_side2, i, j, width, height, scale, rotate=90
                )
                ExtractShapes.append_svg_element(
                    slices["top"], group_top, i, j, width, height, scale, rotate=0
                )

        ExtractShapes.clear_svg_namespaces(root)

        for element in root.findall(".//*"):
            ExtractShapes.clear_svg_namespaces(element)

        # Set SVG size and namespace.
        root.set("xmlns", "http://www.w3.org/2000/svg")
        root.set("height", str(height * len(svgs)))
        root.set("width", str(width * len(svgs[0])))

        return io.BytesIO(ET.tostring(root))

    @staticmethod
    def construct_mesh_from_points(pca, points, features, order=COEFF_ORDER):
        """Constructs mesh given PCA transformation points."""
        coeffs = pd.Series(pca.inverse_transform(points), index=features)
        coeffs_map = np.zeros((2, order + 1, order + 1), dtype=np.float32)

        for l in range(order + 1):
            for m in range(order + 1):
                coeffs_map[0, l, m] = coeffs[f"shcoeffs_L{l}M{m}C"]
                coeffs_map[1, l, m] = coeffs[f"shcoeffs_L{l}M{m}S"]

        mesh, _ = shtools.get_reconstruction_from_coeffs(coeffs_map)
        return mesh

    @staticmethod
    def convert_vtk_to_trimesh(mesh):
        with tempfile.NamedTemporaryFile() as temp:
            writer = vtk.vtkPLYWriter()
            writer.SetInputData(mesh)
            writer.SetFileTypeToASCII()
            writer.SetFileName(f"{temp.name}.ply")
            _ = writer.Write()
            mesh = trimesh.load(f"{temp.name}.ply")

        return mesh

    @staticmethod
    def get_mesh_slices(mesh):
        return {
            "side1": ExtractShapes.get_mesh_slice(mesh, [0, 1, 0]),
            "side2": ExtractShapes.get_mesh_slice(mesh, [1, 0, 0]),
            "top": ExtractShapes.get_mesh_slice(mesh, [0, 0, 1]),
        }

    @staticmethod
    def get_mesh_slice(mesh, normal):
        """Get svg slice of mesh along plane for given normal."""
        mesh_slice = mesh.section_multiplane(mesh.centroid, normal, [0])
        svg = trimesh.path.exchange.svg_io.export_svg(mesh_slice[0])
        return svg

    @staticmethod
    def append_svg_element(svg, root, row, col, width, height, scale=1.0, rotate=0, color="#555"):
        """Append svg element to root."""
        element = ET.fromstring(svg)
        path = element.findall(".//")[0]
        group = ET.SubElement(root, "g", {"transform": f"translate({col*width},{row*height})"})

        cx = width / 2
        cy = height / 2

        path.set("fill", "none")
        path.set("stroke", color)
        path.set("stroke-width", str(width / 80 / scale))
        path.set("transform", f"rotate({rotate},{cx},{cy}) translate({cx},{cy}) scale({scale})")
        group.insert(0, path)

    @staticmethod
    def clear_svg_namespaces(svg):
        _, has_namespace, postfix = svg.tag.partition("}")
        if has_namespace:
            svg.tag = postfix
