#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from typing import Dict, List
import shutil
from itertools import product

from jinja2 import Environment, FileSystemLoader, select_autoescape
import pandas as pd
import numpy as np

PARAMETER_DEFAULTS = {
    "output_name" : "default",
    "init_conditions_file_name" : "hex_id_data_3c5a0327_3500002920_100X_20190419_c-Scene-06-P18-F09_CellNucSegCombined.csv",
    "subcell_radius" : 2.5,
    "dt" : 6,
    "subcell_adhesion" : 0.01,
    "subcell_repulsion" : 0.01,
    "subcell_adhesion_distance" : 4.0,
    "substrate_adhesion" : 0.01,
    "substrate_repulsion" : 0.01,
    "substrate_adhesion_distance" : 4.0,
    "relative_heterotypic_adhesion" : 0.2,
    "subcell_transition_multiplier" : 20.0,
    "subcell_death_rate" : 0.00001,
    "subcell_volume_growth_rate" : 0.1,
    "owner_basal_apoptosis_rate" : 0.0001,
    "owner_death_volume" : 300.0,
    "timesteps_per_owner_motility_switch" : 16.7,
    "owner_motility_bias" : 1.0,
    "normalized_subcell_speed" : 0.01,
}


PARAMETER_SCANS = {
    "Resolution" : [
        # "spatial_resolution" via csv
        "subcell_radius",  # to match csv
        "dt",
    ],
    # "Subcell_Adhesion" : [
    #     "subcell_adhesion",
    #     "subcell_repulsion",
    #     "subcell_adhesion_distance",
    # ],
    # "Substrate_Adhesion" : [
    #     "substrate_adhesion",
    #     "substrate_repulsion",
    #     "substrate_adhesion_distance",
    # ],
    # "Subcell_Dynamics" : [
    #     "subcell_transition_multiplier",
    #     "subcell_death_rate",
    #     "subcell_volume_growth_rate",
    # ],
    # "Apoptosis" : [
    #     "owner_basal_apoptosis_rate",
    #     "owner_death_volume",
    # ],
    # "Motility" : [
    #     "timesteps_per_owner_motility_switch",
    #     "owner_motility_bias",
    #     "normalized_subcell_speed",
    # ],
}


class TemplateRenderer:
    """
    
    """
    def __init__(
        self, 
        template_directory: str, 
        template_name: str, 
        params_path: str, 
        init_cond_directory: str,
        output_dir: str
    ):
        self.template_directory = template_directory
        self.template_name = template_name
        self.params_path = params_path
        self.init_cond_directory = init_cond_directory
        self.init_cond_files = os.listdir(init_cond_directory)
        self.output_dir = output_dir
        self._jinja_environment = None
        self.render()

    def jinja_environment(self) -> Environment:
        """
        Get a jinja2 Environment for rendering XML configs from a template.
        Create it if doesn't already exist.
        """
        if self._jinja_environment is None:
            self._jinja_environment = Environment(
                loader=FileSystemLoader(self.template_directory),
                autoescape=select_autoescape(),
            )
        return self._jinja_environment
    
    def _read_parameters_xlsx(self) -> List[Dict[str, str or float]]:
        """
        Read values to use for parameters from XLSX file
        """
        START_COLUMN = 2
        N_PARAMS = 17
        csv_df = pd.read_excel(
            self.params_path,
            sheet_name="PhysiCell params",
            usecols=range(START_COLUMN, START_COLUMN + N_PARAMS),
            skiprows=4,
            nrows=20,
            dtype=object,
        )
        master_params = {}
        for column in csv_df.columns:
            master_params[column] = csv_df[column].dropna().tolist()
        result = [PARAMETER_DEFAULTS]
        for scan_name, scan in PARAMETER_SCANS.items():
            scan_params = {param_name : master_params[param_name] for param_name in master_params if param_name in scan}
            param_names = list(scan_params.keys())
            for param_set in product(*scan_params.values()):                    
                params = PARAMETER_DEFAULTS.copy()
                params["output_name"] = f"{scan_name}"
                for param_index, param_name in enumerate(param_names):
                    param_value = param_set[param_index]
                    params[param_name] = param_value
                    params["output_name"] += f"-{param_name}_{param_value}"
                result.append(params)
        return result
    
    def _init_cond_file_name(self, subcell_radius: float):
        """
        Get init conditions file name for the given subcell radius
        """
        spatial_resolution = 2. * subcell_radius
        if spatial_resolution >= 1:
            spatial_resolution = int(spatial_resolution)
        spatial_resolution = "_" + str(spatial_resolution)
        return [file_name for file_name in self.init_cond_files if spatial_resolution in file_name][0]
    
    def _render_template(self, param_set: Dict[str, str or float]):
        """
        Render an XML config file for the given parameter set
        and save it in the output directory
        """
        output_name = param_set["output_name"]
        # add derived parameters
        init_cond_file_name = self._init_cond_file_name(param_set["subcell_radius"])
        param_set["init_conditions_file_name"] = init_cond_file_name
        param_set["dt_diffusion"] = 0.01 * param_set["dt"]
        param_set["dt_mechanics"] = 0.1 * param_set["dt"]
        param_set["dt_phenotype"] = param_set["dt"]
        param_set["subcell_volume"] = 4 / 3 * np.pi * param_set["subcell_radius"] ** 3
        param_set["subcell_volume_growth_rate_cytoplasmic"] = 0.1 * param_set["subcell_volume_growth_rate"]
        param_set["owner_motility_switch_time"] = param_set["timesteps_per_owner_motility_switch"] * param_set["dt"]
        param_set["subcell_speed"] = param_set["normalized_subcell_speed"] * param_set["subcell_radius"]
        # remove extra parameters
        del param_set["output_name"]
        del param_set["dt"]
        del param_set["timesteps_per_owner_motility_switch"]
        del param_set["normalized_subcell_speed"]
        # render template
        template = self.jinja_environment().get_template(self.template_name)
        config = template.render(**param_set)
        config_dir = os.path.join(self.output_dir, output_name)
        config_path = os.path.join(config_dir, "PhysiCell_settings.xml")
        if os.path.exists(config_dir):
            shutil.rmtree(config_dir)
        os.mkdir(config_dir)
        with open(config_path, "w") as config_xml:
            config_xml.write(config)
        # copy corresponding initial conditions csv
        init_cond_src_path = os.path.join(self.init_cond_directory, init_cond_file_name)
        init_cond_dest_path = os.path.join(config_dir, init_cond_file_name)
        shutil.copy2(init_cond_src_path, init_cond_dest_path)
            
    def render(self):
        """
        Render XML configs for each combination 
        of the parameter values from the XLSX sheet.
        """
        param_sets = self._read_parameters_xlsx()
        param_set_names = []
        for param_set in param_sets:
            param_set_names.append(param_set["output_name"])
            self._render_template(param_set)
        param_set_names_df = pd.DataFrame(param_set_names)
        param_set_names_df.to_csv(os.path.join(self.output_dir, "manifest.csv"), index=False)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Renders template for PhysiCell XML input files"
    )
    parser.add_argument(
        "template_directory", help="the file path to the template directory"
    )
    parser.add_argument(
        "template_name", help="the name of the template file to use"
    )
    parser.add_argument(
        "params_path", help="the file path to the XLSX file containing parameter values to use"
    )
    parser.add_argument(
        "init_cond_directory", help="the file path to the directory containing CSV files of initial conditions"
    )
    args = parser.parse_args()
    if not os.path.exists("outputs/"):
        os.mkdir("outputs/")
    renderer = TemplateRenderer(
        template_directory=args.template_directory, 
        template_name=args.template_name, 
        params_path=args.params_path, 
        init_cond_directory=args.init_cond_directory,
        output_dir="outputs/",
    )