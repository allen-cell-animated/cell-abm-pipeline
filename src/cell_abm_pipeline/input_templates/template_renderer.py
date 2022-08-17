#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from jinja2 import Environment, FileSystemLoader, select_autoescape


class TemplateRenderer:
    """
    
    """

    def __init__(self, template_file_path: str):
        self.template_file_path = template_file_path
        self._jinja_environment = None

    def jinja_environment(self):
        if self._jinja_environment is None:
            self._jinja_environment = Environment(
                loader=FileSystemLoader(self.parameters["template_directory"]),
                autoescape=select_autoescape(),
            )
        return self._jinja_environment
    
    def _render_template(self, parameters):
        template = self.jinja_environment().get_template(self.template_file_path)
        
        timesteps_per_owner_motility_switch = 16.7
        normalized_subcell_speed = 0.02
        subcell_radius = 20.58327
        subcell_volume_growth_rate=0.02
        total_runs = 100
        dt = 6
        result = []
        for run in range(total_runs):
            result.append(template.render(
                dt_diffusion=0.01 * dt,
                dt_mechanics=0.1 * dt,
                dt_phenotype=dt,
                subcell_adhesion=0.4,
                subcell_repulsion=1.0,
                subcell_adhesion_distance=4.0,
                substrate_adhesion=1.0,
                substrate_repulsion=1.0,
                substrate_adhesion_distance=4.0,
                relative_heterotypic_adhesion=0.2,
                subcell_transition_multiplier=20.0,
                subcell_death_rate=0.0001,
                subcell_volume_growth_rate=subcell_volume_growth_rate,
                subcell_volume_growth_rate_cytoplasmic=0.1 * subcell_volume_growth_rate,
                subcell_volume_growth_rate_nuclear=0.1 * subcell_volume_growth_rate,
                owner_basal_apoptosis_rate=0.0001,
                owner_death_volume=1000000.0,
                owner_motility_switch_time=timesteps_per_owner_motility_switch * dt,
                owner_motility_bias=1.0,
                subcell_speed=normalized_subcell_speed * subcell_radius,
            ))
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Renders template for PhysiCell XML input files"
    )
    parser.add_argument(
        "template_path", help="the file path to the template"
    )
    args = parser.parse_args()
    renderer = TemplateRenderer(args.template_path)