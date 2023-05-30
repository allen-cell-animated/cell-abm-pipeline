from typing import Any

import numpy as np
from jinja2 import StrictUndefined, Template

PARAMETER_DEFAULTS: dict[str, Any] = {
    "subcell_radius": 2.5,
    "dt": 6,
    "subcell_adhesion": 0.05,
    "subcell_repulsion": 0.1,
    "subcell_adhesion_distance": 4.0,
    "substrate_adhesion": 0.15,
    "substrate_repulsion": 0.1,
    "substrate_adhesion_distance": 4.0,
    "relative_heterotypic_adhesion": 0.05,
    "subcell_transition_multiplier": 10.0,
    "subcell_death_rate": 0.00001,
    "subcell_volume_growth_rate": 0.015,
    "owner_basal_apoptosis_rate": 0.00001,
    "owner_death_volume": 300.0,
    "timesteps_per_owner_motility_switch": 16.7,
    "owner_motility_bias": 1.0,
    "normalized_subcell_speed": 0.01,
}


def _render_template(template: str, condition: dict[str, Any]) -> str:
    """
    Render an XML config file template for the given parameter set.
    """

    # Get default parameter values and update with values from conditions dictionary.
    parameters = PARAMETER_DEFAULTS.copy()
    parameters.update(condition)

    # Add derived parameters.
    parameters["dt_diffusion"] = 0.01 * parameters["dt"]
    parameters["dt_mechanics"] = 0.1 * parameters["dt"]
    parameters["dt_phenotype"] = parameters["dt"]
    parameters["subcell_volume"] = 1.5 * 4 / 3 * np.pi * parameters["subcell_radius"] ** 3
    parameters["subcell_volume_growth_rate_cytoplasmic"] = (
        0.1 * parameters["subcell_volume_growth_rate"]
    )
    parameters["owner_motility_switch_time"] = (
        parameters["timesteps_per_owner_motility_switch"] * parameters["dt"]
    )
    parameters["subcell_speed"] = (
        parameters["normalized_subcell_speed"] * parameters["subcell_radius"]
    )

    # Render template.
    jinja_template = Template(template, undefined=StrictUndefined)
    return jinja_template.render(**parameters)


def render_physicell_template(template: str, conditions: list[dict], group: str) -> list[str]:
    rendered_templates = []

    for condition in conditions:
        condition["group"] = group
        rendered_templates.append(_render_template(template, condition))

    return rendered_templates
