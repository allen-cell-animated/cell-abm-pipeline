from prefect import task

from .convert_physicell_to_simularium import convert_physicell_to_simularium
from .parse_mcds_file import parse_mcds_file
from .render_physicell_template import render_physicell_template

convert_physicell_to_simularium = task(convert_physicell_to_simularium)
parse_mcds_file = task(parse_mcds_file)
render_physicell_template = task(render_physicell_template)
