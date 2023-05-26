from prefect import task

from .render_physicell_template import render_physicell_template

render_physicell_template = task(render_physicell_template)
