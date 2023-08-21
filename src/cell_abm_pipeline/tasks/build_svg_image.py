import xml.etree.ElementTree as ET

from prefect import task

PATH_ATTRIBUTES: list[str] = ["fill", "stroke", "stroke-width", "stroke-dasharray"]


@task
def build_svg_image(
    elements: list[dict], width: int, height: int, rotate: float, scale: float
) -> str:
    root = ET.fromstring("<svg></svg>")
    root.set("xmlns", "http://www.w3.org/2000/svg")
    root.set("width", str(width))
    root.set("height", str(height))

    for element in elements:
        path = ET.fromstring("<path></path>")

        cx = width / 2
        cy = height / 2

        path.set("d", "M" + "L".join([f"{x},{y}" for x, y in element["points"]]))

        for attribute in PATH_ATTRIBUTES:
            path.set(attribute, str(element[attribute]) if attribute in element else "none")

        path.set("transform", f"rotate({rotate},{cx},{cy}) translate({cx},{cy}) scale({scale})")
        root.insert(0, path)

    return ET.tostring(root, encoding="unicode")
