import os
import numpy as np
from pathlib import Path
from pyMCDS import pyMCDS
import json


model_name = "new_model"
path_to_xml_files = "../../PhysiCell/output/"
json_output_path = "output/"

# ---------------------------------------------------------------------------------------------------------


def load_simulation_data():
    """ load simulation data from PhysiCell MultiCellDS XML files """

    sorted_files = sorted(Path(path_to_xml_files).glob("output*.xml"))
    data = []
    for file in sorted_files:
        data.append(pyMCDS(file.name, False, path_to_xml_files))

    return np.array(data)


ids = {}
last_id = 0


def get_agent_type(cell_type, cell_phase):
    """ get a unique agent type id for a specific cell type and phase combination """
    global ids, last_id

    if cell_type not in ids:
        ids[cell_type] = {}

    if cell_phase not in ids[cell_type]:
        print(
            "ID {} : cell type = {}, cell phase = {}".format(
                last_id, cell_type, cell_phase
            )
        )
        ids[cell_type][cell_phase] = last_id
        last_id += 1

    return ids[cell_type][cell_phase]


def get_visualization_data_one_frame(index, sim_data_one_frame):
    """ get data from one time step in Simularium format """
    discrete_cells = sim_data_one_frame.get_cell_df()

    data = {}
    data["data"] = []
    data["frameNumber"] = index
    data["time"] = index

    for i in range(len(discrete_cells["position_x"])):

        data["data"].append(1000.0)  # viz type = sphere
        data["data"].append(
            float(
                get_agent_type(
                    int(discrete_cells["cell_type"][i]),  # agent type
                    int(discrete_cells["current_phase"][i]),
                )
            )
        )
        data["data"].append(round(discrete_cells["position_x"][i], 1))  # X position
        data["data"].append(round(discrete_cells["position_y"][i], 1))  # Y position
        data["data"].append(round(discrete_cells["position_z"][i], 1))  # Z position
        data["data"].append(
            round(discrete_cells["orientation_x"][i], 1)
        )  # X orientation
        data["data"].append(
            round(discrete_cells["orientation_x"][i], 1)
        )  # Y orientation
        data["data"].append(
            round(discrete_cells["orientation_x"][i], 1)
        )  # Z orientation
        data["data"].append(
            round(np.cbrt(3.0 / 4.0 * discrete_cells["total_volume"][i] / np.pi), 1)
        )  # scale
        data["data"].append(0.0)  # ?

    return data


def get_visualization_data(sim_data):
    """ get data in Simularium format """

    data = {}
    data["msgType"] = 1
    data["bundleStart"] = 0
    data["bundleSize"] = len(sim_data)
    data["bundleData"] = []

    for i in range(len(sim_data)):
        data["bundleData"].append(get_visualization_data_one_frame(i, sim_data[i]))

    return data


def write_json_data(viz_data):
    """ write all data to a json file """
    if not os.path.exists(json_output_path):
        os.makedirs(json_output_path)
    with open("{}{}.json".format(json_output_path, model_name), "w+") as outfile:
        json.dump(viz_data, outfile)


def convert_xml_to_json():
    """ convert a set of MultiCellDS XML files to a Simularium JSON file """
    write_json_data(get_visualization_data(load_simulation_data()))
    print("wrote Simularium JSON with {} agent types".format(last_id))
