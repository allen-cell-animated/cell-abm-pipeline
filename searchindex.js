Search.setIndex({"docnames": ["basic_metrics", "basic_metrics.plot_projection", "basic_metrics.plot_spatial", "basic_metrics.plot_temporal", "cell_shape", "cell_shape.calculate_coefficients", "cell_shape.compress_coefficients", "cell_shape.extract_shapes", "cell_shape.merge_coefficients", "cell_shape.perform_pca", "cell_shape.plot_pca", "colony_dynamics", "colony_dynamics.analyze_clusters", "colony_dynamics.analyze_measures", "colony_dynamics.compress_neighbors", "colony_dynamics.create_networks", "colony_dynamics.find_neighbors", "colony_dynamics.merge_neighbors", "colony_dynamics.plot_clusters", "colony_dynamics.plot_measures", "colony_dynamics.plot_neighbors", "convert_format", "convert_format.arcade_to_image", "convert_format.arcade_to_mesh", "convert_format.arcade_to_simularium", "index", "initial_conditions", "initial_conditions.convert_arcade", "initial_conditions.create_voronoi", "initial_conditions.download_images", "initial_conditions.generate_coordinates", "initial_conditions.process_samples", "initial_conditions.sample_images", "modules", "resource_usage", "resource_usage.calculate_storage", "resource_usage.extract_clock", "resource_usage.plot_resources", "utilities", "utilities.keys", "utilities.load", "utilities.plot", "utilities.save"], "filenames": ["basic_metrics.rst", "basic_metrics.plot_projection.rst", "basic_metrics.plot_spatial.rst", "basic_metrics.plot_temporal.rst", "cell_shape.rst", "cell_shape.calculate_coefficients.rst", "cell_shape.compress_coefficients.rst", "cell_shape.extract_shapes.rst", "cell_shape.merge_coefficients.rst", "cell_shape.perform_pca.rst", "cell_shape.plot_pca.rst", "colony_dynamics.rst", "colony_dynamics.analyze_clusters.rst", "colony_dynamics.analyze_measures.rst", "colony_dynamics.compress_neighbors.rst", "colony_dynamics.create_networks.rst", "colony_dynamics.find_neighbors.rst", "colony_dynamics.merge_neighbors.rst", "colony_dynamics.plot_clusters.rst", "colony_dynamics.plot_measures.rst", "colony_dynamics.plot_neighbors.rst", "convert_format.rst", "convert_format.arcade_to_image.rst", "convert_format.arcade_to_mesh.rst", "convert_format.arcade_to_simularium.rst", "index.rst", "initial_conditions.rst", "initial_conditions.convert_arcade.rst", "initial_conditions.create_voronoi.rst", "initial_conditions.download_images.rst", "initial_conditions.generate_coordinates.rst", "initial_conditions.process_samples.rst", "initial_conditions.sample_images.rst", "modules.rst", "resource_usage.rst", "resource_usage.calculate_storage.rst", "resource_usage.extract_clock.rst", "resource_usage.plot_resources.rst", "utilities.rst", "utilities.keys.rst", "utilities.load.rst", "utilities.plot.rst", "utilities.save.rst"], "titles": ["basic_metrics package", "basic_metrics.plot_projection module", "basic_metrics.plot_spatial module", "basic_metrics.plot_temporal module", "cell_shape package", "cell_shape.calculate_coefficients module", "cell_shape.compress_coefficients module", "cell_shape.extract_shapes module", "cell_shape.merge_coefficients module", "cell_shape.perform_pca module", "cell_shape.plot_pca module", "colony_dynamics package", "colony_dynamics.analyze_clusters module", "colony_dynamics.analyze_measures module", "colony_dynamics.compress_neighbors module", "colony_dynamics.create_networks module", "colony_dynamics.find_neighbors module", "colony_dynamics.merge_neighbors module", "colony_dynamics.plot_clusters module", "colony_dynamics.plot_measures module", "colony_dynamics.plot_neighbors module", "convert_format package", "convert_format.arcade_to_image module", "convert_format.arcade_to_mesh module", "convert_format.arcade_to_simularium module", "Cell agent-based model pipeline", "initial_conditions package", "initial_conditions.convert_arcade module", "initial_conditions.create_voronoi module", "initial_conditions.download_images module", "initial_conditions.generate_coordinates module", "initial_conditions.process_samples module", "initial_conditions.sample_images module", "cell_abm_pipeline", "resource_usage package", "resource_usage.calculate_storage module", "resource_usage.extract_clock module", "resource_usage.plot_resources module", "utilities package", "utilities.keys module", "utilities.load module", "utilities.plot module", "utilities.save module"], "terms": {"plot_project": [0, 33], "modul": [0, 4, 11, 21, 25, 26, 33, 34, 38], "plot_spati": [0, 33], "plot_tempor": [0, 33], "class": [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 27, 28, 29, 30, 31, 32, 35, 36, 37], "plotproject": 1, "context": [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 27, 28, 29, 30, 31, 32, 35, 36, 37], "sourc": [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 27, 28, 29, 30, 31, 32, 35, 36, 37, 39, 40, 41, 42], "base": [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 27, 28, 29, 30, 31, 32, 35, 36, 37], "object": [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 35, 36, 37], "run": [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 35, 36, 37], "region": [1, 2, 3, 5, 6, 7, 8, 9, 10, 27, 28, 31, 40], "none": [1, 2, 3, 5, 6, 7, 8, 9, 10, 16, 27, 28, 29, 30, 31, 32, 39], "frame": [1, 5, 23, 24, 42], "0": [1, 2, 5, 7, 23, 27, 29, 30, 31, 32, 41], "1": [1, 2, 3, 5, 7, 23, 24, 27, 28, 29, 30, 31, 32], "box": [1, 7, 16, 22, 24, 28, 30], "100": [1, 7, 22, 24, 30], "10": [1, 22, 24, 28, 30], "d": [1, 2, 3, 24, 30], "dt": [1, 3, 24], "scale": [1, 5, 7, 31, 32], "timestamp": [1, 39], "true": [1, 30, 31, 32, 42], "scalebar": 1, "data": [1, 2, 3, 7, 10, 18, 19, 20, 24, 27, 30, 31, 32, 41], "seed": [1, 2, 5, 6, 8, 14, 16, 17, 39, 40], "annot": 1, "static": [1, 2, 3, 5, 7, 9, 12, 13, 15, 16, 22, 23, 24, 27, 28, 29, 30, 31, 32, 35, 36], "add_frame_timestamp": 1, "ax": [1, 30, 31, 32, 41], "length": [1, 24], "width": [1, 24], "includ": [1, 25, 27, 31], "add_frame_scalebar": 1, "plotspati": 2, "convert_data_unit": [2, 3], "plot_volume_distribut": [2, 3], "vmin": 2, "vmax": 2, "4000": 2, "plot_height_distribut": [2, 3], "20": 2, "plot_phase_distribut": 2, "plot_population_distribut": 2, "plottempor": 3, "refer": [3, 10, 27], "plot_total_count": 3, "plot_cell_phas": 3, "plot_phase_dur": 3, "get_phase_dur": 3, "calcul": [3, 5, 12, 13, 25, 27, 28, 30, 32], "phase": [3, 20, 40], "durat": 3, "given": [3, 7, 9, 27, 28, 29, 30, 31, 32], "datafram": [3, 27, 29, 30, 31, 32], "plot_individual_volum": 3, "plot_individual_height": 3, "plot_average_volum": 3, "plot_average_height": 3, "calculate_coeffici": [4, 33], "compress_coeffici": [4, 33], "extract_shap": [4, 33], "merge_coeffici": [4, 33], "perform_pca": [4, 33], "plot_pca": [4, 33], "calculatecoeffici": 5, "kei": [5, 7, 13, 16, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 38, 40, 41, 42], "spheric": [5, 25], "harmon": [5, 25], "coeffici": [5, 6, 8, 25], "all": [5, 16, 25, 29, 30, 41], "cell": [5, 15, 16, 27, 30, 31, 32], "get_location_voxel": [5, 27], "locat": [5, 16, 22, 23, 27, 28, 29, 30, 31, 32], "scale_voxel_arrai": 5, "arrai": [5, 16, 22, 23, 28, 31], "make_voxels_arrai": [5, 16], "voxel": [5, 23, 27, 31], "convert": [5, 15, 25, 27, 31], "list": [5, 12, 15, 16, 27, 28, 29, 30, 31, 32], "get_coeff_nam": 5, "prefix": [5, 7, 24], "str": [5, 27, 28, 29, 30, 31, 32], "suffix": [5, 7], "order": [5, 7, 9, 31], "int": [5, 27, 28, 29, 30, 31, 32], "16": [5, 7], "get": [5, 7, 12, 13, 16, 27, 29, 30, 31, 32], "name": [5, 24, 27, 28, 29, 30, 31, 32, 35, 39], "paramet": [5, 27, 28, 29, 30, 31, 32], "prepend": 5, "each": [5, 16, 27, 31, 32], "append": [5, 7, 28, 31], "parametr": 5, "compresscoeffici": 6, "compress": [6, 14], "individu": [6, 8, 14, 17], "file": [6, 8, 14, 17, 25, 27, 28, 29, 30, 31, 32, 40], "singl": [6, 8, 12, 14, 17], "archiv": [6, 14], "extractshap": 7, "delta": 7, "extract_shape_svg": 7, "pca": [7, 9], "compon": [7, 9, 10], "8": [7, 9], "vector": 7, "coeff": 7, "compile_shape_svg": 7, "svg": 7, "construct_mesh_from_point": 7, "point": [7, 31], "featur": [7, 9, 10], "construct": 7, "mesh": 7, "transform": [7, 27, 28], "convert_vtk_to_trimesh": 7, "get_mesh_slic": 7, "normal": [7, 23], "slice": [7, 28, 30, 31, 32], "along": [7, 27], "plane": 7, "append_svg_el": 7, "root": 7, "row": 7, "col": 7, "rotat": 7, "color": [7, 30, 31, 32], "555": 7, "element": [7, 30], "clear_svg_namespac": 7, "mergecoeffici": 8, "merg": [8, 17], "performpca": 9, "fit_feature_pca": 9, "perform": [9, 28], "apply_data_transform": 9, "plotpca": 10, "plot_pca_variance_explain": 10, "plot_pca_transform_featur": 10, "plot_pca_transform_compar": 10, "analyze_clust": [11, 33], "analyze_measur": [11, 33], "compress_neighbor": [11, 33], "create_network": [11, 33], "find_neighbor": [11, 33], "merge_neighbor": [11, 33], "plot_clust": [11, 33], "plot_measur": [11, 33], "plot_neighbor": [11, 33], "analyzeclust": 12, "calculate_cluster_metr": 12, "cluster": 12, "make_centroid_dict": 12, "df": [12, 42], "creat": [12, 27, 28], "dictionari": [12, 27, 28, 29, 30, 31, 32], "id": [12, 15, 16, 27, 30, 31, 32, 40], "x": [12, 27, 28, 30, 31, 32], "y": [12, 27, 28, 30, 31, 32], "z": [12, 27, 28, 30, 31, 32], "centroid": 12, "get_cluster_s": 12, "group": [12, 35, 39], "size": [12, 27, 30, 31, 32, 41], "get_cluster_centroid": 12, "centroid_dict": 12, "averag": [12, 27], "across": 12, "get_inter_cluster_dist": 12, "distanc": [12, 13, 28, 30, 31, 32], "between": [12, 27, 28, 30, 31, 32], "get_intra_cluster_dist": 12, "within": [12, 30], "analyzemeasur": 13, "calculate_graph_measur": 13, "graph": [13, 25], "get_network_degre": 13, "degre": 13, "network": [13, 15], "get_network_dist": 13, "measur": 13, "get_network_centr": 13, "central": 13, "compressneighbor": 14, "neighbor": [14, 15, 16, 17, 25, 40], "createnetwork": 15, "convert_to_network": 15, "findneighbor": 16, "find": [16, 31], "connect": [16, 29, 31], "flatten_neighbors_list": 16, "attribut": 16, "center": [16, 27, 30, 32], "depth_map": 16, "get_array_neighbor": 16, "uniqu": [16, 27], "calculate_depth_map": 16, "get_bounding_box": 16, "bound": [16, 27, 28, 30, 32, 41], "around": [16, 28], "binari": [16, 28], "get_cropped_arrai": 16, "label": [16, 28, 41], "crop_origin": 16, "fals": [16, 30, 31, 32, 41], "find_edge_id": [16, 31], "mergeneighbor": 17, "plotclust": 18, "plot_cluster_count": 18, "plot_cluster_size_mean": 18, "plot_cluster_size_std": 18, "plot_cluster_fract": 18, "plot_inter_cluster_distances_mean": 18, "plot_inter_cluster_distances_std": 18, "plot_intra_cluster_distances_mean": 18, "plot_intra_cluster_distances_std": 18, "plotmeasur": 19, "plot_degree_distribut": 19, "plot_average_degree_mean": 19, "plot_average_degree_std": 19, "plot_network_dist": 19, "plot_network_centr": 19, "plotneighbor": 20, "view": 20, "arcade_to_imag": [21, 33], "arcade_to_mesh": [21, 33], "arcade_to_simularium": [21, 33], "arcadetoimag": 22, "convert_image_fram": 22, "index": [22, 24, 25, 28, 30, 31, 32, 42], "split_array_chunk": 22, "arcadetomesh": 23, "convert_frame_mesh": 23, "make_mesh_arrai": 23, "make_array_mesh": 23, "make_mesh_object": 23, "vert": 23, "face": [23, 30, 32], "arcadetosimularium": 24, "get_meta_data": 24, "height": [24, 27, 28], "get_dimension_data": 24, "cells_tar": 24, "get_agent_data": 24, "get_display_data": 24, "convert_cells_tar": 24, "tar": [24, 40], "convert_locations_tar": 24, "thi": [25, 29, 31], "repositori": 25, "contain": [25, 29, 32], "us": [25, 27, 28, 31], "work": [25, 27, 28, 29, 30, 31, 32, 35, 40, 42], "can": [25, 29], "call": 25, "via": 25, "cli": 25, "import": 25, "python": 25, "project": 25, "top": 25, "level": 25, "subpackag": 25, "basic_metr": [25, 33], "plot": [25, 30, 31, 32, 33, 38], "basic": 25, "simul": 25, "metric": 25, "resource_usag": [25, 33], "quantifi": 25, "resourc": 25, "usag": 25, "wall": 25, "clock": 25, "storag": 25, "set": [25, 30, 32], "cell_shap": [25, 33], "extract": [25, 29, 32], "shape": [25, 31], "dimension": 25, "reduct": 25, "colony_dynam": [25, 33], "coloni": 25, "dynam": 25, "from": [25, 27, 28, 29, 31, 32], "initial_condit": [25, 33], "sampl": [25, 27, 30, 31, 32], "imag": [25, 27, 28, 29, 30, 31, 32], "input": [25, 27, 28, 29, 30, 31, 32], "format": [25, 27], "variou": 25, "framework": 25, "see": 25, "relev": 25, "readm": 25, "detail": 25, "The": [25, 27, 29, 31, 32], "util": [25, 33], "function": 25, "save": [25, 28, 29, 30, 31, 32, 33, 38], "load": [25, 27, 28, 29, 31, 32, 33, 38], "local": [25, 29], "cloud": 25, "manag": 25, "follow": [25, 31], "tool": 25, "poetri": 25, "packag": [25, 29, 33], "depend": 25, "tox": 25, "autom": 25, "test": 25, "black": 25, "code": 25, "pylint": 25, "lint": 25, "well": 25, "github": 25, "action": 25, "automat": 25, "build": 25, "gener": [25, 30, 31], "document": 25, "clone": 25, "repo": 25, "initi": [25, 27], "init": [25, 27, 30], "activ": 25, "environ": 25, "shell": 25, "makefil": 25, "three": [25, 27, 30, 32], "make": [25, 31], "clean": 25, "type": [25, 27, 30, 31, 32], "check": [25, 31], "your": 25, "you": 25, "also": 25, "just": 25, "doc": 25, "search": 25, "page": 25, "convert_arcad": [26, 33], "create_voronoi": [26, 33], "download_imag": [26, 33], "generate_coordin": [26, 33], "process_sampl": [26, 33], "sample_imag": [26, 33], "convertarcad": 27, "task": [27, 28, 29, 30, 31, 32], "arcad": 27, "structur": [27, 28, 29, 30, 31, 32], "_": [27, 28, 29, 30, 31, 32], "xml": 27, "json": [27, 40], "2": [27, 28, 29, 30, 31, 32], "n": [27, 28, 29, 30, 31, 32], "process": [27, 31], "csv": [27, 29, 30, 31, 32], "For": [27, 31], "convers": 27, "output": [27, 28, 29, 30, 31, 32], "should": 27, "extens": [27, 39], "defin": [27, 28, 29, 30, 31, 32], "folder": [27, 28, 29, 30, 31, 32], "margin": 27, "tupl": [27, 28, 30, 31, 32], "option": [27, 28, 31, 32], "direct": [27, 28, 30, 31, 32], "path": [27, 29, 42], "panda": [27, 29, 30, 31, 32], "iter": [27, 28], "through": 27, "transform_sample_voxel": 27, "coordin": [27, 30, 31, 32], "return": [27, 28, 29, 30, 31, 32], "calculate_sample_bound": 27, "make_setup_fil": 27, "term": 27, "volum": 27, "surfac": 27, "adhes": 27, "substrat": 27, "persist": 27, "byte": 27, "setup": 27, "number": [27, 28, 29, 31], "pott": 27, "hamiltonian": 27, "default": [27, 31, 32], "potts_term": 27, "content": [27, 36, 42], "filter_valid_sampl": 27, "filter": [27, 29, 30, 31], "valid": [27, 30], "condit": 27, "must": 27, "have": 27, "least": 27, "one": 27, "assign": [27, 31], "specifi": [27, 29], "filter_cell_refer": 27, "cell_id": 27, "dict": [27, 31], "convert_to_cel": 27, "convert_to_loc": 27, "get_cell_critical_volum": 27, "float": [27, 30, 31, 32], "avg": 27, "std": 27, "critical_avg": 27, "critical_std": 27, "estim": [27, 28], "critic": 27, "actual": 27, "distribut": 27, "volume_avg": 27, "standard": 27, "deviat": 27, "volume_std": 27, "critical_volume_avg": 27, "critical_volume_std": 27, "get_cell_critical_height": 27, "height_avg": 27, "height_std": 27, "critical_height_avg": 27, "critical_height_std": 27, "get_cell_st": 27, "critical_volum": 27, "threshold_fract": 27, "state": [27, 40], "threshold": [27, 31], "fraction": 27, "monoton": 27, "differ": 27, "v": 27, "x1": 27, "x2": 27, "xn": 27, "correspond": [27, 30, 32], "f1": 27, "f2": 27, "fn": 27, "i": [27, 28, 29, 30, 31, 32, 41], "xi": 27, "f": 27, "fi": 27, "ar": [27, 28, 29, 30, 31, 32], "current": [27, 28], "cell_state_threshold_fract": 27, "get_location_cent": 27, "createvoronoi": 28, "voronoi": 28, "tessel": 28, "start": 28, "om": [28, 29, 32], "tiff": [28, 29, 32], "channel": [28, 31, 32], "_voronoi": 28, "same": 28, "directori": [28, 30, 31, 32], "valu": [28, 31], "boundari": 28, "step": [28, 31], "indic": [28, 32], "target": [28, 30, 31], "dilat": 28, "clamp": 28, "select": [28, 30, 31, 32], "create_boundary_mask": 28, "numpi": [28, 31], "ndarrai": [28, 31], "fill": 28, "mask": 28, "get_mask_bound": 28, "target_rang": 28, "axi": [28, 31], "rang": 28, "If": [28, 29, 31], "minimum": [28, 31], "maximum": [28, 30, 31], "indici": 28, "where": 28, "exist": [28, 29, 31], "non": 28, "zero": 28, "entri": [28, 31], "wider": 28, "than": 28, "lower": 28, "upper": 28, "get_array_slic": 28, "calculate_voronoi_arrai": 28, "downloadimag": 29, "download": 29, "quilt": 29, "manifest": 29, "fov": 29, "segment": 29, "statu": 29, "avail": 29, "mark": 29, "re": 29, "edit": 29, "specif": [29, 31], "delet": 29, "regener": 29, "num_imag": 29, "quilt_packag": 29, "quilt_registri": 29, "request": 29, "up": 29, "get_fov_fil": 29, "pkg": 29, "quilt3": 29, "status": 29, "column": 29, "descript": 29, "fov_seg_path": 29, "fov_path": 29, "raw": [29, 31, 32], "directli": 29, "doe": 29, "s3": 29, "bucket": 29, "generatecoordin": 30, "png": [30, 31, 32], "place": [30, 32], "grid": [30, 31, 32], "rect": [30, 31, 32], "contact": [30, 31, 32], "bool": [30, 31, 32], "hex": [30, 31, 32], "um": [30, 32], "sheet": [30, 31, 32], "otherwis": [30, 31, 32], "insid": 30, "out": 30, "outsid": 30, "radiu": 30, "plot_contact_sheet": [30, 31, 32], "plot_contact_sheet_ax": [30, 31, 32], "matplotlib": [30, 31, 32], "instanc": [30, 31, 32], "get_box_coordin": 30, "possibl": 30, "make_rect_coordin": 30, "xy_incr": 30, "z_increment": 30, "increment": 30, "make_hex_coordin": 30, "offset": [30, 32, 41], "form": [30, 32], "cubic": [30, 32], "fcc": [30, 32], "pack": [30, 32], "transform_cell_coordin": 30, "appli": [30, 31], "processsampl": 31, "after": 31, "sure": 31, "pass": 31, "overwrit": 31, "exclud": 31, "edg": 31, "factor": 31, "touch": 31, "remov": 31, "unconnect": 31, "rescal": 31, "grai": 31, "scale_coordin": 31, "scale_factor": 31, "scale_xi": [31, 32], "108333": [31, 32], "scale_z": [31, 32], "29": [31, 32], "addition": 31, "resolut": [31, 32], "scale_microns_xi": [31, 32], "scale_microns_z": [31, 32], "include_cel": 31, "exclude_cel": 31, "without": 31, "remove_edge_cel": 31, "edge_threshold": 31, "remove_unconnected_region": 31, "connected_threshold": 31, "remove_unconnected_by_connect": 31, "simpl": 31, "remove_unconnected_by_dist": 31, "get_step_s": 31, "multipl": 31, "most": 31, "common": 31, "get_sample_minimum": 31, "get_sample_maximum": 31, "pad": 31, "limit": 31, "convert_to_integer_arrai": 31, "integ": 31, "convert_to_datafram": 31, "get_minimum_dist": 31, "3": 31, "sampleimag": 32, "result": 32, "choic": 32, "taken": 32, "separ": 32, "get_image_bound": 32, "aicsimageio": 32, "aicsimag": 32, "get_sample_indic": 32, "get_hex_sample_indic": 32, "get_rect_sample_indic": 32, "get_image_sampl": 32, "sample_indic": 32, "submodul": 33, "convert_format": 33, "calculate_storag": [33, 34], "extract_clock": [33, 34], "plot_resourc": [33, 34], "calculatestorag": 35, "get_file_summari": 35, "file_kei": [35, 39], "extractclock": 36, "parse_log_fil": 36, "plotresourc": 37, "plot_wall_clock": 37, "plot_object_storag": 37, "make_folder_kei": 39, "subgroup": 39, "make_file_kei": 39, "make_full_kei": 39, "folder_kei": 39, "key_typ": 39, "substitut": 39, "argument": 39, "load_kei": 40, "pattern": 40, "load_buff": 40, "load_tar": 40, "load_tar_memb": 40, "member": 40, "load_datafram": 40, "load_dataframe_object": 40, "obj": [40, 42], "chunksiz": 40, "100000": 40, "dtype": 40, "ag": 40, "uint16": 40, "center_x": 40, "center_i": 40, "center_z": 40, "divis": 40, "uint32": 40, "categori": 40, "max_x": 40, "max_i": 40, "max_z": 40, "min_x": 40, "min_i": 40, "min_z": 40, "num_voxel": 40, "parent": 40, "popul": 40, "uint8": 40, "tick": 40, "load_pickl": 40, "load_imag": 40, "make_plot": 41, "func": 41, "4": 41, "xlabel": 41, "ylabel": 41, "sharex": 41, "sharei": 41, "legend": 41, "make_subplot": 41, "n_row": 41, "n_col": 41, "separate_kei": 41, "select_ax": 41, "j": 41, "make_legend": 41, "interv": 41, "5": 41, "colormap": 41, "magma_r": 41, "save_buff": 42, "bodi": 42, "save_datafram": 42, "save_pickl": 42, "save_imag": 42, "save_json": 42, "save_plot": 42, "save_gif": 42, "make_fold": 42, "format_json": 42}, "objects": {"": [[0, 0, 0, "-", "basic_metrics"], [4, 0, 0, "-", "cell_shape"], [11, 0, 0, "-", "colony_dynamics"], [21, 0, 0, "-", "convert_format"], [26, 0, 0, "-", "initial_conditions"], [34, 0, 0, "-", "resource_usage"], [38, 0, 0, "-", "utilities"]], "basic_metrics": [[1, 0, 0, "-", "plot_projection"], [2, 0, 0, "-", "plot_spatial"], [3, 0, 0, "-", "plot_temporal"]], "basic_metrics.plot_projection": [[1, 1, 1, "", "PlotProjection"]], "basic_metrics.plot_projection.PlotProjection": [[1, 2, 1, "", "add_frame_scalebar"], [1, 2, 1, "", "add_frame_timestamp"], [1, 2, 1, "", "plot_projection"], [1, 2, 1, "", "run"]], "basic_metrics.plot_spatial": [[2, 1, 1, "", "PlotSpatial"]], "basic_metrics.plot_spatial.PlotSpatial": [[2, 2, 1, "", "convert_data_units"], [2, 2, 1, "", "plot_height_distribution"], [2, 2, 1, "", "plot_phase_distribution"], [2, 2, 1, "", "plot_population_distribution"], [2, 2, 1, "", "plot_volume_distribution"], [2, 2, 1, "", "run"]], "basic_metrics.plot_temporal": [[3, 1, 1, "", "PlotTemporal"]], "basic_metrics.plot_temporal.PlotTemporal": [[3, 2, 1, "", "convert_data_units"], [3, 2, 1, "", "get_phase_durations"], [3, 2, 1, "", "plot_average_height"], [3, 2, 1, "", "plot_average_volume"], [3, 2, 1, "", "plot_cell_phases"], [3, 2, 1, "", "plot_height_distribution"], [3, 2, 1, "", "plot_individual_height"], [3, 2, 1, "", "plot_individual_volume"], [3, 2, 1, "", "plot_phase_durations"], [3, 2, 1, "", "plot_total_counts"], [3, 2, 1, "", "plot_volume_distribution"], [3, 2, 1, "", "run"]], "cell_shape": [[5, 0, 0, "-", "calculate_coefficients"], [6, 0, 0, "-", "compress_coefficients"], [7, 0, 0, "-", "extract_shapes"], [8, 0, 0, "-", "merge_coefficients"], [9, 0, 0, "-", "perform_pca"], [10, 0, 0, "-", "plot_pca"]], "cell_shape.calculate_coefficients": [[5, 1, 1, "", "CalculateCoefficients"]], "cell_shape.calculate_coefficients.CalculateCoefficients": [[5, 2, 1, "", "calculate_coefficients"], [5, 2, 1, "", "get_coeff_names"], [5, 2, 1, "", "get_location_voxels"], [5, 2, 1, "", "make_voxels_array"], [5, 2, 1, "", "run"], [5, 2, 1, "", "scale_voxel_array"]], "cell_shape.compress_coefficients": [[6, 1, 1, "", "CompressCoefficients"]], "cell_shape.compress_coefficients.CompressCoefficients": [[6, 2, 1, "", "compress_coefficients"], [6, 2, 1, "", "run"]], "cell_shape.extract_shapes": [[7, 1, 1, "", "ExtractShapes"]], "cell_shape.extract_shapes.ExtractShapes": [[7, 2, 1, "", "append_svg_element"], [7, 2, 1, "", "clear_svg_namespaces"], [7, 2, 1, "", "compile_shape_svg"], [7, 2, 1, "", "construct_mesh_from_points"], [7, 2, 1, "", "convert_vtk_to_trimesh"], [7, 2, 1, "", "extract_shape_svg"], [7, 2, 1, "", "extract_shape_svgs"], [7, 2, 1, "", "extract_shapes"], [7, 2, 1, "", "get_mesh_slice"], [7, 2, 1, "", "get_mesh_slices"], [7, 2, 1, "", "run"]], "cell_shape.merge_coefficients": [[8, 1, 1, "", "MergeCoefficients"]], "cell_shape.merge_coefficients.MergeCoefficients": [[8, 2, 1, "", "merge_coefficients"], [8, 2, 1, "", "run"]], "cell_shape.perform_pca": [[9, 1, 1, "", "PerformPCA"]], "cell_shape.perform_pca.PerformPCA": [[9, 2, 1, "", "apply_data_transform"], [9, 2, 1, "", "fit_feature_pca"], [9, 2, 1, "", "perform_pca"], [9, 2, 1, "", "run"]], "cell_shape.plot_pca": [[10, 1, 1, "", "PlotPCA"]], "cell_shape.plot_pca.PlotPCA": [[10, 2, 1, "", "plot_pca_transform_compare"], [10, 2, 1, "", "plot_pca_transform_features"], [10, 2, 1, "", "plot_pca_variance_explained"], [10, 2, 1, "", "run"]], "colony_dynamics": [[12, 0, 0, "-", "analyze_clusters"], [13, 0, 0, "-", "analyze_measures"], [14, 0, 0, "-", "compress_neighbors"], [15, 0, 0, "-", "create_networks"], [16, 0, 0, "-", "find_neighbors"], [17, 0, 0, "-", "merge_neighbors"], [18, 0, 0, "-", "plot_clusters"], [19, 0, 0, "-", "plot_measures"], [20, 0, 0, "-", "plot_neighbors"]], "colony_dynamics.analyze_clusters": [[12, 1, 1, "", "AnalyzeClusters"]], "colony_dynamics.analyze_clusters.AnalyzeClusters": [[12, 2, 1, "", "analyze_clusters"], [12, 2, 1, "", "calculate_cluster_metrics"], [12, 2, 1, "", "get_cluster_centroid"], [12, 2, 1, "", "get_cluster_sizes"], [12, 2, 1, "", "get_inter_cluster_distances"], [12, 2, 1, "", "get_intra_cluster_distances"], [12, 2, 1, "", "make_centroid_dict"], [12, 2, 1, "", "run"]], "colony_dynamics.analyze_measures": [[13, 1, 1, "", "AnalyzeMeasures"]], "colony_dynamics.analyze_measures.AnalyzeMeasures": [[13, 2, 1, "", "analyze_measures"], [13, 2, 1, "", "calculate_graph_measures"], [13, 2, 1, "", "get_network_centralities"], [13, 2, 1, "", "get_network_degrees"], [13, 2, 1, "", "get_network_distances"], [13, 2, 1, "", "run"]], "colony_dynamics.compress_neighbors": [[14, 1, 1, "", "CompressNeighbors"]], "colony_dynamics.compress_neighbors.CompressNeighbors": [[14, 2, 1, "", "compress_neighbors"], [14, 2, 1, "", "run"]], "colony_dynamics.create_networks": [[15, 1, 1, "", "CreateNetworks"]], "colony_dynamics.create_networks.CreateNetworks": [[15, 2, 1, "", "convert_to_network"], [15, 2, 1, "", "create_networks"], [15, 2, 1, "", "run"]], "colony_dynamics.find_neighbors": [[16, 1, 1, "", "FindNeighbors"]], "colony_dynamics.find_neighbors.FindNeighbors": [[16, 2, 1, "", "calculate_depth_map"], [16, 2, 1, "", "find_edge_ids"], [16, 2, 1, "", "find_neighbors"], [16, 2, 1, "", "flatten_neighbors_list"], [16, 2, 1, "", "get_array_neighbors"], [16, 2, 1, "", "get_bounding_box"], [16, 2, 1, "", "get_cropped_array"], [16, 2, 1, "", "make_voxels_array"], [16, 2, 1, "", "run"]], "colony_dynamics.merge_neighbors": [[17, 1, 1, "", "MergeNeighbors"]], "colony_dynamics.merge_neighbors.MergeNeighbors": [[17, 2, 1, "", "merge_neighbors"], [17, 2, 1, "", "run"]], "colony_dynamics.plot_clusters": [[18, 1, 1, "", "PlotClusters"]], "colony_dynamics.plot_clusters.PlotClusters": [[18, 2, 1, "", "plot_cluster_counts"], [18, 2, 1, "", "plot_cluster_fraction"], [18, 2, 1, "", "plot_cluster_size_mean"], [18, 2, 1, "", "plot_cluster_size_std"], [18, 2, 1, "", "plot_inter_cluster_distances_mean"], [18, 2, 1, "", "plot_inter_cluster_distances_std"], [18, 2, 1, "", "plot_intra_cluster_distances_mean"], [18, 2, 1, "", "plot_intra_cluster_distances_std"], [18, 2, 1, "", "run"]], "colony_dynamics.plot_measures": [[19, 1, 1, "", "PlotMeasures"]], "colony_dynamics.plot_measures.PlotMeasures": [[19, 2, 1, "", "plot_average_degree_mean"], [19, 2, 1, "", "plot_average_degree_std"], [19, 2, 1, "", "plot_degree_distribution"], [19, 2, 1, "", "plot_network_centrality"], [19, 2, 1, "", "plot_network_distances"], [19, 2, 1, "", "run"]], "colony_dynamics.plot_neighbors": [[20, 1, 1, "", "PlotNeighbors"]], "colony_dynamics.plot_neighbors.PlotNeighbors": [[20, 2, 1, "", "plot_neighbors"], [20, 2, 1, "", "run"]], "convert_format": [[22, 0, 0, "-", "arcade_to_image"], [23, 0, 0, "-", "arcade_to_mesh"], [24, 0, 0, "-", "arcade_to_simularium"]], "convert_format.arcade_to_image": [[22, 1, 1, "", "ArcadeToImage"]], "convert_format.arcade_to_image.ArcadeToImage": [[22, 2, 1, "", "arcade_to_image"], [22, 2, 1, "", "convert_image_frame"], [22, 2, 1, "", "run"], [22, 2, 1, "", "split_array_chunks"]], "convert_format.arcade_to_mesh": [[23, 1, 1, "", "ArcadeToMesh"]], "convert_format.arcade_to_mesh.ArcadeToMesh": [[23, 2, 1, "", "arcade_to_mesh"], [23, 2, 1, "", "convert_frame_meshes"], [23, 2, 1, "", "make_array_mesh"], [23, 2, 1, "", "make_mesh_array"], [23, 2, 1, "", "make_mesh_object"], [23, 2, 1, "", "run"]], "convert_format.arcade_to_simularium": [[24, 1, 1, "", "ArcadeToSimularium"]], "convert_format.arcade_to_simularium.ArcadeToSimularium": [[24, 2, 1, "", "arcade_to_simularium"], [24, 2, 1, "", "convert_cells_tar"], [24, 2, 1, "", "convert_locations_tar"], [24, 2, 1, "", "get_agent_data"], [24, 2, 1, "", "get_dimension_data"], [24, 2, 1, "", "get_display_data"], [24, 2, 1, "", "get_meta_data"], [24, 2, 1, "", "run"]], "initial_conditions": [[27, 0, 0, "-", "convert_arcade"], [28, 0, 0, "-", "create_voronoi"], [29, 0, 0, "-", "download_images"], [30, 0, 0, "-", "generate_coordinates"], [31, 0, 0, "-", "process_samples"], [32, 0, 0, "-", "sample_images"]], "initial_conditions.convert_arcade": [[27, 1, 1, "", "ConvertARCADE"]], "initial_conditions.convert_arcade.ConvertARCADE": [[27, 2, 1, "", "calculate_sample_bounds"], [27, 3, 1, "", "context"], [27, 2, 1, "", "convert_arcade"], [27, 2, 1, "", "convert_to_cell"], [27, 2, 1, "", "convert_to_location"], [27, 3, 1, "", "files"], [27, 2, 1, "", "filter_cell_reference"], [27, 2, 1, "", "filter_valid_samples"], [27, 3, 1, "", "folders"], [27, 2, 1, "", "get_cell_critical_height"], [27, 2, 1, "", "get_cell_critical_volume"], [27, 2, 1, "", "get_cell_state"], [27, 2, 1, "", "get_location_center"], [27, 2, 1, "", "get_location_voxels"], [27, 2, 1, "", "make_setup_file"], [27, 2, 1, "", "run"], [27, 2, 1, "", "transform_sample_voxels"]], "initial_conditions.create_voronoi": [[28, 1, 1, "", "CreateVoronoi"]], "initial_conditions.create_voronoi.CreateVoronoi": [[28, 2, 1, "", "calculate_voronoi_array"], [28, 3, 1, "", "context"], [28, 2, 1, "", "create_boundary_mask"], [28, 2, 1, "", "create_voronoi"], [28, 3, 1, "", "files"], [28, 3, 1, "", "folders"], [28, 2, 1, "", "get_array_slices"], [28, 2, 1, "", "get_mask_bounds"], [28, 2, 1, "", "run"]], "initial_conditions.download_images": [[29, 1, 1, "", "DownloadImages"]], "initial_conditions.download_images.DownloadImages": [[29, 3, 1, "", "context"], [29, 2, 1, "", "download_images"], [29, 3, 1, "", "files"], [29, 3, 1, "", "folders"], [29, 2, 1, "", "get_fov_files"], [29, 2, 1, "", "run"]], "initial_conditions.generate_coordinates": [[30, 1, 1, "", "GenerateCoordinates"]], "initial_conditions.generate_coordinates.GenerateCoordinates": [[30, 3, 1, "", "context"], [30, 3, 1, "", "files"], [30, 3, 1, "", "folders"], [30, 2, 1, "", "generate_coordinates"], [30, 2, 1, "", "get_box_coordinates"], [30, 2, 1, "", "make_hex_coordinates"], [30, 2, 1, "", "make_rect_coordinates"], [30, 2, 1, "", "plot_contact_sheet"], [30, 2, 1, "", "plot_contact_sheet_axes"], [30, 2, 1, "", "run"], [30, 2, 1, "", "transform_cell_coordinates"]], "initial_conditions.process_samples": [[31, 1, 1, "", "ProcessSamples"]], "initial_conditions.process_samples.ProcessSamples": [[31, 3, 1, "", "context"], [31, 2, 1, "", "convert_to_dataframe"], [31, 2, 1, "", "convert_to_integer_array"], [31, 2, 1, "", "exclude_cells"], [31, 3, 1, "", "files"], [31, 2, 1, "", "find_edge_ids"], [31, 3, 1, "", "folders"], [31, 2, 1, "", "get_minimum_distance"], [31, 2, 1, "", "get_sample_maximums"], [31, 2, 1, "", "get_sample_minimums"], [31, 2, 1, "", "get_step_size"], [31, 2, 1, "", "get_step_sizes"], [31, 2, 1, "", "include_cells"], [31, 2, 1, "", "plot_contact_sheet"], [31, 2, 1, "", "plot_contact_sheet_axes"], [31, 2, 1, "", "process_samples"], [31, 2, 1, "", "remove_edge_cells"], [31, 2, 1, "", "remove_unconnected_by_connectivity"], [31, 2, 1, "", "remove_unconnected_by_distance"], [31, 2, 1, "", "remove_unconnected_regions"], [31, 2, 1, "", "run"], [31, 2, 1, "", "scale_coordinates"]], "initial_conditions.sample_images": [[32, 1, 1, "", "SampleImages"]], "initial_conditions.sample_images.SampleImages": [[32, 3, 1, "", "context"], [32, 3, 1, "", "files"], [32, 3, 1, "", "folders"], [32, 2, 1, "", "get_hex_sample_indices"], [32, 2, 1, "", "get_image_bounds"], [32, 2, 1, "", "get_image_samples"], [32, 2, 1, "", "get_rect_sample_indices"], [32, 2, 1, "", "get_sample_indices"], [32, 2, 1, "", "plot_contact_sheet"], [32, 2, 1, "", "plot_contact_sheet_axes"], [32, 2, 1, "", "run"], [32, 2, 1, "", "sample_images"]], "resource_usage": [[35, 0, 0, "-", "calculate_storage"], [36, 0, 0, "-", "extract_clock"], [37, 0, 0, "-", "plot_resources"]], "resource_usage.calculate_storage": [[35, 1, 1, "", "CalculateStorage"]], "resource_usage.calculate_storage.CalculateStorage": [[35, 2, 1, "", "calculate_storage"], [35, 2, 1, "", "get_file_summary"], [35, 2, 1, "", "run"]], "resource_usage.extract_clock": [[36, 1, 1, "", "ExtractClock"]], "resource_usage.extract_clock.ExtractClock": [[36, 2, 1, "", "extract_clock"], [36, 2, 1, "", "parse_log_file"], [36, 2, 1, "", "run"]], "resource_usage.plot_resources": [[37, 1, 1, "", "PlotResources"]], "resource_usage.plot_resources.PlotResources": [[37, 2, 1, "", "plot_object_storage"], [37, 2, 1, "", "plot_wall_clock"], [37, 2, 1, "", "run"]], "utilities": [[39, 0, 0, "-", "keys"], [40, 0, 0, "-", "load"], [41, 0, 0, "-", "plot"], [42, 0, 0, "-", "save"]], "utilities.keys": [[39, 4, 1, "", "make_file_key"], [39, 4, 1, "", "make_folder_key"], [39, 4, 1, "", "make_full_key"]], "utilities.load": [[40, 4, 1, "", "load_buffer"], [40, 4, 1, "", "load_dataframe"], [40, 4, 1, "", "load_dataframe_object"], [40, 4, 1, "", "load_image"], [40, 4, 1, "", "load_keys"], [40, 4, 1, "", "load_pickle"], [40, 4, 1, "", "load_tar"], [40, 4, 1, "", "load_tar_member"]], "utilities.plot": [[41, 4, 1, "", "make_legend"], [41, 4, 1, "", "make_plot"], [41, 4, 1, "", "make_subplots"], [41, 4, 1, "", "select_axes"], [41, 4, 1, "", "separate_keys"]], "utilities.save": [[42, 4, 1, "", "format_json"], [42, 4, 1, "", "make_folders"], [42, 4, 1, "", "save_buffer"], [42, 4, 1, "", "save_dataframe"], [42, 4, 1, "", "save_gif"], [42, 4, 1, "", "save_image"], [42, 4, 1, "", "save_json"], [42, 4, 1, "", "save_pickle"], [42, 4, 1, "", "save_plot"]]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:method", "3": "py:attribute", "4": "py:function"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "method", "Python method"], "3": ["py", "attribute", "Python attribute"], "4": ["py", "function", "Python function"]}, "titleterms": {"basic_metr": [0, 1, 2, 3], "packag": [0, 4, 11, 21, 26, 34, 38], "submodul": [0, 4, 11, 21, 26, 34, 38], "plot_project": 1, "modul": [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 27, 28, 29, 30, 31, 32, 35, 36, 37, 39, 40, 41, 42], "plot_spati": 2, "plot_tempor": 3, "cell_shap": [4, 5, 6, 7, 8, 9, 10], "calculate_coeffici": 5, "compress_coeffici": 6, "extract_shap": 7, "merge_coeffici": 8, "perform_pca": 9, "plot_pca": 10, "colony_dynam": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20], "analyze_clust": 12, "analyze_measur": 13, "compress_neighbor": 14, "create_network": 15, "find_neighbor": 16, "merge_neighbor": 17, "plot_clust": 18, "plot_measur": 19, "plot_neighbor": 20, "convert_format": [21, 22, 23, 24], "arcade_to_imag": 22, "arcade_to_mesh": 23, "arcade_to_simularium": 24, "cell": 25, "agent": 25, "base": 25, "model": 25, "pipelin": 25, "featur": 25, "instal": 25, "command": 25, "indic": 25, "tabl": 25, "initial_condit": [26, 27, 28, 29, 30, 31, 32], "convert_arcad": 27, "create_voronoi": 28, "download_imag": 29, "generate_coordin": 30, "process_sampl": 31, "sample_imag": 32, "cell_abm_pipelin": 33, "resource_usag": [34, 35, 36, 37], "calculate_storag": 35, "extract_clock": 36, "plot_resourc": 37, "util": [38, 39, 40, 41, 42], "kei": 39, "load": 40, "plot": 41, "save": 42}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.viewcode": 1, "sphinx": 56}})