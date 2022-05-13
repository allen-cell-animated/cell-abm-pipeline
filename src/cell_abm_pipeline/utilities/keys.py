from datetime import date


def make_folder_key(name, group, subgroup, timestamp):
    timestamp = f"/{date.today().strftime('%Y-%m-%d')}" if timestamp else ""
    subgroup = f"/{group}.{subgroup}" if subgroup else ""
    group = f"/{group}" if group else ""
    return f"{name}{timestamp}{group}{subgroup}/"


def make_file_key(name, extension, key, seed):
    key = f"_{key}" if key else key
    seed = f"_{seed}" if seed else seed
    extension = ".".join([ext for ext in extension if ext])
    extension = f".{extension}" if extension else extension
    return f"{name}{key}{seed}{extension}"


def make_full_key(folder_keys, file_keys, key_type, substitutes=None, arguments=None):
    folder_key = folder_keys[key_type]
    file_key = file_keys[key_type]

    if not isinstance(folder_key, str):
        folder_key = folder_key(arguments)

    if not isinstance(file_key, str):
        file_key = file_key(arguments)

    if substitutes is not None:
        file_key = file_key % substitutes

    full_key = folder_key + file_key
    return full_key.replace("__", "_").replace("_.", ".")
