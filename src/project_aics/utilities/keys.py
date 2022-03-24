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


def make_full_key(name, group, extension, subgroup="", key="", seed="", timestamp=True):
    folder_key = make_folder_key(name, group, subgroup, timestamp)
    file_key = make_file_key(name, extension, key, seed)
    return f"{folder_key}{file_key}"
