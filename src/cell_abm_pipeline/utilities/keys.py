from datetime import date
from glob import glob

import boto3


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


def make_full_key(folder_keys, file_keys, key_type, substitutes, arguments=None):
    folder_key = folder_keys[key_type]
    file_key = file_keys[key_type]

    if isinstance(file_key, str):
        full_key = folder_key + file_key % substitutes
    else:
        full_key = folder_key + file_key(arguments) % substitutes

    return full_key.replace("__", "_")


def get_keys(working, pattern):
    if working[:5] == "s3://":
        return _get_keys_from_s3(working[5:], pattern)
    else:
        return _get_keys_from_fs(working, pattern)


def _get_keys_from_fs(path, pattern):
    file_list = glob(f"{path}{pattern}")
    return [file.replace(path, "") for file in file_list]


def _get_keys_from_s3(bucket, pattern):
    prefix = pattern.split("*")[0]
    bucket_obj = boto3.resource("s3").Bucket(bucket)
    objs = list(bucket_obj.objects.filter(Prefix=prefix))
    return [obj.key for obj in objs]
