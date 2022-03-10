import io
import json
import lzma
import os
import pickle
import tarfile
from datetime import date

import boto3
import pandas as pd

MAX_CONTENT_LENGTH = 2**31 - 1

CHUNKSIZE = 100000

DTYPES = {
    "SEED": "uint8",
    "ID": "uint32",
    "NEIGHBOR": "uint32",
    "REGION": "uint32",
    "TICK": "uint16",
    "PARENT": "uint32",
    "POPULATION": "uint8",
    "AGE": "uint16",
    "DIVISIONS": "uint16",
    "STATE": "category",
    "PHASE": "category",
    "NUM_VOXELS": "uint32",
    "CENTER_X": "uint16",
    "CENTER_Y": "uint16",
    "CENTER_Z": "uint16",
    "MIN_X": "uint16",
    "MIN_Y": "uint16",
    "MIN_Z": "uint16",
    "MAX_X": "uint16",
    "MAX_Y": "uint16",
    "MAX_Z": "uint16",
    "KEY": "category",
}


def load_buffer(working, key):
    if working[:5] == "s3://":
        return load_buffer_from_s3(working[5:], key)
    else:
        return load_buffer_from_fs(working, key)


def load_buffer_from_fs(path, key):
    full_path = f"{path}{key}"
    return io.BytesIO(open(full_path, "rb").read())


def load_buffer_from_s3(bucket, key):
    """
    Loads body from bucket for given key.
    """
    s3_client = boto3.client("s3")
    obj = s3_client.get_object(Bucket=bucket, Key=key)

    # Check if body needs to be loaded in chunks.
    content_length = obj["ContentLength"]

    if content_length > MAX_CONTENT_LENGTH:
        print("Loading chunks ...")
        body = bytearray()
        for chunk in obj["Body"].iter_chunks(chunk_size=MAX_CONTENT_LENGTH):
            body += chunk
        return io.BytesIO(body)
    else:
        return io.BytesIO(obj["Body"].read())


def load_tar(working, key):
    if working[:5] == "s3://":
        return load_tar_from_s3(working[5:], key)
    else:
        return load_tar_from_fs(working, key)


def load_tar_from_fs(path, key):
    full_path = f"{path}{key}"
    return tarfile.open(full_path, mode="r:xz")


def load_tar_from_s3(bucket, key):
    """
    Loads tar archive from bucket with given key.
    """
    buffer = load_buffer_from_s3(bucket, key)
    return tarfile.open(fileobj=buffer, mode="r:xz")


def load_tar_member(tar, member):
    """
    Loads member of a tar file to json.
    """
    file = tar.extractfile(member)
    member_contents = [line.decode("utf-8") for line in file.readlines()]
    return json.loads("".join(member_contents))


def load_dataframe(working, key):
    if working[:5] == "s3://":
        return load_dataframe_from_s3(working[5:], key)
    else:
        return load_dataframe_from_fs(working, key)


def load_dataframe_from_fs(path, key):
    full_path = f"{path}{key}"
    return load_dataframe_object(full_path)


def load_dataframe_from_s3(bucket, key):
    contents = load_xz_from_s3(bucket, key)
    return load_dataframe_object(io.BytesIO(contents))


def load_dataframe_object(obj, chunksize=CHUNKSIZE, dtypes=DTYPES):
    df = pd.DataFrame()

    for chunk in pd.read_csv(obj, chunksize=chunksize, dtype=dtypes):
        df = pd.concat([df, chunk])

    zero_columns = {key: "uint8" for key in df.columns[(df == 0).all()]}
    df = df.astype(zero_columns)

    return df


def load_xz_from_s3(bucket, key):
    """
    Loads XZ compressed file from bucket with given key.
    """
    buffer = load_buffer_from_s3(bucket, key)
    return lzma.decompress(buffer.getbuffer())


def save_buffer(working, key, body):
    if working[:5] == "s3://":
        return save_buffer_to_s3(working[5:], key, body)
    else:
        return save_buffer_to_fs(working, key, body)


def save_buffer_to_fs(path, key, body):
    full_path = f"{path}{key}"
    make_folders(full_path)
    with open(full_path, "wb") as file:
        file.write(body.getbuffer())


def save_buffer_to_s3(bucket, key, body):
    """
    Saves body to bucket with given key.
    """
    s3_client = boto3.client("s3")
    s3_client.put_object(Bucket=bucket, Key=key, Body=body.getvalue())


def save_df(working, key, df, index=True):
    if working[:5] == "s3://":
        return save_df_to_s3(working[5:], key, df, index)
    else:
        return save_df_to_fs(working, key, df, index)


def save_df_to_fs(path, key, df, index=True):
    full_path = f"{path}{key}"
    make_folders(full_path)
    df.to_csv(full_path, index=index)


def save_df_to_s3(bucket, key, df, index=True):
    """
    Saves dataframe to bucket with given key.
    """
    with io.StringIO() as buffer:
        df.to_csv(buffer, index=index)
        save_buffer_to_s3(bucket, key, buffer)


def save_pickle_to_fs(path, key, obj):
    full_path = f"{path}{key}"
    make_folders(full_path)
    pickle.dump(obj, open(full_path, "wb"))


def make_folders(path):
    folders = "/".join(path.split("/")[:-1])
    os.makedirs(folders, exist_ok=True)


def make_folder_key(name, group, subgroup, timestamp):
    timestamp = f"/{date.today().strftime('%Y-%m-%d')}" if timestamp else ""
    subgroup = f"/{group}.{subgroup}" if subgroup else ""
    return f"{name}{timestamp}/{group}{subgroup}/"


def make_file_key(name, extension, key, seed):
    key = f"_{key}" if key else key
    seed = f"_{seed}" if seed else seed
    extension = ".".join([ext for ext in extension if ext])
    return f"{name}{key}{seed}.{extension}"


def make_full_key(name, group, extension, subgroup="", key="", seed="", timestamp=True):
    folder_key = make_folder_key(name, group, subgroup, timestamp)
    file_key = make_file_key(name, extension, key, seed)
    return f"{folder_key}{file_key}"
