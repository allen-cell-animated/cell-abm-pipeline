import io
import os
import json
import lzma
import pickle
import tarfile
import warnings
from glob import glob

import boto3
import pandas as pd
from aicsimageio import AICSImage
from aicsimageio.readers.tiff_reader import TiffReader

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


def load_keys(working, pattern):
    if working[:5] == "s3://":
        return _load_keys_from_s3(working[5:], pattern)
    else:
        return _load_keys_from_fs(working, pattern)


def _load_keys_from_fs(path, pattern):
    file_list = glob(f"{path}{pattern}")
    return [file.replace(path, "") for file in file_list]


def _load_keys_from_s3(bucket, pattern):
    prefix = pattern.split("*")[0]
    bucket_obj = boto3.resource("s3").Bucket(bucket)
    objs = list(bucket_obj.objects.filter(Prefix=prefix))
    return [obj.key for obj in objs]


def load_buffer(working, key):
    if working[:5] == "s3://":
        return _load_buffer_from_s3(working[5:], key)
    else:
        return _load_buffer_from_fs(working, key)


def _load_buffer_from_fs(path, key):
    full_path = f"{path}{key}"
    return io.BytesIO(open(full_path, "rb").read())


def _load_buffer_from_s3(bucket, key):
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
        return _load_tar_from_s3(working[5:], key)
    else:
        return _load_tar_from_fs(working, key)


def _load_tar_from_fs(path, key):
    full_path = f"{path}{key}"
    return tarfile.open(full_path, mode="r:xz")


def _load_tar_from_s3(bucket, key):
    """
    Loads tar archive from bucket with given key.
    """
    buffer = _load_buffer_from_s3(bucket, key)
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
        return _load_dataframe_from_s3(working[5:], key)
    else:
        return _load_dataframe_from_fs(working, key)


def _load_dataframe_from_fs(path, key):
    full_path = f"{path}{key}"
    return load_dataframe_object(full_path)


def _load_dataframe_from_s3(bucket, key):
    if key.split(".")[-1] == "xz":
        contents = _load_xz_from_s3(bucket, key)
    else:
        contents = _load_csv_from_s3(bucket, key)
    return load_dataframe_object(io.BytesIO(contents))


def load_dataframe_object(obj, chunksize=CHUNKSIZE, dtypes=DTYPES):
    df = pd.DataFrame()

    for chunk in pd.read_csv(obj, chunksize=chunksize, dtype=dtypes):
        df = pd.concat([df, chunk])

    zero_columns = {key: "uint8" for key in df.columns[(df == 0).all()]}
    df = df.astype(zero_columns)

    return df


def _load_csv_from_s3(bucket, key):
    """
    Loads XZ compressed file from bucket with given key.
    """
    buffer = _load_buffer_from_s3(bucket, key)
    return buffer.getbuffer()


def _load_xz_from_s3(bucket, key):
    """
    Loads XZ compressed file from bucket with given key.
    """
    buffer = _load_buffer_from_s3(bucket, key)
    return lzma.decompress(buffer.getbuffer())


def load_pickle(working, key):
    if working[:5] == "s3://":
        return _load_pickle_from_s3(working[5:], key)
    else:
        return _load_pickle_from_fs(working, key)


def _load_pickle_from_s3(bucket, key):
    # TODO: implement load_pickle_from_s3
    warnings.warn("load_pickle_from_s3 not implemented, object not loaded")
    return None


def _load_pickle_from_fs(path, key):
    full_path = f"{path}{key}"
    return pickle.load(open(full_path, "rb"))


def load_image(working, key):
    if working[:5] == "s3://":
        return _load_image_from_s3(working[5:], key)
    else:
        return _load_image_from_fs(working, key)


def _load_image_from_s3(bucket, key):
    full_key = f"s3://{bucket}/{key}"
    try:
        return AICSImage(full_key)
    except:
        return AICSImage(full_key.replace(".ome.", "."), reader=TiffReader, dim_order="ZYX")


def _load_image_from_fs(path, key):
    """
    Loads TIFF image as AICSImage object.
    """
    full_path = f"{path}{key}"

    if os.path.exists(full_path):
        return AICSImage(full_path, reader=TiffReader)
    else:
        return AICSImage(full_path.replace(".ome.", "."), reader=TiffReader, dim_order="ZYX")
