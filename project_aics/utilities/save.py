import io
import os
import pickle

import boto3
import matplotlib.pyplot as plt
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter


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


def save_dataframe(working, key, df, index=True):
    if working[:5] == "s3://":
        return save_dataframe_to_s3(working[5:], key, df, index)
    else:
        return save_dataframe_to_fs(working, key, df, index)


def save_dataframe_to_fs(path, key, df, index=True):
    full_path = f"{path}{key}"
    make_folders(full_path)
    df.to_csv(full_path, index=index)


def save_dataframe_to_s3(bucket, key, df, index=True):
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


def save_image_to_fs(path, key, img):
    full_path = f"{path}{key}"
    make_folders(full_path)
    OmeTiffWriter.save(img, full_path)


def save_plot(working, key):
    if working[:5] == "s3://":
        return save_plot_to_s3(working[5:], key)
    else:
        return save_plot_to_fs(working, key)


def save_plot_to_fs(path, key):
    full_path = f"{path}{key}"
    make_folders(full_path)
    plt.savefig(full_path)


def save_plot_to_s3(bucket, key):
    with io.BytesIO() as buffer:
        plt.savefig(buffer)
        save_buffer_to_s3(bucket, key, buffer)


def make_folders(path):
    folders = "/".join(path.split("/")[:-1])
    os.makedirs(folders, exist_ok=True)
