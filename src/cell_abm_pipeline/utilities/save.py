import io
import os
import re
import json
import pickle
import warnings
import tempfile

import boto3
import matplotlib.pyplot as plt
import imageio
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter


def save_buffer(working, key, body):
    if working[:5] == "s3://":
        return _save_buffer_to_s3(working[5:], key, body)
    else:
        return _save_buffer_to_fs(working, key, body)


def _save_buffer_to_fs(path, key, body):
    full_path = f"{path}{key}"
    make_folders(full_path)
    with open(full_path, "wb") as file:
        file.write(body.getbuffer())


def _save_buffer_to_s3(bucket, key, body):
    """
    Saves body to bucket with given key.
    """
    s3_client = boto3.client("s3")
    s3_client.put_object(Bucket=bucket, Key=key, Body=body.getvalue())


def save_dataframe(working, key, df, index=True):
    if working[:5] == "s3://":
        return _save_dataframe_to_s3(working[5:], key, df, index)
    else:
        return _save_dataframe_to_fs(working, key, df, index)


def _save_dataframe_to_fs(path, key, df, index=True):
    full_path = f"{path}{key}"
    make_folders(full_path)
    df.to_csv(full_path, index=index)


def _save_dataframe_to_s3(bucket, key, df, index=True):
    """
    Saves dataframe to bucket with given key.
    """
    with io.StringIO() as buffer:
        df.to_csv(buffer, index=index)
        _save_buffer_to_s3(bucket, key, buffer)


def save_pickle(working, key, obj):
    if working[:5] == "s3://":
        return _save_pickle_to_s3(working[5:], key, obj)
    else:
        return _save_pickle_to_fs(working, key, obj)


def _save_pickle_to_s3(bucket, key, obj):
    # TODO: implement save_pickle_to_s3
    warnings.warn("save_pickle_to_s3 not implemented, object not saved")


def _save_pickle_to_fs(path, key, obj):
    full_path = f"{path}{key}"
    make_folders(full_path)
    pickle.dump(obj, open(full_path, "wb"))


def save_image(working, key, obj):
    if working[:5] == "s3://":
        return _save_image_to_s3(working[5:], key, obj)
    else:
        return _save_image_to_fs(working, key, obj)


def _save_image_to_s3(bucket, key, obj):
    with tempfile.NamedTemporaryFile() as temp_file:
        OmeTiffWriter.save(obj, temp_file.name)
        full_key = f"s3://{bucket}/{key}"
        _save_buffer_to_s3(bucket, key, io.BytesIO(temp_file.read()))


def _save_image_to_fs(path, key, img):
    full_path = f"{path}{key}"
    make_folders(full_path)
    OmeTiffWriter.save(img, full_path)


def save_json(working, key, obj):
    if working[:5] == "s3://":
        return _save_json_to_s3(working[5:], key, obj)
    else:
        return _save_json_to_fs(working, key, obj)


def _save_json_to_s3(bucket, key, obj):
    # TODO: implement save_json_to_s3
    warnings.warn("save_json_to_s3 not implemented, json not saved")


def _save_json_to_fs(path, key, contents):
    full_path = f"{path}{key}"
    make_folders(full_path)
    with open(full_path, "w") as f:
        f.write(format_json(json.dumps(contents, indent=2, separators=(",", ":"))))


def save_plot(working, key):
    if working[:5] == "s3://":
        return _save_plot_to_s3(working[5:], key)
    else:
        return _save_plot_to_fs(working, key)


def _save_plot_to_fs(path, key):
    full_path = f"{path}{key}"
    make_folders(full_path)
    plt.savefig(full_path)


def _save_plot_to_s3(bucket, key):
    with io.BytesIO() as buffer:
        plt.savefig(buffer)
        _save_buffer_to_s3(bucket, key, buffer)


def save_gif(working, key, frames):
    if working[:5] == "s3://":
        return _save_gif_to_s3(working[5:], key, frames)
    else:
        return _save_gif_to_fs(working, key, frames)


def _save_gif_to_fs(path, key, frames):
    full_path = f"{path}{key}"
    make_folders(full_path)

    with imageio.get_writer(full_path, mode="I") as writer:
        for frame in frames:
            image = imageio.imread(f"{path}{frame}")
            writer.append_data(image)


def _save_gif_to_s3(bucket, key, frames):
    # TODO: implement save_gif_to_s3
    warnings.warn("save_gif_to_s3 not implemented, object not saved")


def make_folders(path):
    folders = "/".join(path.split("/")[:-1])
    os.makedirs(folders, exist_ok=True)


def format_json(contents):
    contents = contents.replace(":", ": ")
    for arr in re.findall('\[\n\s+[A-z0-9$",\-\.\n\s]*\]', contents):
        contents = contents.replace(arr, re.sub(r",\n\s+", r",", arr))
    contents = re.sub(r'\[\n\s+([A-Za-z0-9,"$\.\-]+)\n\s+\]', r"[\1]", contents)
    contents = contents.replace("],[", "],\n          [")
    contents = re.sub("([0-9]{1}),", r"\1, ", contents)
    return contents
