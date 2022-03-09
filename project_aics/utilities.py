import boto3
import io
import json
import tarfile


MAX_CONTENT_LENGTH = 2**31 - 1


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


def save_buffer(working, key, body):
    if working[:5] == "s3://":
        return save_buffer_to_s3(working[5:], key, body)
    else:
        return save_buffer_to_fs(working, key, body)


def save_buffer_to_fs(path, key, body):
    full_path = f"{path}{key}"
    with open(full_path, "wb") as f:
        f.write(body.getbuffer())


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
    df.to_csv(full_path, index=index)


def save_df_to_s3(bucket, key, df, index=True):
    """
    Saves dataframe to bucket with given key.
    """
    with io.StringIO() as buffer:
        df.to_csv(buffer, index=index)
        save_buffer_to_s3(bucket, key, buffer)