import io
import json
import boto3
import tarfile

MAX_CONTENT_LENGTH = 2**31 - 1


def load_from_s3(bucket, key):
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


def load_tar(location, key):
    if location[:5] == "s3://":
        return load_tar_from_s3(location[5:], key)
    else:
        return load_tar_from_path(location, key)


def load_tar_from_path(path, key):
    full_path = f"{path}{key}"
    return tarfile.open(full_path, mode="r:xz")


def load_tar_from_s3(bucket, key):
    """
    Loads tar archive from bucket with given key.
    """
    buffer = load_from_s3(bucket, key)
    return tarfile.open(fileobj=buffer, mode="r:xz")


def load_tar_member(tar, member):
    """
    Loads member of a tar file to json.
    """
    file = tar.extractfile(member)
    member_contents = [line.decode("utf-8") for line in file.readlines()]
    return json.loads("".join(member_contents))


def save_to_s3(bucket, key, body):
    """
    Saves body to bucket with given key.
    """
    s3_client = boto3.client("s3")
    s3_client.put_object(Bucket=bucket, Key=key, Body=body.getvalue())


def save_df(df, location, key, index=True):
    if location[:5] == "s3://":
        return save_df_to_s3(df, location[5:], key, index)
    else:
        return save_df_to_path(df, location, key, index)


def save_df_to_path(df, path, key, index=True):
    full_path = f"{path}{key}"
    df.to_csv(full_path, index=index)


def save_df_to_s3(df, bucket, key, index=True):
    """
    Saves dataframe to bucket with given key.
    """
    with io.StringIO() as buffer:
        df.to_csv(buffer, index=index)
        save_to_s3(bucket, key, buffer)
