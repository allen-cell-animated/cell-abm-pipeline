from prefect import task


@task
def convert_data_units(data, ds, dt, region=None):
    data["time"] = dt * data["TICK"]
    data["volume"] = ds * ds * ds * data["NUM_VOXELS"]
    data["height"] = ds * (data["MAX_Z"] - data["MIN_Z"] + 1)

    if region:
        data[f"volume.{region}"] = ds * ds * ds * data[f"NUM_VOXELS.{region}"]
        data[f"height.{region}"] = ds * (data[f"MAX_Z.{region}"] - data[f"MIN_Z.{region}"])
