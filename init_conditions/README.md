# CLI for sampling segmented images to create inputs for models

## Installation

Run `conda env create -f cellabm.yml` to create environment.

## Usage

### Download images

This step will download segmentation data and save XY projections.

```bash
$ python cli.py download-images [OPTIONS]

  -n, --num-images INTEGER  number of FOV images to download
  -p, --output-path PATH    path to directory to save outputs (default =
                            current directory)
```

Images are downloaded from the Quilt package and registry specified in the `constants.py` file.
The step also saves an `fovs.txt` file (if one doesn't already exist) listing all the available images.
Update the `fovs.txt` with specific files to force specific downloads.
Images that have been downloaded will not be re-downloaded.

Downloaded images are saved to a folder `downloaded_images/<image-id>/` directory in the output path.
Each XY projection is saved as `downloaded_images/<image-id>/<image-id>_z<z-slice>.png`

### Sample images

```bash
$ python cli.py sample-images [OPTIONS]

  -p, --output-path PATH           path to directory to save outputs (default =
                                   current directory)
  -g, --grid-type [hex|cartesian]  sampling grid type (default = hex)
  -r, --resolution FLOAT           microns between samples (default = 1.0)
  --contact / --no-contact         True if contact sheet of images is saved,
                                   False otherwise (default = True)
```

Downloaded images are sampled to extract samples in the form (`id`, `x`, `y`, `z`).
Samples are saved `<grid-type>_samples/<image-id>/<grid-type>_samples_<image-id>.csv`.
Images that have already been sampled will not be re-sampled.

### Process Samples

```bash
$ python cli.py process-samples [OPTIONS] SAMPLES

  SAMPLES                          Path to samples file.

  --edges / --no-edges             True if cells touching edges are removed,
                                   False otherwise (default = True)
  --connected / --no-connected     True if unconnected voxels are removed,
                                   False otherwise (default = True)
  --scale / --no-scale             True if coordinates are scaled, False
                                   otherwise (default = True)
  -s, --scale-factor FLOAT         Scaling factor for coordinates (default =
                                   1.0).
  -g, --grid-type [hex|cartesian]  sampling grid type (default = hex)
  --contact / --no-contact         True if contact sheet of images is saved,
                                   False otherwise (default = True)
```

Selected processing steps can be applied to the selected samples file.
Processed samples are saved to the same location as the samples file with the extension `.PROCESSED.csv`.

Processing options include:

- removing cells that are touching the edge of the FOV
- removed unconnected voxels (`cartesian` grid only)
- rescale voxels to &#181;m (using `SCALE_MICRONS` and `SCALE_MICRONS_Z` specified in the `constants.py` file) as well as the given scale factor

## Convert samples

```bash
$ python cli.py convert-samples [OPTIONS] SAMPLES

  SAMPLES               Path to samples file.

  -f, --format TEXT     Format to convert to (options = arcade)
  -b, --box INTEGER...  Size of bounding box in x, y, and z directions
                        (default = 100 x 100 x 10)
```

Convert samples (processed or not) into selected formats.
More than one format can be selected (e.g. `-f <FORMAT> -f <FORMAT>`).
Files are saved to the same directory as the input samples file.

Conversion options and output files:

- ARCADE
    - `.xml` template setup file for running simulation
    - `.CELLS.json` lists each cell and cell features
    - `.LOCATIONS.json` lists all voxels for each cell
