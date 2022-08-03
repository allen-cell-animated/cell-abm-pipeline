The `initial_conditions` package contains modules for sampling from images and converting into input formats for various model frameworks.

```
Usage: initial-conditions [OPTIONS] COMMAND [ARGS]...

Options:
  -n, --name TEXT     Name of data set
  -k, --keys TEXT     Key(s) for data set
  -w, --working TEXT  Working location (local path or S3 bucket)  [default: .]
  --help              Show this message and exit.

Commands:
  convert-arcade   Convert samples into ARCADE input formats.
  create-voronoi   Create Voronoi tessellation from starting image.
  download-images  Download images from Quilt package.
  process-samples  Process samples with selected processing steps.
  sample-images    Sample cell ids and coordinates from images.
```

All modules require a `Context` object that defines the working context.
With the CLI, context must be set before any subcommand is called:

```bash
$ initial-conditions <context options> <subcommand>
```

## Download images from Quilt package

The `DownloadImages` module can be called via CLI using:

```
Usage: initial-conditions download-images [OPTIONS]

  Download images from Quilt package.

Options:
  -n, --num-images INTEGER  Number of images to download.  [default: 0]
  --help                    Show this message and exit.
```

## Sample cell ids and coordinates from images

The `SampleImages` module can be called via CLI using:

```
Usage: initial-conditions sample-images [OPTIONS]

  Sample cell ids and coordinates from images.

Options:
  -g, --grid [rect|hex]     Type of sampling grid.  [default: rect]
  -r, --resolution FLOAT    Distance between samples (um).  [default: 1.0]
  -c, --channels INTEGER    Image channel indices.  [default: 0]
  --contact / --no-contact  True if contact sheet of images is saved, False
                            otherwise.  [default: True]
  --help                    Show this message and exit.
```

## Process samples with selected processing steps

The `ProcessSamples` module can be called via CLI using:

```
Usage: initial-conditions process-samples [OPTIONS]

  Process samples with selected processing steps.

Options:
  -g, --grid [rect|hex]         Type of sampling grid.  [default: rect]
  --scale FLOAT                 Coordinate scaling factor.
  --select INTEGER              Specific cell ids to select.
  --edges / --no-edges          True if cells touching edges are removed,
                                False otherwise.  [default: False]
  --connected / --no-connected  True if unconnected voxels are removed, False
                                otherwise.  [default: False]
  --contact / --no-contact      True if contact sheet of images is saved,
                                False otherwise.  [default: True]
  --help                        Show this message and exit.
```

## Create Voronoi tessellation from given starting image

The `CreateVoronoi` module can be called via CLI using:

```
Usage: initial-conditions create-voronoi [OPTIONS]

  Create Voronoi tessellation from starting image.

Options:
  -i, --iterations INTEGER  Number of boundary estimation steps.  [default: 2]
  -c, --channels INTEGER    Image channel indices.  [default: 0]
  -h, --height INTEGER      Target height for tesselation.  [default: 10]
  --help                    Show this message and exit.
```

## Convert samples into ARCADE input formats

The `ConvertARCADE` module can be called via CLI using:

```
Usage: initial-conditions convert-arcade [OPTIONS]

  Convert samples into ARCADE input formats.

Options:
  -m, --margins INTEGER...  Margin size in x, y, and z directions.  [default:
                            0, 0, 0]
  --region TEXT             Region key to include in conversion.
  --reference TEXT          Path to reference data for conversion.
  --help                    Show this message and exit.
```
