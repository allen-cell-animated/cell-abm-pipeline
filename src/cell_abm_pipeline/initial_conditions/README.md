The `initial_conditions` package contains modules for sampling from images and converting into input formats for various model frameworks.

```bash
Usage: initial-conditions [OPTIONS] COMMAND [ARGS]...

Options:
  -n, --name TEXT     Name of data set
  -k, --keys TEXT     Key(s) for data set
  -w, --working TEXT  Working location (local path or S3 bucket)  [default: .]
  --help              Show this message and exit.

Commands:
  convert-arcade   Convert samples into ARCADE input formats.
  create-voronoi   Create Voronoi tesselation from given starting image.
  download-images  Download images from Quilt package.
  process-samples  Process samples with selected processing steps.
  sample-images    Sample ids and coordinates from images.
```

All modules require a `Context` object that defines the working context.
With the CLI, context must be set before any subcommand is called:

```bash
$ initial-conditions <context options> <subcommand>
```

## Download images from Quilt package

The `DownloadImages` module can be called via CLI using:

```bash
Usage: initial-conditions download-images [OPTIONS]

  Download images from Quilt package.

Options:
  -n, --num-images INTEGER  Number of images to download.  [default: 0]
  --help                    Show this message and exit.
```

## Sample id and coordinates from images

The `SampleImages` module can be called via CLI using:

```bash
Usage: initial-conditions sample-images [OPTIONS]

  Sample ids and coordinates from images.

Options:
  -g, --grid [rect|hex]     Type of sampling grid.  [default: rect]
  -r, --resolution FLOAT    Distance between samples (um).  [default: 1.0]
  -c, --channels INTEGER    Image channel indices.  [default: 0]
  --contact / --no-contact  True if contact sheet of images is saved, False
                            otherwise.
  --help                    Show this message and exit.
```
