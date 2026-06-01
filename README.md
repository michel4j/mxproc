# mxproc
MX Automated Data Processing Pipeline

This project is the successor to the AutoProcess pipeline. The goal is to support not just XDS processing as
AutoProcess did, but also DIALS eventually.

A user-friendly graphical interface for interactive use is also planned for the future.

## Installation

You can install `mxproc` via `pip`:

```bash
pip install mxproc
```

Alternatively, if you are using [Poetry](https://python-poetry.org/):

```bash
poetry add mxproc
```

## Available Scripts

Once installed, the following command-line scripts are available for use in your data processing pipelines:

* `auto.init`: Initialize data processing
* `auto.xds`: Alias for "xds_par" on a cluster environment, see command line arguments
* `auto.index`: Run auto-indexing
* `auto.spots`: Find spots for indexing
* `auto.integrate`: Run integration
* `auto.process`: Run complete data processing [start with this command]
* `auto.scale`: Run scaling
* `auto.strategy`: Determine data collection strategy
* `auto.symmetry`: Determine crystal symmetry

## Dependencies

- Python >= 3.10
- `mxio`, `parsefire`, `matplotlib`, `numpy`, `pyyaml`, `tqdm`, `pandas`, `prettytable`, `vg`, `scipy`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
