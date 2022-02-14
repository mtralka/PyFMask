# PyFMask4.3

## Installation

This program uses `poetry` for package management. Note, `GDAL` **is** required but not listed in the `pyproject.toml` due to build limitations on pypi. `GDAL` must be installed in the runtime environment.

### Build and install from source

- Ensure poetry is installed

- Navigate to the directory containing `pyproject.toml`

- `poetry install`

- `poetry build`

Follow steps for **Install from pre-built `.whl`** to install the project locally

**or**

The `pyfmask` script configured within poetry allows for simple CLI access without installation

- `poetry run pyfmask [ARGS]`

### Install from pre-built '.whl'

Although this package is not listed on PyPi due to project restrictions, we can achieve the same function result from distributing the project `.whl`s and installing locally through pip/pipx/etc.

- `pip install path/to/.whl`

Now, `pyfmask` is fully installed and accessible anywhere in the installed env

### Use without installation

- As shown in `example.ipynb`, you can import `PFmask` from `pyfmask` from a file in the same root directory

## Usage

### CLI

```shell
pyfmask [ARGS]
```

### Object

```python
from pyfmask import Fmask

infile: str = path/to/infile/{*._MTL.txt, MTD_*.xml}
outfile_dir: str = path/to/outfile/dir
dem_path: str = path/to/dem/dir
gswo_path: str = path/to/gswo/dir
out_name: str = "pyfmask_example.tif"

control = FMask(infile=infile, out_dir=outfile_dir,dem_path=dem_path, 
gswo_path=gswo_path, out_name=out_name, auto_save=True, save_cloud_prob=False, auto_run=True
)

```
