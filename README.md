# adam_test_data: Test data generation for ADAM and related software
#### A Python package by the Asteroid Institute, a program of the B612 Foundation
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://img.shields.io/badge/Python-3.11%2B-blue)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![pip - Build, Lint, Test, and Coverage](https://github.com/B612-Asteroid-Institute/adam_test_data/actions/workflows/pip-build-lint-test-coverage.yml/badge.svg)](https://github.com/B612-Asteroid-Institute/adam_test_data/actions/workflows/pip-build-lint-test-coverage.yml)

## Installation

`adam_test_data` can currently be installed from source:

```pip install .[dev]```

## Usage

Loading in population models:

```python
from adam_test_data.datasets import load_S3M
from adam_test_data.datasets import load_P9

S3M_DIR = "S3M_v09.05.15"
P9_DIR = "P9"

S3M = load_S3M(S3M_DIR)
P9 = load_P9(P9_DIR)
```

Loading in a pointing table for a survey or telescope. Here we create one for NSC DR2:

```python
import numpy as np
import pandas as pd
import pyarrow as pa

from astropy.time import Time
from adam_test_data.observatories.presets import load_W84
from adam_test_data.pointings import Pointings

w84 = load_W84()

# Load in the NSC DR2 exposures (this is external data not included in the package)
nsc_dr2_exposures = pd.read_csv("nsc_dr2_exposure.csv")
nsc_dr2_exposures["depth5sig"] = nsc_dr2_exposures["depth95"]
nsc_dr2_exposures_w84 = nsc_dr2_exposures[nsc_dr2_exposures["instrument"] == "c4d"]

# Create a Pointings object using the exposures info
w84_pointings = Pointings.from_kwargs(
    observationId=nsc_dr2_exposures_w84["exposure"],
    observationStartMJD_TAI=Time(nsc_dr2_exposures_w84["mjd"], format="mjd", scale="utc").tai.mjd,
    visitTime=nsc_dr2_exposures_w84["exptime"],
    visitExposureTime=nsc_dr2_exposures_w84["exptime"],
    filter=nsc_dr2_exposures_w84["filter"],
    seeingFwhmGeom_arcsec=nsc_dr2_exposures_w84["fwhm"],
    seeingFwhmEff_arcsec=nsc_dr2_exposures_w84["fwhm"],
    fieldFiveSigmaDepth_mag=nsc_dr2_exposures_w84["depth5sig"],
    fieldRA_deg=nsc_dr2_exposures_w84["ra"],
    fieldDec_deg=nsc_dr2_exposures_w84["dec"],
    rotSkyPos_deg=np.zeros(len(nsc_dr2_exposures_w84)),
    observatory_code=pa.repeat("W84", len(nsc_dr2_exposures_w84)),
    name="NSC",
)
```

Generate test data for one of the populations and one of the observatories:
```python
from adam_test_data.main import generate_test_data

catalog_file, noise_files, summary = generate_test_data(
    "S3M_NSC_W84", 
    s3m, 
    w84_pointings, 
    w84, 
    noise_densities=[100, 200], 
    max_processes=30, 
    chunk_size=500, 
    cleanup=True
)
```