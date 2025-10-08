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
from adam_test_data.datasets import load_NEOMOD

S3M_DIR = "S3M_v09.05.15"
P9_DIR = "P9"
NEOMOD_FILE = "NEOMOD.txt"

S3M = load_S3M(S3M_DIR)
P9 = load_P9(P9_DIR)
NEOMOD = load_NEOMOD(NEOMOD_FILE, seed=20251002)
```

Create a simulated survey:
```python
import numpy as np

from adam_test_data.survey import Survey, SurveyFootprint, create_survey_pointings
from adam_test_data.observatories.observatory import FieldOfView

days = 90
start_night = 61000
end_night = start_night + days

x05 = load_X05()
x05.fov = FieldOfView(
    camera_model="circle",
    fill_factor=1.0,
    circle_radius=1.5 * np.sqrt(10 / np.pi),
    footprint_edge_threshold=None,
)

survey = Survey(
    name="X05-quad",
    observatory_code="X05",
    local_observing_start_time=9,
    observing_duration=8,
    visits_per_night=4,
    start_night=start_night,
    end_night=end_night,
    filters=["u", "g", "r", "i", "z", "y"],
    exposure_time=30,
    slew_time=5,
    max_zenith_angle=70,
)

survey_footprint = SurveyFootprint.create(nside=16)
survey_pointings = create_survey_pointings(survey, survey_footprint, {"X05": x05})
survey_pointings.to_parquet("survey_pointings_20251002.parquet")
```

Create a synthethic catalog for the desired populations:
```python
import quivr as qv
from adam_test_data.main import generate_test_data

population = qv.concatenate([S3M, P9, NEOMOD])

catalog_file, noise_files, summary = generate_test_data(
    "Population-X05-Quads-20251002", 
    population, 
    survey_pointings, 
    x05, 
    noise_densities=[10, 100, 500, 1000], 
    max_processes=40, 
    chunk_size=500, 
    noise_chunk_size=20,
    cleanup=True
)
```