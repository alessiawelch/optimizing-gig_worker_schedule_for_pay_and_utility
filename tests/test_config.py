from __future__ import annotations

import collections
import re
from typing import Tuple

import numpy as np
import pytest

from single_driver_simulated_environment import config as C

# computed grid extents must reach or exceed configured LAT/LON max
def test_grid_covers_domain() -> None:
    lat_extent = C.GRID_ROW * C.CELL_DEG
    lon_extent = C.GRID_COL * C.CELL_DEG

    assert C.LAT_MIN + lat_extent >= C.LAT_MAX - 1e-9
    assert C.LON_MIN + lon_extent >= C.LON_MAX - 1e-9

# START_STATE must lie inside the bounding box
def test_start_state_in_bounds() -> None:
    lat, lng = C.START_STATE
    assert C.LAT_MIN <= lat <= C.LAT_MAX
    assert C.LON_MIN <= lng <= C.LON_MAX

# check conversion is consistent
def test_currency_conversion_fixed_cost() -> None:
    rate = C.FIXED_COST_PER_MIN_USD / C.FIXED_COST_PER_MIN_POUNDS
    assert pytest.approx(rate, rel=1e-8) == 1.36

# check conversion is consistent
def test_currency_conversion_distance_cost() -> None:
    rate = C.COST_PER_KM_USD / C.COST_PER_KM_POUNDS
    assert pytest.approx(rate, rel=1e-8) == 1.36

# ensure the hourly probabilities tensor still works
def test_hourly_probabilities_shape_and_sum() -> None:
    hp = C.HOURLY_PROBABILITIES

    # Expect shape (24, rows, cols)
    assert hp.shape == (24, C.GRID_ROW, C.GRID_COL)

    # Each hour should form a proper probability distribution.
    for h in range(24):
        assert np.isclose(hp[h].sum(), 1.0, atol=1e-6)
        assert (hp[h] >= 0).all()

# ensure the osrm text works
def test_osrm_fmt_placeholders() -> None:
    tokens = re.findall(r"{(\w+)}", C.OSRM_FMT)
    counts = collections.Counter(tokens)

    expected = {
        "lon1": 1,
        "lat1": 1,
        "lon2": 1,
        "lat2": 1,
    }
    assert counts == expected
