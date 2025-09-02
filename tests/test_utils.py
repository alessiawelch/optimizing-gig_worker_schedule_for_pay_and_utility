from __future__ import annotations
import numpy as np
from single_driver_simulated_environment import utils as U
from single_driver_simulated_environment.config import LAT_MIN, CELL_DEG, LON_MIN, LAT_MAX, LON_MAX, GRID_ROW, GRID_COL
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple
from unittest.mock import Mock
import pytest
import requests  



@pytest.mark.parametrize(
    "pt",
    [
        (0.0, 0.0),
        (LAT_MIN, LON_MIN),
        (51.5, -0.2),
    ],
)

# distance to itself should be 0
def test_calc_distance_zero(pt: Tuple[float, float]):
    assert U.calc_distance(pt, pt) < 1e-9


@pytest.mark.parametrize(
    "A,B,C",
    [
        ((51.5, -0.15), (51.49, -0.12), (51.51, -0.17)),
        ((LAT_MIN, LON_MIN), (LAT_MIN + 0.3, LON_MIN + 0.4), (LAT_MIN + 0.6, LON_MIN + 0.1)),
    ],
)

# distance obeys the triangle inequality
def test_calc_distance_triangle_inequality(A, B, C):
    ab = U.calc_distance(A, B)
    bc = U.calc_distance(B, C)
    ac = U.calc_distance(A, C)
    assert ac <= ab + bc + 1e-6

# distance between AB and BA should be same
def test_calc_distance_symmetry():
    A = (51.5, -0.15)
    B = (51.49, -0.22)
    assert np.isclose(U.calc_distance(A, B), U.calc_distance(B, A))

# test grid locations are accurate
def test_location_to_grid_edges():
    cases = [
        (LAT_MIN, LON_MIN, (0, 0)),
        (LAT_MIN, LON_MIN + CELL_DEG, (0, 1)),
        (LAT_MIN + CELL_DEG, LON_MIN, (1, 0)),
        (LAT_MAX, LON_MAX, (GRID_ROW-1, GRID_COL-1)),
        (LAT_MAX - 1e-8, LON_MAX - 1e-8, (GRID_ROW-1, GRID_COL-1))
    ]
    for lat, lng, expected in cases:
        assert U.location_to_grid(lat, lng) == expected

# random locations should always fall inside their originating cell
def test_grid_random_location_in_bounds():
    for r in range(10):
        for c in range(10):
            lat, lng = U.grid_to_random_location(r, c)
            assert LAT_MIN + r * CELL_DEG <= lat < LAT_MIN + (r + 1) * CELL_DEG
            assert LON_MIN + c * CELL_DEG <= lng < LON_MIN + (c + 1) * CELL_DEG

# ensure location to grid works twice
def test_location_grid_round_trip_mid_cell():
    lat, lng = 51.455, -0.123
    r, c = U.location_to_grid(lat, lng)
    lat2 = LAT_MIN + r * CELL_DEG + 0.5 * CELL_DEG
    lng2 = LON_MIN + c * CELL_DEG + 0.5 * CELL_DEG
    r2, c2 = U.location_to_grid(lat2, lng2)
    assert (r, c) == (r2, c2)


@pytest.fixture()
def _fake_osrm_resp():
    """Return a factory that builds a fake OSRM JSON payload."""

    def _factory(sec: float, km: float, olat=51.5, olng=-0.1, dlat=51.49, dlng=-0.12):
        return {
            "routes": [
                {
                    "duration": sec,
                    "distance": km * 1000,
                    "geometry": {
                        "coordinates": [
                            [olng, olat],
                            [dlng, dlat],
                        ]
                    },
                }
            ]
        }

    return _factory

# when OSRM responds 200 OK, helper returns parsed values and fallback=False
def test_osrm_call_success(monkeypatch: pytest.MonkeyPatch, _fake_osrm_resp):

    def _ok_req(url, timeout):
        m = Mock()
        m.raise_for_status = lambda: None
        m.json.return_value = _fake_osrm_resp(sec=120, km=4.2)
        return m

    monkeypatch.setattr(requests, "get", _ok_req)

    sec, km, poly, fallback = U.osrm_call(51.5, -0.1, 51.49, -0.12)
    assert sec == 120
    assert km == 4.2
    if isinstance(poly, dict) and "coordinates" in poly:
        poly = [(lat, lon) for lon, lat in poly["coordinates"]]

    assert poly == [(51.5, -0.1), (51.49, -0.12)]
    assert fallback is False

# if every request errors, helper falls back to calc distance and marks fallback=True
def test_osrm_call_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(requests, "get", lambda *a, **kw: (_ for _ in ()).throw(requests.Timeout))

    sec, km, poly, fallback = U.osrm_call(51.5, -0.1, 51.49, -0.12, retries=1)

    expected_km = U.haversine_vec(51.5, -0.1, 51.49, -0.12)
    assert math.isclose(km, expected_km, rel_tol=1e-6)
    assert fallback is True
