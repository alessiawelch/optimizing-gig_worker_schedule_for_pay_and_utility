from __future__ import annotations

import math
from collections import deque
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from single_driver_simulated_environment.config import LAT_MIN, LON_MIN, SECONDS_PER_EPOCH

import importlib

driver_mod = importlib.import_module(
    "single_driver_simulated_environment.driver"
) 
Driver = driver_mod.Driver

# Return deterministic (sec, km, polyline, fallback=False)
def _stub_osrm_call(olat, olng, dlat, dlng, retries=3):
    km = 1.0
    sec = 60.0
    polyline = [(olat, olng), (dlat, dlng)]
    return sec, km, polyline, False

# Return deterministic (sec, km, polyline, fallback=False)
def _stub_osrm_call2(olat, olng, dlat, dlng, retries=3):
    km = 2.0
    sec = 120.0
    polyline = [(olat, olng), (0, 0), (dlat, dlng)]
    return sec, km, polyline, False


# Patch model prediction helpers to deterministic constants
def _patch_ml(monkeypatch: pytest.MonkeyPatch, ratio=1.0, duration=60.0, price=10.0):
    monkeypatch.setattr(driver_mod, "predict_distance_ratio", lambda df: np.full(len(df), ratio))
    monkeypatch.setattr(driver_mod, "predict_duration", lambda df: np.full(len(df), duration))
    monkeypatch.setattr(driver_mod, "predict_price", lambda df: price)


# test the constructor works
def test_driver_initial_state(monkeypatch: pytest.MonkeyPatch):
    _patch_ml(monkeypatch)
    monkeypatch.setattr(driver_mod, "osrm_call", _stub_osrm_call)

    start_loc = (LAT_MIN + 0.01, LON_MIN + 0.01)
    d = Driver("d1", start_loc)

    assert d.id == "d1"
    assert d.loc == start_loc
    assert d.on_job is False and d.job_timer == 0
    assert list(d.grid) == list(driver_mod.location_to_grid(*start_loc))
    assert d.route_queue == deque()


# test that the reject move action works
def test_reject_move_action(monkeypatch: pytest.MonkeyPatch):
    _patch_ml(monkeypatch)
    monkeypatch.setattr(driver_mod, "osrm_call", _stub_osrm_call)

    d = Driver("d2", (LAT_MIN + 0.02, LON_MIN + 0.02))

    new_loc = (LAT_MIN + 0.03, LON_MIN + 0.03)
    km_pred = d.reject_move_action(*new_loc)


    # Exactly one segment enqueued
    assert len(d.route_queue) == 1


# test start job action
def test_start_job(monkeypatch):
    _patch_ml(monkeypatch)
    monkeypatch.setattr(driver_mod, "osrm_call", _stub_osrm_call, raising=True)

    d = Driver("d3", (LAT_MIN + .04, LON_MIN + .04))
    d.route_queue.clear()
    d.route_queue.append(([[]], 10, (1,0)))
    ride = (LAT_MIN + .05, LON_MIN + .05,
            LAT_MIN + .06, LON_MIN + .06, 0)

    pi, ri, pp = d.job_information(ride)
    d.start_job(ride, epoch=0, pickup_info=pi, ride_info=ri, price_pred=pp)

    assert d.on_job is True
    assert d.current_job[0] == ride


def test_start_job_resets_repositioning(monkeypatch):
    _patch_ml(monkeypatch, ratio=1.0, duration=60.0, price=9.99)
    monkeypatch.setattr(driver_mod, "osrm_call", _stub_osrm_call)

    d = Driver("R2", (LAT_MIN + .03, LON_MIN + .03))

    # Kick-off a dummy reposition first
    d.reject_move_action(LAT_MIN + .04, LON_MIN + .04)
    assert d.repositioning is True
    d.route_queue.append(([], 5, (4,4)))

    # Prepare a ride and associated info
    ride = (LAT_MIN + .05, LON_MIN + .05,
            LAT_MIN + .06, LON_MIN + .06, 0)
    pi, ri, pp = d.job_information(ride)

    d.start_job(ride, epoch=0, pickup_info=pi, ride_info=ri, price_pred=pp)

    assert d.on_job is True
    assert d.repositioning is False
    assert math.isclose(d.repositioning_km, 0.0)
    # Route-queue now contains pickup- & trip-segments (2 items expected)
    assert len(d.route_queue) == 2

def test_advance_route_many_segments(monkeypatch):
    monkeypatch.setattr(driver_mod, "osrm_call", _stub_osrm_call)

    d = Driver("seg", (LAT_MIN, LON_MIN))
    poly = [(0,0)] + [(i*0.001, 0) for i in range(1,11)]       
    d.enque_route_info(poly, 10*60.0, poly[-1])

    for i in range(0, 9):
        d.update()               
        # Still 10 vertices left
        assert len(d.route_queue[0][0]) == (10-i)
        # Timer went down by 60 s
        assert math.isclose(d.route_queue[0][1], (9-i)*60.0)


def test_advance_route_many_segments(monkeypatch):
    monkeypatch.setattr(driver_mod, "osrm_call", _stub_osrm_call)


    d = Driver("seg", (LAT_MIN, LON_MIN))
    poly = [(0,0)] + [(i*0.001, 0) for i in range(1,11)]      
    d.enque_route_info(poly, 10*60.0, poly[-1])

    for i in range(0, 9):
        d.update()    
        # Still 10 vertices left
        assert len(d.route_queue[0][0]) == (10-i)
        # Timer went down by 60 s
        assert math.isclose(d.route_queue[0][1], (9-i)*60.0)

def _stub_osrm_call3(*_):
    return 120.0, 3.0, [(0,0), (1,1), (2,2)], False


def test_reject_move_same_cell(monkeypatch):
    _patch_ml(monkeypatch)
    monkeypatch.setattr(driver_mod, "osrm_call", _stub_osrm_call)

    d = Driver("same", (LAT_MIN+.7, LON_MIN+.7))
    r, c = d.grid
    d.reject_move_action(*d.loc)

    assert d.repositioning
    assert len(d.route_queue) >= 1


def test_advance_along_route_handles_empty_polyline(monkeypatch):
    d = Driver("T0", (LAT_MIN+0.1, LON_MIN+0.1))
    empty_seg = ([], 30.0, (LAT_MIN+0.11, LON_MIN+0.11))

    # should return without error and snap to end_location
    advanced, new_loc = d.advance_along_route(empty_seg)

    assert new_loc == empty_seg[2]         
    assert advanced[0] == []                
    assert advanced[1] == 0.0             
