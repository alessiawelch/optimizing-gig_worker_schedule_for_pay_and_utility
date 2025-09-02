from __future__ import annotations

import math
import importlib
from types import SimpleNamespace


from typing import Tuple

import numpy as np
from collections import deque
import pandas as pd
import pytest

env_mod = importlib.import_module(
    "single_driver_simulated_environment.simulated_env"
) 
SingleDriverEnv = env_mod.SingleDriverEnv
C = importlib.import_module("single_driver_simulated_environment.config")

class StubDriver:
    def __init__(self, driver_id, start_loc, start_date):
        self.id = driver_id
        self.loc = start_loc
        self.grid = (0, 0)
        self.on_job = False
        self.job_timer = 0
        self.current_job = None
        self.idle_period = 0
        self.past_locations = []
        self.past_offers = []
        self.rewards = 0.0
        self.pay = 0.0
        self.repositioning_km = 0.0
        self.repositioning = False
        self.route_queue = deque()
        self.start_date = start_date
        self.current_time = pd.Timestamp(start_date)
        self.FAST = False

    # Environment will call these:
    def update(self):
        if self.on_job:
            self.job_timer -= 1
            if self.job_timer <= 0:
                self.on_job = False
                self.current_job = None
        self.current_time = self.current_time + pd.Timedelta(minutes=1)

    def start_job(self, ride, epoch, pickup_info=None, ride_info=None, price_pred=0.0):
        self.repositioning = False
        self.repositioning_km = 0.0
        self.idle_period = 0
        sec = 120.0
        km  = 1.0
        pay = 10.0
        self.current_job = (ride, sec, km, pay, epoch)
        self.job_timer = 2
        self.on_job = True

    def move_locations(self, *_a, **_kw):
        return None, 1.0, 60.0, [(0, 0), (0.01, 0.01)]

    def job_information(self, offer):
        pickup_info = {"stub": True}
        ride_info   = {"stub": True}
        price_pred  = 30.0
        return pickup_info, ride_info, price_pred

    def fake_reject_move(self, lat, lng):
        self.repositioning     = True
        self.repositioning_km      = 1.0              
        # queue element: (poly, sec_left, end_loc)
        self.route_queue = deque([("poly", 180.0, (lat, lng))])
        return 3.0                                    

    def fake_update(self):
        # pop one minute from the dummy queue
        if self.route_queue:
            poly, sec, end = self.route_queue[0]
            sec -= 60.0
            if sec <= 0:
                self.route_queue.popleft()
            else:
                self.route_queue[0] = (poly, sec, end)
        # clear flag when done
        if not self.route_queue and self.repositioning:
            self.repositioning = False
            self.repositioning_km  = 0.0

# Helper to monkey‑patch the environment dependencies
def _patch_env(monkeypatch: pytest.MonkeyPatch):
    # stub Driver
    monkeypatch.setattr(env_mod, "Driver", StubDriver, raising=True)

    monkeypatch.setattr(
        env_mod,                  
        "DEMAND_TABLE",
        {h: .1 for h in range(24)},     
        raising=True
    )

    # uniform probability matrix 
    def _uniform(self):
        P = np.full((self.grid_row, self.grid_col), 1.0, dtype=np.float32)
        return P / P.size        

    monkeypatch.setattr(SingleDriverEnv, "_current_prob", _uniform, raising=True)


# ensure that the reset works correctly
def test_reset_observation(monkeypatch: pytest.MonkeyPatch):
    _patch_env(monkeypatch)
    env = SingleDriverEnv(grid_row=3, grid_col=3, max_epochs=5)
    obs = env.reset()

    assert env.observation_space.contains(obs), "observation not present"
    assert obs.shape == (23,), "size is incorrect"
    assert obs.dtype == np.float32, "type is wrong"


# ensure step works with no offers
def test_step_idle_reward(monkeypatch: pytest.MonkeyPatch):
    _patch_env(monkeypatch)
    env = SingleDriverEnv(grid_row=3, grid_col=3, max_epochs=5)
    env.reset()
    P_here = 1 / (env.grid_row * env.grid_col)

    # Force empty ride pool and driver idle
    env.ride_pool.clear()

    obs, reward, done, info = env.step(1)  # REJECT_STAY

    # After REJECT_STAY the env increments idle_period to 1 before computing penalty
    expected_reward = (
        C.FIXED_COST_PER_MIN_USD
        + P_here * C.LOCATION_BONUS * 1e2
        + C.IDLE_PENALTY * 1  # post-increment idle period
    )

    assert not done
    assert math.isclose(reward, expected_reward, rel_tol=1e-4)
    assert env.observation_space.contains(obs)
    assert info["offer"] is None


# ensure the pay is the expected outcome
def test_accept_ride_reward(monkeypatch: pytest.MonkeyPatch):
    _patch_env(monkeypatch)
    env = SingleDriverEnv(grid_row=3, grid_col=3, max_epochs=5)
    env.reset()

    # insert a deterministic ride at epoch 0
    ride = (C.LAT_MIN + 0.01, C.LON_MIN + 0.01, C.LAT_MIN + 0.02, C.LON_MIN + 0.02, 0)
    env.ride_pool.append(ride)

    # Patch _select_closest to return our ride directly
    monkeypatch.setattr(env, "_select_closest", lambda loc: ride)
    dummy_pi = (1.0, 60.0, [])  # pickup_info  : (km, sec, route)
    dummy_ri = (1.0, 60.0, [])  # ride_info    : (km, sec, route)
    monkeypatch.setattr(
        env,
        "_stolen_ride",
        lambda _offer: (False, dummy_pi, dummy_ri, 10.0),  # not stolen
        raising=True,
    )

    obs, reward, done, info = env.step(0)  # ACCEPT_RIDE

    # Probability in square (uniform), but no bonus is added while on_job=True
    # Driver starts with pay=0, so adjusted_pay = 4 * 10.0 = 40.0
    expected_reward = (
        C.FIXED_COST_PER_MIN_USD  # base per-minute cost
        + 10.0 * 4              
    )

    assert math.isclose(reward, expected_reward, rel_tol=1e-4)
    assert info["on_job"] is True
    assert len(env.ride_pool) == 0  # ride removed
    assert env.observation_space.contains(obs)

# ensure that it finished at correct epoch
def test_done_flag(monkeypatch: pytest.MonkeyPatch):
    _patch_env(monkeypatch)
    env = SingleDriverEnv(grid_row=3, grid_col=3, max_epochs=2)
    env.reset()

    for _ in range(2):
        obs, reward, done, _ = env.step(1) 
    assert done is True

# ensure the action is translated correctly
def test_translate_action(monkeypatch: pytest.MonkeyPatch):
    _patch_env(monkeypatch)
    env = SingleDriverEnv(grid_row=3, grid_col=3)

    # Accept
    act, r, c = env._translate_action(0)
    assert act == C.ACCEPT_RIDE and (r, c) == (0, 0)

    # Reject stay
    act, r, c = env._translate_action(1)
    assert act == C.REJECT_STAY and (r, c) == (0, 0)

    # Move to row 2 col 1
    idx = 2 + 2 * env.grid_col + 1  
    act, r, c = env._translate_action(idx)
    assert act == C.REJECT_MOVE and (r, c) == (2, 1)


# ensure that the local probabilities are chosen correctly
def test_local_prob_window(monkeypatch: pytest.MonkeyPatch):
    k = C.WINDOW_SIZE
    _patch_env(monkeypatch)
    env = SingleDriverEnv(grid_row=7, grid_col=7)
    env.reset()

    # place driver off‑centre to test edge‑handling
    env.driver.grid = (5, 5)
    win = env._local_prob_window(k)

    assert win.shape == (k, k)
    # central element equals probability at driver.grid
    assert math.isclose(win[k // 2, k // 2], env._current_prob()[5, 5])
    # corners outside grid remain zero
    if 5 + k//2 >= env.grid_row:
        assert win[-1, -1] == 0.0


# test that the rides are generated correctly
def test_generate_epoch_rides(monkeypatch: pytest.MonkeyPatch):
    _patch_env(monkeypatch)
    env = SingleDriverEnv(grid_row=4, grid_col=4, max_epochs=1)
    env.reset()

    env.ride_pool.clear()
    env._generate_epoch_rides()

    # Expected ride count ≤ max_rides_per_epoch and > 0
    assert 0 <= len(env.ride_pool) <= 3

    for ride in env.ride_pool:
        olat, olng, dlat, dlng, _ = ride
        # Within bounds
        assert C.LAT_MIN <= olat <= C.LAT_MAX
        assert C.LON_MIN <= olng <= C.LON_MAX
        assert C.LAT_MIN <= dlat <= C.LAT_MAX
        assert C.LON_MIN <= dlng <= C.LON_MAX
        # origin and dest not equal
        assert (olat, olng) != (dlat, dlng)


@pytest.mark.parametrize(
    "epoch, expected",
    [
        (0, (C.START_HOUR % 24, 0)),             # very first tick
        (59, (C.START_HOUR % 24, 59)),           # last minute of that hour
        (60, ((C.START_HOUR + 1) % 24, 0)),      # one hour later
        (1440, (C.START_HOUR % 24, 0)),          # exactly 24h later
    ],
)

# testing that the epoch converts correctly
def test_epoch_to_time(epoch, expected):
    env = SingleDriverEnv(grid_row=1, grid_col=1, max_epochs=epoch + 1)
    env.epoch = epoch
    env.driver.current_time = env.start_date + pd.Timedelta(minutes=epoch)
    assert env._epoch_to_time() == expected


def test_current_prob_returns_correct_hour():
    # Build a dummy schedule where slice[h] = h (easy to spot errors)
    schedule = np.arange(24, dtype=np.int32).reshape(24, 1, 1)

    env = SingleDriverEnv(grid_row=1, grid_col=1, origin_prob_schedule=schedule)
    for hour in range(24):
        env.epoch = (hour - C.START_HOUR) % 24 * 60
        env.driver.current_time = env.start_date + pd.Timedelta(minutes=env.epoch)
        prob = env._current_prob()
        assert prob.shape == (1, 1)
        assert prob[0, 0] == hour

def test_median_fare_full_and_backup(monkeypatch):
    env = SingleDriverEnv(grid_row=4, grid_col=4)
    full_key   = (0, 0, 1, 1, "mid_day")
    origin_key = (0, 0, "mid_day")

    monkeypatch.setattr(env_mod, "PRICE_FULL",   {}, raising=True)
    monkeypatch.setattr(env_mod, "PRICE_BACKUP", {}, raising=True)

    # fake price tables
    monkeypatch.setitem(env_mod.PRICE_FULL,   full_key,   25.0)
    monkeypatch.setitem(env_mod.PRICE_BACKUP, origin_key, 18.0)

    assert env._median_fare(*full_key) == 25.0               # full hit
    assert env._median_fare(0, 0, 2, 2, "mid_day") == 18.0   # backup hit
    assert env._median_fare(3, 3, 2, 2, "night") is None     # miss


@pytest.mark.parametrize("premium, rng_val, expect_stolen", [ 
    (2.0, 0.0,  True), 
    (0.0, 0.99, False),
])
def test_stolen_ride_branches(monkeypatch, premium, rng_val, expect_stolen):
    _patch_env(monkeypatch)
    env = SingleDriverEnv(grid_row=1, grid_col=1)
    env.NORMALIZE_COEFFICIENT = 0.0  # control for test

    offer = (*C.START_STATE, C.START_STATE[0]+0.01, C.START_STATE[1]+0.01, env.epoch)

    monkeypatch.setattr(env.driver, "job_information",
                         lambda _o: ({}, {}, (1+premium)*10.0))
    monkeypatch.setattr(env, "_median_fare", lambda *_a, **_k: 10.0)
    monkeypatch.setattr(env, "rng", SimpleNamespace(random=lambda: rng_val))

    got, *_ = env._stolen_ride(offer)
    assert got is expect_stolen

def test_dest_candidates_detailed(monkeypatch):
    factor = env_mod.AVG_DISTANCE_GRID_FACTOR  

    fake_full = {
        (0, 0, 1, 2, "mid_day"): {"median_dist_km": 2.0, "ride_count": 10, "rf_global": 0.5},
        (0, 0, 2, 3, "mid_day"): {"median_dist_km": 1.0, "ride_count":  5, "rf_global": 0.2},
        (1, 1, 4, 4, "mid_day"): {"median_dist_km": 1.0, "ride_count":100, "rf_global": 1.0},
    }
    monkeypatch.setattr(env_mod, "DISTANCE_FULL",   fake_full, raising=True)
    monkeypatch.setattr(env_mod, "DISTANCE_BACKUP", {},       raising=True)


    env = SingleDriverEnv(grid_row=5, grid_col=5, max_epochs=1)

    cands = env._dest_candidates(orow=0, ocol=0, tg="mid_day")
    assert set(cands) == {(1, 2, 2.5), (2, 3, 1.0)}

def test_dest_candidates_none(monkeypatch):
    env = SingleDriverEnv(grid_row=3, grid_col=3, max_epochs=1)

    monkeypatch.setattr(env_mod, "DISTANCE_FULL",   {}, raising=True)
    monkeypatch.setattr(env_mod, "DISTANCE_BACKUP", {}, raising=True)

    # no data at all for (1,1,'evening')
    assert env._dest_candidates(orow=1, ocol=1, tg="evening") == []

