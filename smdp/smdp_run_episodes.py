from __future__ import annotations


from datetime import date, datetime, timedelta
import random
import pandas as pd
import numpy as np
import os, copy, csv
from typing import List, Dict, Tuple, Callable, Optional
import torch

from single_driver_simulated_environment_new.simulated_env import SingleDriverEnv
from single_driver_simulated_environment_new.config import REJECT_STAY, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, CELL_DEG, GRID_COL, GRID_ROW
from single_driver_simulated_environment_new.utils import (
    location_to_price_grid, hour_to_group, location_to_grid, grid_to_random_location
)
from smdp.value_net import ValueNet
from smdp.planner import td_target


SCENARIO_RNG_SEED = 10
_srng = random.Random(SCENARIO_RNG_SEED)

def _random_date_between(srng: random.Random, start_year=2025, end_date=None) -> date:
    if end_date is None:
        end_date = datetime.today().date()
    start = date(start_year, 1, 1)
    span_days = (end_date - start).days
    return start + timedelta(days=srng.randint(0, span_days))

def _random_start_loc_in_grid(srng: random.Random):
    r = srng.randrange(GRID_ROW)
    c = srng.randrange(GRID_COL)
    return grid_to_random_location(r, c, rng=srng)  

def make_random_scenarios(runs=8, n=20,
                          hour_min=12, hour_max=17,
                          minute_choices=(0, 0)):

    scenarios = []
    sample = [
        [date(2020,2,21), 10, 30, (51.50685119628906,-0.009800000116229), 2454155475],
        [date(2019,6,15), 8, 45, (51.50913619995117,-0.0611257627606391), 1551164510],
        [date(2020,3,6), 9, 30, (51.54750061035156,-0.0284700002521276), 4163490463],
        [date(2020,3,8), 10, 0, (51.5446891784668,-0.0323599986732006), 2804517990],
        [date(2017,12,1), 11, 0, (51.53342819213867,-0.0572855211794376), 2436257803]
    ]
    for i in range(runs):       
        d       = sample[i][0]
        hour    = sample[i][1]  
        minute  = sample[i][2]
        lat, lng = sample[i][3]
        for _ in range(n):
            seed_32 = _srng.getrandbits(32)                 
            scenarios.append((seed_32, d.isoformat(), hour, minute, float(lat), float(lng)))
    return scenarios


SCENARIOS = make_random_scenarios(5, 5)




CELL_FILE_ADDITION = "smaller_pool_size"
CKPT_PATH         = "smdp_value_net_v2.pt"
POLICIES_TO_RUN   = ["always_accept", "value_net", "accept_if_premium", "accept_else_bestP", "random_uniform"]

PREMIUM_THRESHOLD = 0.10
GAMMA             = 0.995
TOP_K_MOVES       = 8
MAX_EPOCHS        = 600                 
OUT_RUNS_CSV      = f"eval_runs_{CELL_FILE_ADDITION}.csv"
OUT_SUMMARY_CSV   = f"eval_summary_{CELL_FILE_ADDITION}.csv"
TRACE_FIRST       = 0                
TRACE_LIMIT       = 1000
TRACE_OUTDIR      = f"eval_traces_{CELL_FILE_ADDITION}"


# Decision boundary 
def at_decision_boundary(env) -> bool:
    if getattr(env, "driver", None) is None:
        return True
    if env.driver.on_job:
        return False
    return (not getattr(env.driver, "repositioning", False)) or (env.last_offer is not None)

TICK = REJECT_STAY  

def move_action_index(env, row, col) -> int:
    return 2 + row * env.grid_col + col

def enumerate_options(env, top_k_moves: int = 8) -> List[int]:
    opts = [1]  # STAY
    if (env.last_offer is not None) and (not env.driver.on_job):
        opts.append(0)  # ACCEPT
    P = env._current_prob()
    r0, c0 = env.driver.grid
    flat = [(float(P[r, c]), r, c)
            for r in range(env.grid_row)
            for c in range(env.grid_col)
            if not (r == r0 and c == c0)]
    flat.sort(key=lambda t: t[0], reverse=True)
    for _, r, c in flat[:top_k_moves]:
        opts.append(move_action_index(env, r, c))
    return opts

def _add_unique_cell_from_loc(env, acc_set: set):
    lat, lng = env.driver.loc
    rr, cc = location_to_price_grid(lat, lng)
    acc_set.add((int(rr), int(cc)))

def _offer_stats(env) -> Tuple[bool, float, float]:
    present = (env.last_offer is not None) and (not env.driver.on_job)
    if not present:
        return False, -1.0, -1.0
    olat, olng, dlat, dlng, _ = env.last_offer
    hour, _ = env._epoch_to_time()
    o_row, o_col = location_to_price_grid(olat, olng)
    d_row, d_col = location_to_price_grid(dlat, dlng)
    tg = hour_to_group(hour)
    median = env._median_fare(o_row, o_col, d_row, d_col, tg) or -1.0
    pred = float(getattr(env, "last_offer_price", -1.0))
    return True, pred, median

def roll_option_collect(env, first_action_idx: int, gamma: float, acc: Dict[str, float]) -> Tuple[float, int, np.ndarray, bool]:
    rewards, k = [], 0
    obs, r, done, info = env.step(first_action_idx)
    rewards.append(r); k += 1
    # first minute stats
    acc["total_reward"] += float(r)
    acc["onride_min"]   += 1 if info.get("on_job", False) else 0
    acc["stolen"]       += 1 if info.get("stolen", False) else 0
    acc["p_sum"]        += float(info.get("P_here", 0.0))
    _add_unique_cell_from_loc(env, acc["unique_cells"])
    # continue until next boundary
    while (not done) and (not at_decision_boundary(env)):
        obs, r, done, info = env.step(TICK)
        rewards.append(r); k += 1
        acc["total_reward"] += float(r)
        acc["onride_min"]   += 1 if info.get("on_job", False) else 0
        acc["stolen"]       += 1 if info.get("stolen", False) else 0
        acc["p_sum"]        += float(info.get("P_here", 0.0))
        _add_unique_cell_from_loc(env, acc["unique_cells"])
    # discounted sum 
    g, R_disc = 1.0, 0.0
    for rr in rewards:
        R_disc += g * float(rr); g *= gamma
    return R_disc, k, obs, done

# Side-effect-free option scorer (deterministic)
@torch.no_grad()
def _score_option_on_fork(env, a, gamma):
    import random
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        fork = copy.deepcopy(env)
        rewards, k = [], 0
        obs, r, done, _ = fork.step(a); rewards.append(r); k += 1
        while (not done) and (not at_decision_boundary(fork)):
            obs, r, done, _ = fork.step(TICK)
            rewards.append(r); k += 1
        g, R_disc = 1.0, 0.0
        for rr in rewards:
            R_disc += g * float(rr); g *= gamma
        return R_disc, k, obs, done
    finally:
        import random
        random.setstate(py_state)
        np.random.set_state(np_state)

@torch.no_grad()
def smdp_scores_for_state(env, V, gamma: float, options: List[int]) -> int:
    recs = []
    for a in options:
        R_disc, k, s_next, done = _score_option_on_fork(env, a, gamma)
        recs.append((R_disc, k, s_next, done))
    idx_live = [i for i, (_, _, _, d) in enumerate(recs) if not d]
    v_next = np.zeros(len(recs), dtype=np.float32)
    if idx_live:
        s_batch = torch.as_tensor(np.stack([recs[i][2] for i in idx_live]), dtype=torch.float32)
        v_batch = V(s_batch).squeeze(-1).cpu().numpy()
        for j, i in enumerate(idx_live): v_next[i] = float(v_batch[j])
    scores = []
    for i, (R_disc, k, _, done) in enumerate(recs):
        scores.append(td_target(R_disc, k, 0.0 if done else v_next[i], gamma))
    scores = np.asarray(scores, dtype=np.float32)
    a_star = options[int(scores.argmax())]
    return a_star

# Policies
def policy_value_net(env, V, gamma, top_k_moves) -> int:
    options = enumerate_options(env, top_k_moves)
    return smdp_scores_for_state(env, V, gamma, options)

def policy_always_accept(env, **_) -> int:
    return 0 if (env.last_offer is not None and not env.driver.on_job) else 1

def policy_accept_else_bestP(env, **_) -> int:
    if (env.last_offer is not None) and (not env.driver.on_job):
        return 0
    P = env._current_prob()
    best_idx = int(np.argmax(P))
    r, c = divmod(best_idx, env.grid_col)
    if (r, c) == tuple(env.driver.grid):
        return 1
    return move_action_index(env, r, c)

def policy_topP_accept_only(env, q=0.10, **_):
    present = (env.last_offer is not None) and (not env.driver.on_job)
    P = env._current_prob()
    cutoff = np.quantile(P.ravel(), 1.0 - q)
    rr, cc = location_to_price_grid(env.driver.loc[0], env.driver.loc[1])
    rr, cc = int(rr), int(cc)
    in_hot = (0 <= rr < env.grid_row and 0 <= cc < env.grid_col and float(P[rr, cc]) >= cutoff)
    if present and in_hot:
        return 0  # ACCEPT
    # otherwise reposition to best-P (or stay if already there)
    best_idx = int(np.argmax(P)); r, c = divmod(best_idx, env.grid_col)
    return 1 if (r, c) == tuple(env.driver.grid) else move_action_index(env, r, c)

@torch.no_grad()
def policy_kstep_greedy(env, gamma=0.995, top_k_moves=8, **_):
    options = enumerate_options(env, top_k_moves)
    best_a, best_score = options[0], -1e30
    for a in options:
        R_disc, _, _, _ = _score_option_on_fork(env, a, gamma)
        if R_disc > best_score:
            best_score, best_a = R_disc, a
    return best_a

def policy_accept_if_premium(env, threshold: float = 0.25, **_) -> int:
    present, pred, median = _offer_stats(env)
    if present:
        if median > 0:
            premium = (pred - median) / max(1e-6, median)
        else:
            premium = 1.0 if pred > 0 else -1.0
        if premium > threshold:
            return 0  # ACCEPT
    # fallback
    P = env._current_prob()
    best_idx = int(np.argmax(P))
    r, c = divmod(best_idx, env.grid_col)
    if (r, c) == tuple(env.driver.grid):
        return 1
    return move_action_index(env, r, c)

def policy_random_uniform(env, top_k_moves=8, **_):
    options = enumerate_options(env, top_k_moves)
    # derive a per-state seed without touching global RNGs
    t = getattr(env.driver, "current_time", None)
    tkey = int(pd.Timestamp(t).value // 60_000_000_000) if t is not None else env.epoch  
    r, c = env.driver.grid
    seed = (tkey ^ (r << 16) ^ (c << 8)) & 0xFFFFFFFF
    rng = random.Random(seed)
    return rng.choice(options)

# Env preparation for one scenario
def prepare_env(max_epochs: int,
                seed: int,
                start_hour: int,
                start_minute: int,
                start_lat: float,
                start_lng: float,
                start_date: date) -> SingleDriverEnv:

    env = SingleDriverEnv(max_epochs=max_epochs)
    env.seed(int(seed) & 0xFFFFFFFF)
    env.reset()

    # Set calendar date & time directly on the driver clock
    base_dt = pd.Timestamp(start_date).replace(hour=int(start_hour), minute=int(start_minute), second=0, microsecond=0)
    if hasattr(env, "start_date"):
        env.start_date = base_dt
    if hasattr(env, "driver"):
        env.driver.current_time = base_dt

    # Set start location (lat,lng) and full-resolution grid
    if hasattr(env, "driver"):
        d = env.driver
        d.on_job = False
        d.repositioning = False
        if hasattr(d, "route_queue") and d.route_queue is not None:
            try: d.route_queue.clear()
            except Exception: pass
        d.job_timer = 0
        d.idle_period = 0
        d.current_job = None
        d.current_premium = 0.0
        d.loc = (float(start_lat), float(start_lng))
        r, c = location_to_grid(d.loc[0], d.loc[1])
        d.grid = (int(r), int(c))

    # Clear offer/pool and regenerate rides for the new time/place
    env.last_offer = None
    env.ride_pool = []
    if hasattr(env, "_generate_epoch_rides"): env._generate_epoch_rides()
    if hasattr(env, "_rebuild_origins_np"):  env._rebuild_origins_np()
    if hasattr(env, "_update_rank_cache"):   env._update_rank_cache()
    return env

# One episode on a provided env
def run_episode_on_env(env: SingleDriverEnv,
                       name: str,
                       policy_fn: Callable,
                       V: Optional[ValueNet],
                       gamma=0.995,
                       top_k_moves=8,
                       trace: bool = False,
                       trace_limit: int = 0,
                       trace_outdir: Optional[str] = None,
                       trace_tag: str = "") -> Dict[str, float]:

    obs, done = env._get_obs(), False

    decisions = 0
    counts = {"accept": 0, "stay": 0, "move": 0}
    offer_boundaries = 0
    Ks = []
    acc = {"total_reward": 0.0, "onride_min": 0, "stolen": 0, "p_sum": 0.0, "unique_cells": set()}

    trace_rows: List[Dict[str, object]] = []

    def _maybe_write_trace():
        if not trace or not trace_rows or not trace_outdir:
            return
        os.makedirs(trace_outdir, exist_ok=True)
        path = os.path.join(trace_outdir, f"{name}{trace_tag}_X.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
            w.writeheader()
            for row in trace_rows: w.writerow(row)
        print(f"[trace] wrote {len(trace_rows)} rows to {path}")

    while not done:
        if not at_decision_boundary(env):
            obs, r, done, info = env.step(TICK)
            acc["total_reward"] += float(r)
            acc["onride_min"]   += 1 if info.get("on_job", False) else 0
            acc["stolen"]       += 1 if info.get("stolen", False) else 0
            acc["p_sum"]        += float(info.get("P_here", 0.0))
            _add_unique_cell_from_loc(env, acc["unique_cells"])
            continue

        # decision boundary context
        hour, minute = env._epoch_to_time()
        P = env._current_prob()
        lat, lng = env.driver.loc
        rr, cc = location_to_price_grid(lat, lng)
        rr_i, cc_i = int(rr), int(cc)
        p_here = float(P[rr_i, cc_i]) if (0 <= rr_i < env.grid_row and 0 <= cc_i < env.grid_col) else 0.0

        present, pred_price, median_fare = _offer_stats(env)
        if present: offer_boundaries += 1

        # choose action & execute fully
        a = policy_fn(env, V=V, gamma=gamma, top_k_moves=top_k_moves)
        label = ("ACCEPT" if a == 0 else "STAY" if a == 1 else "MOVE")
        mrow = mcol = -1
        if a >= 2:
            adj = a - 2
            mrow, mcol = divmod(adj, env.grid_col)

        pre_total = acc["total_reward"]
        _, k, obs, done = roll_option_collect(env, a, gamma, acc)
        opt_reward = acc["total_reward"] - pre_total

        Ks.append(k)
        decisions += 1
        if a == 0:   counts["accept"] += 1
        elif a == 1: counts["stay"]   += 1
        else:        counts["move"]   += 1

        if trace and (len(trace_rows) < trace_limit):
            trace_rows.append({
                "epoch": env.epoch, "time_h": hour, "time_m": minute,
                "grid_r": rr_i, "grid_c": cc_i, "P_here": p_here,
                "offer_present": int(present),
                "pred_price": float(pred_price), "median_fare": float(median_fare),
                "action": label, "move_r": int(mrow), "move_c": int(mcol),
                "k_minutes": int(k), "opt_reward": float(opt_reward)
            })

    _maybe_write_trace()

    minutes = env.epoch
    hours = max(1e-6, minutes / 60.0)
    return {
        "name": name,
        "minutes": minutes,
        "hours": hours,
        "episode_return": float(acc["total_reward"]),
        "reward_per_hour": float(acc["total_reward"]) / hours,
        "pay_total": float(env.driver.pay),
        "pay_per_hour": float(env.driver.pay) / hours,
        "decisions": decisions,
        "accepts": counts["accept"],
        "stays": counts["stay"],
        "moves": counts["move"],
        "offer_boundaries": offer_boundaries,
        "accept_rate_at_offer": (counts["accept"] / offer_boundaries) if offer_boundaries else 0.0,
        "avg_option_minutes": float(np.mean(Ks)) if Ks else 0.0,
        "median_option_minutes": float(np.median(Ks)) if Ks else 0.0,
        "onride_minutes": int(acc["onride_min"]),
        "stolen_events": int(acc["stolen"]),
        "unique_cells_count": int(len(acc["unique_cells"])),
        "avg_p_here": float(acc["p_sum"] / minutes) if minutes > 0 else 0.0,
    }

# Aggregation
def _mean_sd(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, sd

def summarize(stats: List[Dict[str, float]]):
    def agg(key): return _mean_sd([s[key] for s in stats])
    return {
        "ep_return_mean": agg("episode_return")[0], "ep_return_sd": agg("episode_return")[1],
        "pay_mean":       agg("pay_total")[0],      "pay_sd":       agg("pay_total")[1],
        "onride_mean":    agg("onride_minutes")[0], "onride_sd":    agg("onride_minutes")[1],
        "stolen_mean":    agg("stolen_events")[0],  "stolen_sd":    agg("stolen_events")[1],
        "accept_mean":    agg("accepts")[0],        "accept_sd":    agg("accepts")[1],
        "stay_mean":      agg("stays")[0],          "stay_sd":      agg("stays")[1],
        "move_mean":      agg("moves")[0],          "move_sd":      agg("moves")[1],
        "unique_cells_mean": agg("unique_cells_count")[0], "unique_cells_sd": agg("unique_cells_count")[1],
        "avg_p_here_mean": agg("avg_p_here")[0],    "avg_p_here_sd": agg("avg_p_here")[1],
    }

def evaluate_scenarios_from_list(ckpt_path: str,
                                 policies_list: List[str],
                                 scenarios: List[Tuple[int, str, int, int, float, float]],
                                 gamma: float,
                                 top_k_moves: int,
                                 max_epochs: int,
                                 out_runs_csv: str,
                                 out_summary_csv: str,
                                 trace_first: int,
                                 trace_limit: int,
                                 trace_outdir: Optional[str],
                                 premium_threshold: float) -> None:

    V = None
    if "value_net" in policies_list:
        tmp_env = SingleDriverEnv(max_epochs=max_epochs)
        obs_dim = tmp_env.observation_space.shape[0]
        V = ValueNet(obs_dim)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        state = state.get("state_dict", state)
        V.load_state_dict(state); V.eval()

    def _premium_policy(env, **_): return policy_accept_if_premium(env, threshold=premium_threshold)
    def _topP_10(env, **_):        return policy_topP_accept_only(env, q=0.10)
    def _kstep(env, **kw):         return policy_kstep_greedy(env, gamma=gamma, top_k_moves=top_k_moves)
    def _rand(env, **kw):          return policy_random_uniform(env, top_k_moves=top_k_moves)

    policy_map: Dict[str, Callable] = {
        "value_net":          lambda env, **kw: policy_value_net(env, V=V, gamma=gamma, top_k_moves=top_k_moves),
        "always_accept":      policy_always_accept,
        "accept_else_bestP":  policy_accept_else_bestP,
        "accept_if_premium":  _premium_policy,
        "topP_accept_only":    _topP_10,
        "kstep_greedy":        _kstep,
        "random_uniform":      _rand
    }
    policies: List[Tuple[str, Callable]] = []
    for p in policies_list:
        if p not in policy_map:
            raise ValueError(f"Unknown policy: {p}")
        policies.append((p, policy_map[p]))

    run_header = ["policy","seed","date","start_hour","start_minute","start_lat","start_lng",
                  "minutes","hours","episode_return","reward_per_hour","pay_total","pay_per_hour",
                  "decisions","accepts","stays","moves","offer_boundaries","accept_rate_at_offer",
                  "avg_option_minutes","median_option_minutes","onride_minutes","stolen_events",
                  "unique_cells_count","avg_p_here"]
    os.makedirs(os.path.dirname(out_runs_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_summary_csv) or ".", exist_ok=True)
    write_runs_header = not os.path.exists(out_runs_csv) or os.path.getsize(out_runs_csv) == 0
    write_sum_header  = not os.path.exists(out_summary_csv) or os.path.getsize(out_summary_csv) == 0
    fr = open(out_runs_csv, "a", newline="")
    wr = csv.DictWriter(fr, fieldnames=run_header)
    if write_runs_header: wr.writeheader()

    sum_header = ["name","ep_return_mean","ep_return_sd","pay_mean","pay_sd",
                  "onride_mean","onride_sd","stolen_mean","stolen_sd",
                  "accept_mean","accept_sd","stay_mean","stay_sd",
                  "move_mean","move_sd","unique_cells_mean","unique_cells_sd",
                  "avg_p_here_mean","avg_p_here_sd"]
    fs = open(out_summary_csv, "a", newline="")
    ws = csv.DictWriter(fs, fieldnames=sum_header)
    if write_sum_header: ws.writeheader()

    try:
        for pname, pfn in policies:
            all_stats = []
            for idx, (seed, d_iso, sh, sm, lat, lng) in enumerate(scenarios, start=1):
                # parse date
                d = date.fromisoformat(d_iso)
                # prepare env for this scenario
                env = prepare_env(max_epochs=max_epochs,
                                  seed=seed,
                                  start_hour=sh,
                                  start_minute=sm,
                                  start_lat=lat,
                                  start_lng=lng,
                                  start_date=d)

                do_trace = (idx <= trace_first) and (trace_limit > 0)
                tag = f"_{pname}_s{idx}_{d}"

                ep = run_episode_on_env(env, pname, pfn, V=V, gamma=gamma, top_k_moves=top_k_moves,
                                        trace=do_trace, trace_limit=trace_limit,
                                        trace_outdir=trace_outdir, trace_tag=tag)
                all_stats.append(ep)

                wr.writerow({
                    "policy": pname,
                    "seed": int(seed),
                    "date": d_iso,
                    "start_hour": int(sh),
                    "start_minute": int(sm),
                    "start_lat": float(lat),
                    "start_lng": float(lng),
                    **{k: ep[k] for k in run_header if k in ep}
                })
                fr.flush()

                print(f"[{pname} #{idx}/{len(scenarios)}] seed={seed} "
                      f"date={d_iso} @{sh:02d}:{sm:02d} lat={lat:.5f} lng={lng:.5f}  "
                      f"pay=${ep['pay_total']:.2f}  pay/hr={ep['pay_per_hour']:.2f}  "
                      f"reward/hr={ep['reward_per_hour']:.2f}  acc@offer={ep['accept_rate_at_offer']:.2f}  "
                      f"decisions={ep['decisions']}  avg_k={ep['avg_option_minutes']:.1f}")

            summ = summarize(all_stats)
            ws.writerow({"name": pname, **summ})
            fs.flush()

            print("\n=== Aggregate:", pname, "===")
            for k in sum_header[1:]:
                print(f"{k}: {summ[k]:.4f}")
            print()
    finally:
        fr.close()
        fs.close()

def main():
    print("Generated SCENARIOS (seed, date, hour, minute, lat, lng):")
    for s in SCENARIOS:
        print(s)
    evaluate_scenarios_from_list(
        ckpt_path=CKPT_PATH,
        policies_list=POLICIES_TO_RUN,
        scenarios=SCENARIOS,
        gamma=GAMMA,
        top_k_moves=TOP_K_MOVES,
        max_epochs=MAX_EPOCHS,
        out_runs_csv=OUT_RUNS_CSV,
        out_summary_csv=OUT_SUMMARY_CSV,
        trace_first=TRACE_FIRST,
        trace_limit=TRACE_LIMIT,
        trace_outdir=TRACE_OUTDIR,
        premium_threshold=PREMIUM_THRESHOLD
    )

if __name__ == "__main__":
    main()
