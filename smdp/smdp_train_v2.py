import copy, random, time, os, math
from typing import List
import numpy as np
import torch
import torch.nn.functional as F

# progress bars
try:
    from tqdm.auto import tqdm
except Exception:
    class _NoTQDM:
        def __init__(self, total=None, desc=None, **kw): self.n=0
        def update(self, n=1): self.n += n
        def set_postfix(self, **kw): pass
        def close(self): pass
    def tqdm(*a, **k): return _NoTQDM()

from single_driver_simulated_environment_new.simulated_env import SingleDriverEnv
from single_driver_simulated_environment_new.config import ACCEPT_RIDE, REJECT_STAY
from smdp.value_net import ValueNet
from smdp.planner import td_target

def at_decision_boundary(env) -> bool:
    # if env doesn't have a driver yet, treat as boundary
    if getattr(env, "driver", None) is None:
        return True
    # while on a job we are inside an option (not a boundary)
    if env.driver.on_job:
        return False
    return (not getattr(env.driver, "repositioning", False)) or (env.last_offer is not None)

TICK = REJECT_STAY  # 1-minute time advance while an option unfolds

def move_action_index(env, row, col) -> int:
    return 2 + row * env.grid_col + col

def discounted_sum(rewards, gamma: float) -> float:
    # sum_t gamma^t * r_t  
    s, g = 0.0, 1.0
    for r in rewards:
        s += g * float(r); g *= gamma
    return s

def roll_option(env, first_action_idx: int, gamma: float):
    rewards, k = [], 0
    obs, r, done, info = env.step(first_action_idx)
    rewards.append(r); k += 1
    if done:
        return discounted_sum(rewards, gamma), k, obs, True
    # continue 1-min ticks until we hit a boundary again
    while (not done) and (not at_decision_boundary(env)):
        obs, r, done, info = env.step(TICK)
        rewards.append(r); k += 1
    R_disc = discounted_sum(rewards, gamma)
    return R_disc, k, obs, done

def enumerate_options(env, top_k_moves: int = 12) -> List[int]:
    opts = [1]  # STAY
    if (env.last_offer is not None) and (not env.driver.on_job):
        opts.append(0)  # ACCEPT
    # choose MOVE targets by descending probability in the hourly demand map
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

def try_option(env, a, gamma):
    if hasattr(env, "simulate_option"):
        # simulate_option must follow the same contract as roll_option
        return env.simulate_option(a, gamma) 
    fork = copy.deepcopy(env)
    return roll_option(fork, a, gamma)

# roll, batch, and compute SMDP targets
@torch.no_grad()
def smdp_scores_for_state(env, V, gamma: float, options: List[int]):
    records = []
    for a in options:
        R_disc, k, s_next, done = try_option(env, a, gamma)
        records.append((R_disc, k, s_next, done))

    # batch evaluate V on non-terminal next states
    idx_live = [i for i, rec in enumerate(records) if not rec[3]]
    v_next_batch = np.zeros(len(records), dtype=np.float32)
    if idx_live:
        s_batch = torch.as_tensor(np.stack([records[i][2] for i in idx_live]),
                                  dtype=torch.float32)
        v_batch = V(s_batch).squeeze(-1).cpu().numpy()
        for j, i in enumerate(idx_live):
            v_next_batch[i] = float(v_batch[j])

    # SMDP TD targets
    scores = []
    for i, (R_disc, k, _, done) in enumerate(records):
        v_next = 0.0 if done else float(v_next_batch[i])
        scores.append(td_target(R_disc, k, v_next, gamma))

    scores = np.asarray(scores, dtype=np.float32)
    return scores, options[int(scores.argmax())]

def soft_pick(options: List[int], scores: List[float], temp: float) -> int:
    z = np.array(scores, dtype=np.float64) / max(1e-6, temp)
    z = z - z.max()         
    p = np.exp(z); p = p / p.sum()
    return int(np.random.choice(options, p=p))

def train_value_iteration(
    steps_per_iter: int = 1200,   # number of boundaries to collect per sweep
    iters: int = 50,              # fitted-iteration sweeps
    train_epochs: int = 2,        # SGD epochs per sweep
    batch_size: int = 256,
    gamma: float = 0.995,         # per-minute discount
    top_k_moves: int = 8,         # MOVE option set size
    temp_start: float = 1.2,      # exploration temperature
    temp_end: float = 0.4,
    max_epochs: int = 600,        # min per episode
    max_wall_time_hours: float = 8.0,
    checkpoint_path: str = "smdp_value_net_old_data.pt",
    checkpoint_every_sec: int = 600
):
    # simple deadline guard
    t0 = time.time()
    deadline = t0 + max_wall_time_hours * 3600.0
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    torch.set_num_threads(1)
    device = torch.device("cpu")

    # env and model
    env = SingleDriverEnv(max_epochs=max_epochs)
    obs = env.reset()

    obs_dim = env.observation_space.shape[0]
    V = ValueNet(obs_dim).to(device)
    opt = torch.optim.Adam(V.parameters(), lr=3e-4)

    # linear temp schedule across iterations
    def temperature(ii):
        if iters <= 1: return temp_end
        t = ii / (iters - 1)
        return temp_start + (temp_end - temp_start) * t

    last_ckpt = t0
    total_boundaries = 0

    # periodic checkpoint with a tiny training diary
    def maybe_checkpoint(ii, total_boundaries):
        nonlocal last_ckpt
        now = time.time()
        if now - last_ckpt >= checkpoint_every_sec:
            torch.save({
                "state_dict": V.state_dict(),
                "iter": ii,
                "boundaries": total_boundaries,
                "elapsed_sec": now - t0,
                "config": {
                    "steps_per_iter": steps_per_iter, "iters": iters,
                    "train_epochs": train_epochs, "batch_size": batch_size,
                    "gamma": gamma, "top_k_moves": top_k_moves,
                    "max_epochs": max_epochs
                }
            }, checkpoint_path)
            last_ckpt = now

    # PROGRESS BAR 
    iter_bar = tqdm(total=iters, desc="Iterations", leave=True)
    try:
        for ii in range(iters):
            if time.time() >= deadline:
                break
            tau = temperature(ii)
            X, Y = [], []         

            obs = env.reset()
            done = False

            # boundaries collection bar
            collect_bar = tqdm(total=steps_per_iter, desc=f"Collect [{ii+1}/{iters}]", leave=False)

            while len(Y) < steps_per_iter:
                if time.time() >= deadline:
                    break
                if done:
                    obs = env.reset()
                    done = False

                # if not at decision point, tick forward
                if not at_decision_boundary(env):
                    obs, r, done, info = env.step(TICK)
                    continue

                # snapshot state s and enumerate options
                s = obs.copy()
                options = enumerate_options(env, top_k_moves=top_k_moves)

                # compute option scores and best y*
                scores, a_star = smdp_scores_for_state(env, V, gamma, options)
                y_star = float(scores.max())
                X.append(s); Y.append(y_star)

                # sample an option to move the env forward (explore with temp tau)
                a_sample = soft_pick(options, scores.tolist(), tau)
                _, _, obs, done = roll_option(env, a_sample, gamma)
                total_boundaries += 1

                collect_bar.update(1)
                # show rolling ETA to deadline
                eta_h = max(0.0, (deadline - time.time())/3600.0)
                collect_bar.set_postfix(boundaries=total_boundaries, eta_h=f"{eta_h:0.2f}")

                maybe_checkpoint(ii, total_boundaries)

            collect_bar.close()

            if len(Y) == 0 or time.time() >= deadline:
                break

            X_t = torch.as_tensor(np.stack(X), dtype=torch.float32, device=device)
            Y_t = torch.as_tensor(np.array(Y, dtype=np.float32), dtype=torch.float32, device=device)

            num_batches = math.ceil(X_t.size(0) / batch_size) * max(1, train_epochs)
            train_bar = tqdm(total=num_batches, desc=f"Train  [{ii+1}/{iters}]", leave=False)

            for _ in range(train_epochs):
                if time.time() >= deadline:
                    break
                idx = torch.randperm(X_t.size(0), device=device)
                for j in range(0, X_t.size(0), batch_size):
                    if time.time() >= deadline:
                        break
                    sel = idx[j:j+batch_size]
                    v_pred = V(X_t[sel]).squeeze(-1)
                    y_tar = Y_t[sel]
                    loss = F.mse_loss(v_pred, y_tar)

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(V.parameters(), 5.0)
                    opt.step()

                    train_bar.update(1)
                    eta_h = max(0.0, (deadline - time.time())/3600.0)
                    train_bar.set_postfix(loss=float(loss.item()), eta_h=f"{eta_h:0.2f}")

                maybe_checkpoint(ii, total_boundaries)

            train_bar.close()

            # outer progress
            iter_bar.update(1)
            iter_bar.set_postfix(total_boundaries=total_boundaries,
                                 elapsed_min=f"{(time.time()-t0)/60.0:0.1f}",
                                 eta_h=f"{max(0.0,(deadline-time.time())/3600.0):0.2f}")

            if time.time() >= deadline:
                break

    finally:
        iter_bar.close()
        # final save with weights
        torch.save({"state_dict": V.state_dict()}, checkpoint_path)
        print(f"Saved final weights to {checkpoint_path}")

    print("Training complete.")
    return V

if __name__ == "__main__":
    # to run iterations
    train_value_iteration(
        steps_per_iter=1200,
        iters=50,
        train_epochs=2,
        batch_size=256,
        gamma=0.995,
        top_k_moves=8,
        max_epochs=600,             
        max_wall_time_hours=9.0,
        checkpoint_path="smdp_value_net_v2.pt",
        checkpoint_every_sec=600
    )
