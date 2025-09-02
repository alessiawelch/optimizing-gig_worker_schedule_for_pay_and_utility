import torch

def td_target(R, k, v_next, gamma):
    # R is already discounted within the option: R = sum_{i=0}^{k-1} gamma^i r_i
    num = (gamma ** k) - 1.0
    den = k * (gamma - 1.0) if k > 0 else 1.0
    immediate = R * (num / den if den != 0 else 1.0)
    return immediate + (gamma ** k) * v_next

@torch.no_grad()
def score_option(adapter, V, obs_tensor, R, k, next_obs):
    v_next = V(torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0)).item()
    return td_target(R, k, v_next, adapter.gamma)

def choose_option(adapter, V, obs, epsilon=0.05):
    import random, numpy as np
    options = adapter.enumerate_options(obs)
    if not options:
        return None
    if random.random() < epsilon:
        return random.choice(options)
    best, best_u = None, -1e18
    for o in options:
        R, k, s_next, done, _ = adapter.simulate_option(obs, o)
        u = score_option(adapter, V, obs, R, k, s_next)
        if u > best_u:
            best, best_u = o, u
    return best
