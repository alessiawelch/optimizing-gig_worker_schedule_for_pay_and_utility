import random
import scipy.stats 
import collections
import os, warnings
from gym import Env, spaces
import copy
import numpy as np
import math
from single_driver_simulated_environment_new.config import (GRID_COL, GRID_ROW, HOURLY_PROBABILITIES, START_STATE, COST_PER_KM_USD, FIXED_COST_PER_MIN_USD, 
                    ACCEPT_RIDE, REJECT_MOVE, REJECT_STAY, CANCEL_PENALTY, MAX_RIDE_LIFETIME, START_HOUR, PRICE_FULL, PRICE_BACKUP, QUALITY_W, CHEAP_PENAL,
                    LAT_MAX, LON_MAX, LAT_MIN, LON_MIN, WINDOW_SIZE, MAX_RIDE_DURATION, DRIVER_ID, LOCATION_BONUS, NORMALIZE_COEFFICIENT,
                    START_DATE, AVG_DISTANCE_GRID_FACTOR, DISTANCE_FULL, DISTANCE_BACKUP, DEMAND_TABLE, MONEY_THRESHOLD_USD, IDLE_PENALTY, RATE_FULL, RATE_BACKUP)
from single_driver_simulated_environment_new.utils import grid_to_random_location, location_to_price_grid, hour_to_group, haversine_vec
from single_driver_simulated_environment_new.driver import Driver
import pandas as pd

warnings.simplefilter("always")
os.environ["PYTHONWARNINGS"] = "always"
OBS_SIZE =23
class SingleDriverEnv(Env):
    def __init__(self,
                 grid_row = GRID_ROW,
                 grid_col = GRID_COL,
                 max_epochs=600,                             
                 origin_prob_schedule=HOURLY_PROBABILITIES,
                 start_date=START_DATE,
                 start_loc=START_STATE):
        super().__init__()
        self.grid_row = grid_row                             # The row height of grid
        self.grid_col = grid_col                             # The column height of the grid
        self.max_epochs = max_epochs                         # The number of epochs to run (default is 10 hours)
        self.origin_prob_schedule = origin_prob_schedule     # For each grid the probability that a ride originates in this square                         
        self.start_date = pd.Timestamp(start_date)           # The current date for the simulator including start hour
        self.start_loc = start_loc                           # The start location of the driver
        # can be either 0, 1, and 2 + encoding of space to move 
        self.action_space = spaces.Discrete(2 + self.grid_row * self.grid_col)

        # observation space 
        self.window_size = WINDOW_SIZE
        # length of the observation space (not including window) includes:
        # current epoch; driver lat, lng; driver status; job timer;
        # idle period; offer origin; offer dest; offer_age; price premium
        # relative square info
        core_len = 14                       

        # length including the local probability window
        obs_len = core_len + self.window_size * self.window_size      

        # the lowest/highest values for each, defaults to 0 or 1
        low  = np.zeros(obs_len, dtype=np.float32)   
        high = np.ones (obs_len, dtype=np.float32)   

        # epoch setting
        high[0]  = self.max_epochs 

        # driver lat and lng
        low [1]  = LAT_MIN
        high[1]  = LAT_MAX
        low [2]  = LON_MIN
        high[2]  = LON_MAX

        # on job flag (either 0 or 1)
        high[3]  = 1   
        # job_timer                               
        high[4]  = MAX_RIDE_DURATION       
        # idle_period          
        high[5]  = self.max_epochs                   

        # offer origin lat / lng
        low [6]  = -1 # LAT_MIN
        high[6]  = LAT_MAX
        low [7]  = -1 # LON_MIN
        high[7]  = LON_MAX
        # offer dest   lat / lng
        low [8]  = -1 # LAT_MIN
        high[8]  = LAT_MAX
        low [9]  = -1 # LON_MIN
        high[9]  = LON_MAX
        # offer age
        low[10] = -1
        high[10] = self.max_epochs

        # price_premium (0 – 3)
        low [11] = 0.0        
        high[11] = 3.0

        # relative square info 
        low [12] = 0.0        
        high[12] = 1.0
        low [13] = 0.0        
        high[13] = 1.0


        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # internal state
        self.epoch = 0
        # defaults to driver starting in front of paddington station
        self.driver =  Driver(DRIVER_ID, self.start_loc, self.start_date) 
        self.ride_pool = []
        self.last_offer = None
        self.last_offer_price = 0
        self.rng = random.Random()
        self.np_rng = np.random.default_rng()

        self._origins_np = np.empty((0, 2), np.float32)
        self._rank_cache = None  

        self._dest_cache = collections.defaultdict(list)
        for (sr, sc, dr, dc, tg), data in DISTANCE_FULL.items():
            weight = (data["ride_count"] * data["rf_global"]) / data["median_dist_km"]
            self._dest_cache[(sr, sc, tg)].append((dr, dc, weight))


        for (sr, sc, tg), data in DISTANCE_BACKUP.items():
            key = (sr, sc, tg)
            if key not in self._dest_cache:          # no detailed stats for this origin/time-group
                weight = (data["ride_count"] * data["rf_global"]) / data["median_dist_km"]
                self._dest_cache[key] = [(sr, sc, weight)]   # destination is the same super-block

    # resets the environment to original state
    def reset(self):
        self.epoch = 0
        self.driver = Driver(DRIVER_ID, self.start_loc, self.start_date)
        self.ride_pool = []
        self.last_offer = None
        self._generate_epoch_rides() 
        self._update_rank_cache()
        return self._get_obs()
    
    # functions used in rebuilding SMDP
    def at_decision_boundary(self):
        if self.driver.on_job:
            return False
        # decide if reposition finished or an offer arrived mid-reposition
        return (not self.driver.repositioning) or (self.last_offer is not None)

    # save current state
    def _snapshot(self):
        return {
            "epoch": self.epoch,
            "driver": copy.deepcopy(self.driver),
            "ride_pool": list(self.ride_pool),
            "last_offer": self.last_offer,
            "rng_state": self.rng.getstate(),
            "py_rng": random.getstate(),
            "np_rng": np.random.get_state(),
        }

    # restore old state
    def _restore(self, snap):
        self.epoch = snap["epoch"]
        self.driver = snap["driver"]
        self.ride_pool = list(snap["ride_pool"])
        self.last_offer = snap["last_offer"]
        self.rng.setstate(snap["rng_state"])
        random.setstate(snap["py_rng"])
        np.random.set_state(snap["np_rng"])

        self._rebuild_origins_np()
        self._update_rank_cache()

    # test an option without affecting main environment
    def simulate_option(self, action_idx: int, gamma: float):
        # FAST ensures that OSRM isn't used to speed up calculations only used in 
        # simulations
        self.driver.FAST = True
        snap = self._snapshot()
        rewards, k = [], 0
        obs, r, done, _ = self.step(action_idx); rewards.append(r); k += 1
        while (not done) and (not self.at_decision_boundary()):
            _, r, done, _ = self.step(REJECT_STAY); rewards.append(r); k += 1

        # discounted sum
        g, R_disc = 1.0, 0.0
        for rr in rewards: R_disc += g * float(rr); g *= gamma

        self.driver.FAST = False
        s_next = self._get_obs()

        self._restore(snap)
        return R_disc, k, s_next, done


    def step(self, action):
        # try the main action and catch error so training can continue in worst case
        try:
            return self._step_core(action)  
        except Exception as ex:
            import traceback, os, sys
            pid = os.getpid()

            # catch any errors and log
            print(f"\n[env pid {pid}] Uncaught exception in step()", file=sys.stdout)
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()

            warnings.warn(
                f"[env pid {pid}] exception in step(): {ex.__class__.__name__}: {ex}",
                RuntimeWarning
            )

            # reset so training can continue
            obs    = np.zeros(OBS_SIZE, dtype=np.float32)
            reward = 0.0
            done   = True
            info   = {"error": str(ex)}
            return obs, reward, done, info
        
    # function that defines behavior for each time stamp
    def _step_core(self, actions):
        reward = FIXED_COST_PER_MIN_USD
        done = False
        stolen = False
        price_pred = 0

        self.driver.update()

        # gives fixed cost per minute that driver is changing locations
        if self.driver.repositioning:
            reward += self.driver.repositioning_km * COST_PER_KM_USD

        # offer only if idle and there are rides
        if not self.driver.on_job:
            offer = self._select_closest(self.driver.loc) if self.ride_pool else None
            action, mrow, mcol = self._translate_action(actions)

            self.last_offer = offer

            # generate information if ride is stolen, and store past price (-1 if it does not exist)
            if offer is not None:
                stolen_ride_outcome, pickup_info, ride_info, price_pred = self._stolen_ride(offer)
                self.last_offer_price = price_pred
            else:
                self.last_offer_price = -1
            
            if action == ACCEPT_RIDE:
                # if there is no offer, then default to idle
                if offer == None:
                    self.driver.idle_period += 1
                else:    
                    # chance that another driver accepts the ride
                    if(stolen_ride_outcome):
                        # remove offer and defult to idle period - moved lower down
                        self.driver.idle_period += 1
                        stolen = True       
                    else:
                        # start the job
                        self.driver.start_job(offer, self.epoch, pickup_info, ride_info, price_pred)
                        # make reward THE pay minus the cost per mile
                        _, duration, distance, pay, _ = self.driver.current_job
                        # update the pay of the driver
                        self.driver.pay += pay
                        #update the pay so that it is weighted more if less than the threshold
                        adjusted_pay = pay
                        if self.driver.pay <= MONEY_THRESHOLD_USD:
                            adjusted_pay = 4*pay # make it 4 times more valuable
                        
                        distance_cost = distance * COST_PER_KM_USD

                        reward = reward + adjusted_pay #+ distance_cost

                        origin_lat, origin_lng, dest_lat,  dest_lng, _ = offer
                        o_row, o_col = location_to_price_grid(origin_lat, origin_lng)
                        d_row, d_col = location_to_price_grid(dest_lat,  dest_lng)
                        hour_label   = hour_to_group(self._epoch_to_time()[0])

                        # lookup median fare for this origin destination and time
                        dur_min = max((duration / 60.0), 1e-6)
                        pred_rate = price_pred / dur_min
                        median_rate = self._median_rate(o_row, o_col, d_row, d_col, hour_label)
                        if median_rate:                          
                            premium = np.clip(pred_rate / median_rate - 1.0, -1.0, 1.0)  
                            self.driver.current_premium = premium      
                            if premium >= 0.25: 
                                reward += QUALITY_W * premium  
                            elif premium > 0:
                                reward += 5.0 * premium
                            else:
                                reward += CHEAP_PENAL * premium 

            if action == REJECT_STAY:
                self.driver.idle_period += 1
                    
            elif action == REJECT_MOVE:
                if (mrow, mcol) == self.driver.grid:
                    self.driver.idle_period += 1
                else:
                    # give penalty for changing mind to a diff location
                    if (mrow, mcol) != self.driver.repositioning_loc:
                        if self.driver.repositioning:
                            reward += CANCEL_PENALTY
                            self.driver.route_queue.clear() 
                        dlat, dlng = grid_to_random_location(mrow, mcol, rng=self.np_rng)
                        self.driver.reject_move_action(dlat, dlng)
                        reward += self.driver.repositioning_km * COST_PER_KM_USD
                    self.driver.idle_period += 1
            
            if offer is not None:
                #remove offer from available offers
                self.ride_pool.remove(offer)
                self._rebuild_origins_np() 

        else:
            # give a small reward while on ride
            reward += self.driver.current_premium 
            offer = None
            action = None

        P_here = self._current_prob()[self.driver.grid]

        # give bonus for being in a good square
        if not self.driver.on_job:
            reward += LOCATION_BONUS * P_here * 100 

        idle_penalty = IDLE_PENALTY * self.driver.idle_period
        reward += idle_penalty

        # needed to supress warnings
        if isinstance(reward, np.ndarray):
            reward = reward.item()
        else:
            reward = float(reward)         

        self.driver.rewards += reward
        # advance number of epochs
        self.epoch += 1
        self._update_rank_cache() 

        # remove rides that have not been accepted within 5 epochs (5 minutes)
        cutoff = self.epoch - MAX_RIDE_LIFETIME
        self.ride_pool = [r for r in self.ride_pool if r[4] > cutoff]
        self._rebuild_origins_np()

        # check if we have reached the max number of epochs
        done = (self.epoch >= self.max_epochs)
        if not done:
            self._generate_epoch_rides()

        # info for us to check
        info = {
            "on_job":      self.driver.on_job,
            "job_timer":   self.driver.job_timer,
            "offer":       offer,           
            "action":      actions, 
            "stolen":      stolen,         
            "driver_loc":  self.driver.loc,
            "epoch":       self.epoch,
            "ride_pool":   len(self.ride_pool), 
            "total_pay":   self.driver.pay,
            "P_here":      P_here,
            "offer_price": price_pred
        } 

        return self._get_obs(), reward, done, info 
    
    # refresh cached (N,2) array of offer origins for fast vectorized nearest-origin lookups
    # keep it empty when pool is empty to skip work cleanly
    def _rebuild_origins_np(self):                       
        if self.ride_pool:
            self._origins_np = np.asarray(
                self.ride_pool, dtype=np.float32)[:, :2]  # (N,2)
        else:
            self._origins_np = np.empty((0, 2), np.float32)

    # cache per-cell demand percentile from current hour’s prob grid
    def _update_rank_cache(self):
        P_flat = self._current_prob().ravel()
        self._rank_cache = scipy.stats.rankdata(P_flat) / P_flat.size

    # find closest ride to the driver by comparing duration to pickup spot
    def _select_closest(self, loc):
        if self._origins_np.size == 0:
            return None
        lat, lng = map(float, loc)

        # use haversine vector for this calculations since its way faster
        dists = haversine_vec(lat, lng,
                            self._origins_np[:, 0],
                            self._origins_np[:, 1])
        # needs to be within 3 km of the 
        mask = dists <= 3.0
        if not np.any(mask):
            return None

        best = int(np.argmin(dists[mask]))
        return self.ride_pool[np.flatnonzero(mask)[best]]
    
    # returns the probability matrix for that hour
    def _current_prob(self):
        hour, _ = self._epoch_to_time()
        return self.origin_prob_schedule[hour]

    # converts the epoch to the hour of the day
    def _epoch_to_time(self):
        return int(self.driver.current_time.hour), int(self.driver.current_time.minute)

    # returns action in expected format
    def _translate_action(self, idx):
        if idx == 0:  return (ACCEPT_RIDE, 0, 0)             
        if idx == 1:  return (REJECT_STAY, 0, 0)  
        adj = idx - 2
        row, col = divmod(adj, self.grid_col)
        return (REJECT_MOVE, row, col) 
    
    # determines median fare based on price row and grid
    def _median_fare(self, o_row, o_col, d_row, d_col, tg):
        full_key   = (o_row, o_col, d_row, d_col, tg)
        origin_key = (o_row, o_col, tg)
        if full_key in PRICE_FULL:
            return PRICE_FULL[full_key]
        return PRICE_BACKUP.get(origin_key, None) 
    
    # determines the median rate at that hour and grid squares
    def _median_rate(self, o_row, o_col, d_row, d_col, tg):
        full_key   = (o_row, o_col, d_row, d_col, tg)
        origin_key = (o_row, o_col, tg)

        if full_key in RATE_FULL:
            return RATE_FULL[full_key]

        return RATE_BACKUP.get(origin_key, None)
    
    def _dest_candidates(self, orow: int, ocol: int, tg: str):
        max_super_row = math.ceil(self.grid_row / AVG_DISTANCE_GRID_FACTOR)
        max_super_col = math.ceil(self.grid_col / AVG_DISTANCE_GRID_FACTOR)

        sr = min(orow // AVG_DISTANCE_GRID_FACTOR, max_super_row - 1)
        sc = min(ocol // AVG_DISTANCE_GRID_FACTOR, max_super_col - 1)
        return self._dest_cache.get((sr, sc, tg), [])
      
    # create rides based on probabilities
    def _stolen_ride(self, offer):
        # obtain predicted price
        pickup_info, ride_info, price_pred = self.driver.job_information(offer)

        # start and end point of the ride offered
        origin_lat, origin_lng, dest_lat, dest_lng, offer_epoch = offer
        hour, _ = self._epoch_to_time()

        # create columns for look up
        o_row, o_col = location_to_price_grid(origin_lat, origin_lng)
        d_row, d_col = location_to_price_grid(dest_lat, dest_lng)
        hour_label = hour_to_group(hour)

        median_fare = self._median_fare(o_row, o_col, d_row, d_col, hour_label)

        # if the median fare is not calculated default to not stolen
        if median_fare is None:          
            return False, pickup_info, ride_info, price_pred
        
        # if the predicted price is less than median rate there is a 5% chance that a ride is stolen
        if price_pred < median_fare:
            p_steal = .05

        # else there is higher probability the greater above the median
        else:
            premium = max(0.0, price_pred / median_fare - 1.0)   
            z = premium - NORMALIZE_COEFFICIENT
            p_steal = 1.0 / (1.0 + math.exp(-z))

        stolen_ride_outcome = self.rng.random() < p_steal
        return stolen_ride_outcome, pickup_info, ride_info, price_pred
    
    # generates rides based on demand per grid
    def _generate_epoch_rides(self):
        P      = self._current_prob()
        hour, _ = self._epoch_to_time() 
        demand = DEMAND_TABLE[hour]
        num_rides = int((demand * self.grid_col * self.grid_row) / 10)

        if num_rides <= 0:
            return

        # flatten AND normalize 
        p = P.ravel().astype(float)
        p_sum = p.sum()
        if not np.isfinite(p_sum) or p_sum <= 0:
            p = np.full_like(p, 1.0 / p.size, dtype=float)
        else:
            p /= p_sum

        # counts per cell, then expand to repeated indices
        counts = self.np_rng.multinomial(num_rides, p)           
        flat_ix = np.repeat(np.arange(p.size), counts)        
        self.np_rng.shuffle(flat_ix)                             
     

        tg = hour_to_group(hour)

        for idx in flat_ix:
            orow, ocol = divmod(idx, self.grid_col)

            # try to pick a super-block from learned distances
            cands = self._dest_candidates(orow, ocol, tg)
            if cands:
                supers, weights = zip(*[((sr, sc), w) for sr, sc, w in cands])
                sr, sc = random.choices(supers, weights=weights, k=1)[0]

                # super to raw cell window
                raw_min_r = sr * AVG_DISTANCE_GRID_FACTOR
                low_r     = max(0, min(raw_min_r, self.grid_row - 1))
                high_r    = min(low_r + AVG_DISTANCE_GRID_FACTOR,
                                self.grid_row)
                raw_min_c = sc * AVG_DISTANCE_GRID_FACTOR
                low_c     = max(0, min(raw_min_c, self.grid_col - 1))
                high_c    = min(low_c + AVG_DISTANCE_GRID_FACTOR,
                                self.grid_col)

                drow = self.rng.randrange(low_r, high_r)
                dcol = self.rng.randrange(low_c, high_c)

            else:
                # no stats pick one of the 4 adjacent neighbors
                neighbors = []
                for dr_off, dc_off in ((1,0),(-1,0),(0,1),(0,-1)):
                    rr, cc = orow + dr_off, ocol + dc_off
                    if 0 <= rr < self.grid_row and 0 <= cc < self.grid_col:
                        neighbors.append((rr, cc))

                if neighbors:
                    drow, dcol = self.rng.choice(neighbors)
                else:
                    drow, dcol = orow, ocol

         
            start_lat, start_lng = grid_to_random_location(orow, ocol, rng=self.np_rng)
            end_lat,   end_lng   = grid_to_random_location(drow, dcol, rng=self.np_rng)
            self.ride_pool.append((start_lat, start_lng,
                                   end_lat,   end_lng,
                                   self.epoch))
       
        POOL_CAP = 1500
        if len(self.ride_pool) > POOL_CAP:
            lat, lng = self.driver.loc
            arr   = np.asarray(self.ride_pool, dtype=np.float32)      
            dists = haversine_vec(lat, lng, arr[:, 0], arr[:, 1])
            keep  = np.argsort(dists)[:POOL_CAP]                       
            self.ride_pool   = [self.ride_pool[i] for i in keep]
            self._origins_np = arr[keep, :2]                         
        
        self._rebuild_origins_np()   

    
    def _local_prob_window(self, k = WINDOW_SIZE):
        half = k // 2
        P    = self._current_prob()
        dr, dc = self.driver.grid
        window = np.zeros((k, k), dtype=np.float32)
        for i in range(-half, half + 1):
            for j in range(-half, half + 1):
                r, c = dr + i, dc + j
                if 0 <= r < self.grid_row and 0 <= c < self.grid_col:
                    window[i + half, j + half] = P[r, c]
        return window

    # returns the observation
    def _get_obs(self):
        if self.last_offer and not self.driver.on_job:
            olat, olng, dlat, dlng, off_e = self.last_offer
            pred_price = self.last_offer_price
            
            hour, _ = self._epoch_to_time()

            o_row, o_col = location_to_price_grid(olat, olng)
            d_row, d_col = location_to_price_grid(dlat, dlng)
            tg           = hour_to_group(hour)

            median = self._median_fare(o_row, o_col, d_row, d_col, tg)
        else:
            olat = olng = dlat = dlng = off_e = -1.0
            pred_price = -1.0
            median = 1

        driver_lat, driver_lng  = self.driver.loc
        on_job  = 1.0 if self.driver.on_job else 0.0

        if (median is None) or (not np.isfinite(median)) or (median <= 0) or (pred_price is None) or (pred_price < 0):
            premium = 0.0
        else:
            premium = float(max(0.0, float(pred_price) / float(median) - 1.0))


        core = np.array([
            self.epoch,
            driver_lat, driver_lng,
            on_job,
            self.driver.job_timer,
            self.driver.idle_period,
            olat, olng, dlat, dlng, off_e, premium
        ], dtype=np.float32)

        if self.driver.FAST:
            P = self._current_prob()
            global_demand = np.sum(P) / (self.grid_row * self.grid_col)
            rank_of_cell = 0.5  
            global_feats = np.array([global_demand, rank_of_cell], dtype=np.float32)
            local_prob = np.zeros(self.window_size * self.window_size, dtype=np.float32)
            return np.concatenate([core, global_feats, local_prob], dtype=np.float32)

        if self._rank_cache is None: self._update_rank_cache()
        P = self._current_prob()
        global_demand = np.sum(P) / (self.grid_row * self.grid_col)                    
        rank_of_cell = self._rank_cache[self.driver.grid[0] * self.grid_col
                                + self.driver.grid[1]]
        global_feats  = np.array([global_demand, rank_of_cell], dtype=np.float32)

        local_prob = self._local_prob_window(self.window_size).astype(np.float32).ravel()


        obs = np.concatenate([core, global_feats, local_prob], dtype=np.float32)
        pid = os.getpid()
        if obs.shape != (OBS_SIZE,):
            print(f"[pid {pid}] *** BAD SHAPE {obs.shape} ***", flush=True)
        if obs.dtype != np.float32:
            print(f"[pid {pid}] *** BAD DTYPE {obs.dtype} ***", flush=True)
        return obs

    def seed(self, seed=None):
        if seed is None:
            seed = int(np.random.SeedSequence().entropy) & 0xFFFFFFFF
        seed = int(seed) & 0xFFFFFFFF
        if not hasattr(self, "rng"):
            self.rng = random.Random()
        self.rng.seed(seed)
        random.seed(seed)
        np.random.seed(seed)             
        self.np_rng = np.random.default_rng(seed)  
        return [seed]