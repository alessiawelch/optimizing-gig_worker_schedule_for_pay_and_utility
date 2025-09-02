import numpy as np
from pathlib import Path
from pandas import Timestamp
import pandas as pd

# constants for actions
ACCEPT_RIDE = 0
REJECT_STAY = 1
REJECT_MOVE = 2

# pound to dollar exchange rate
EXCHANGE_RATE = 1.36

# assumes rental, valet, phone and data, licensing, user charges 
# assumes working 10 hr/day, 6 days a week
FIXED_COST_PER_MIN_POUNDS = -0.093904915
FIXED_COST_PER_MIN_USD = FIXED_COST_PER_MIN_POUNDS * EXCHANGE_RATE

# assumes cost of 273 pounds for 1000 km for EV maintance cost
COST_PER_KM_POUNDS = -0.169877771
COST_PER_KM_USD = COST_PER_KM_POUNDS * EXCHANGE_RATE

# max and min lat and longitude in dataset
LAT_MIN, LAT_MAX = 51.4, 51.58          
LON_MIN, LON_MAX = -0.25,  0.07 

# constants used to determine grid row and grid column size
CELL_DEG = 0.0075
GRID_ROW = int(np.ceil((LAT_MAX - LAT_MIN) / CELL_DEG))       # 24
GRID_COL = int(np.ceil((LON_MAX - LON_MIN) / CELL_DEG))       # 43

# date and time which the simulation is taking space
START_DATE = Timestamp("2025-06-01 15:00") 
START_HOUR = START_DATE.hour

# start location and IDfor the driver
START_STATE = (51.515928, -0.174977)  # outside paddington station
DRIVER_ID = "random_id"

# max length that rides are around for before being removed from ride pool
MAX_RIDE_LIFETIME = 5

# constant number of months in year
MONTHS_IN_YEAR = 12

# number of seconds in an epoch (set at 1 epoch = 1 minute = 60 seconds)
SECONDS_PER_EPOCH = 60

# longest amount of time a ride can last, max set at 10 hours (length of episode)
MAX_RIDE_DURATION = 600

# size of the local observation space that the driver can learn from (9 total cells)
WINDOW_SIZE = 3

# bonus for being in a good location
LOCATION_BONUS = 0.5

# penalty for changing mind in direction (to discourage constant flip flopping)
CANCEL_PENALTY = -1

# penalty for changing mind in direction (to discourage constant flip flopping)
IDLE_PENALTY = -1

# price thershold 
# calculated that need 611 to break even and minimum living wage is 13.85 pounds and 60 hour weeks
# needed income per week is 1442
MONEY_THRESHOLD_POUNDS = 240
MONEY_THRESHOLD_USD = MONEY_THRESHOLD_POUNDS * EXCHANGE_RATE

# used to determine premiums
QUALITY_W   = 20.0    
CHEAP_PENAL = 5.0 

# link needed for OSRM call
OSRM_FMT    = "http://localhost:5000/route/v1/driving/{lon1:.6f},{lat1:.6f};{lon2:.6f},{lat2:.6f}?overview=full&geometries=geojson"

# READING PROBABILITY FILES

# obtaining correct path
THIS_DIR  = Path(__file__).resolve().parent         
PROJECT   = THIS_DIR.parent.parent                  
DATA_DIR  = PROJECT / "driver_data" / "combined_path" / "new_test" / "original"
CELL_FILE_ADDITION = "original_0075_v2"

# used to determine which cells should get the most rides
#PROB_FILE = DATA_DIR / f"hourly_origin_grids_hotspots_{CELL_FILE_ADDITION}_v3.npy" # artificial hotspots
PROB_FILE = DATA_DIR / f"hourly_origin_prob_grid_{CELL_FILE_ADDITION}_no_epsilon.npy" # real data
HOURLY_PROBABILITIES = np.load(PROB_FILE, allow_pickle=False)

# used to find median price from one general area to another, used to determine the probability
# that a ride should be stolen, depending on how far predicted price is from median
AVG_PRICE_FILE = DATA_DIR / f"origin_dest_hour_lookup_price_{CELL_FILE_ADDITION}.parquet"
AVG_PRICE = pd.read_parquet(AVG_PRICE_FILE)
AVG_PRICE_GRID_FACTOR = 6
PRICE_FULL = {                     
            (int(r.origin_super_row), int(r.origin_super_col),
             int(r.dest_super_row),   int(r.dest_super_col),
             r.time_group): float(r.median_price_usd)
            for r in AVG_PRICE[~AVG_PRICE["dest_super_row"].isna()]
              .itertuples(index=False)
        }
PRICE_BACKUP = {  
            (int(r.origin_super_row), int(r.origin_super_col),
             r.time_group): float(r.median_price_usd)
            for r in AVG_PRICE[AVG_PRICE["dest_super_row"].isna()]
              .itertuples(index=False)
        }
NORMALIZE_COEFFICIENT = 0.2

# used to find median distance from one general area to another, used to determine 
AVG_DISTANCE_FILE = DATA_DIR / f"origin_dest_hour_lookup_distance_{CELL_FILE_ADDITION}.parquet"
AVG_DISTANCE = pd.read_parquet(AVG_DISTANCE_FILE)
AVG_DISTANCE_GRID_FACTOR = 6
DISTANCE_FULL = {
        (
            int(r.origin_super_row),
            int(r.origin_super_col),
            int(r.dest_super_row),
            int(r.dest_super_col),
            r.time_group
        ): {
            "median_dist_km": float(r.median_dist_km),
            "ride_count":     int(r.ride_count),
            "rf_global":      float(r.rf_global)
        }
            for r in AVG_DISTANCE[~AVG_DISTANCE["dest_super_row"].isna()]
              .itertuples(index=False)
        }
DISTANCE_BACKUP = {
        (
            int(r.origin_super_row),
            int(r.origin_super_col),
            r.time_group
        ): {
            "median_dist_km": float(r.median_dist_km),
            "ride_count":     int(r.ride_count),
            "rf_global":      float(r.rf_global)
        }
            for r in AVG_DISTANCE[AVG_DISTANCE["dest_super_row"].isna()]
              .itertuples(index=False)
        }
NORMALIZE_COEFFICIENT = 0.2

# used for demand rides
DEMAND_FILE = DATA_DIR / f"hour_lookup_demand_{CELL_FILE_ADDITION}.parquet"
DEMAND = pd.read_parquet(DEMAND_FILE)
DEMAND_TABLE = {                     
            int(r.hour): float(r.rf_global_adjusted)
            for r in DEMAND
              .itertuples(index=False)
        }

AVG_PRICE_RATE = DATA_DIR / f"origin_dest_hour_lookup_rate_{CELL_FILE_ADDITION}.parquet"
AVG_RATE = pd.read_parquet(AVG_PRICE_RATE)
AVG_RATE_GRID_FACTOR = 6
RATE_FULL = {                     
            (int(r.origin_super_row), int(r.origin_super_col),
             int(r.dest_super_row),   int(r.dest_super_col),
             r.time_group): float(r.median_rate_usd)
            for r in AVG_RATE[~AVG_RATE["dest_super_row"].isna()]
              .itertuples(index=False)
        }
RATE_BACKUP = {  
            (int(r.origin_super_row), int(r.origin_super_col),
             r.time_group): float(r.median_rate_usd)
            for r in AVG_RATE[AVG_RATE["dest_super_row"].isna()]
              .itertuples(index=False)
        }

