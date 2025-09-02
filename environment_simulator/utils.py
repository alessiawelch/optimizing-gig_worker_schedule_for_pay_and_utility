
from geopy import distance              
import requests, time
import random
import numpy as np
from single_driver_simulated_environment_new.config import CELL_DEG, LAT_MIN, LON_MIN, OSRM_FMT, AVG_PRICE_GRID_FACTOR, GRID_ROW, GRID_COL


# takes in lng and lat pair
def calc_distance(location_1, location_2):
    return abs(distance.distance(location_1, location_2).km)

# Helper to translate LNG and LAT to Grid location
def location_to_grid(lat, lng):
    row = int((lat - LAT_MIN) // CELL_DEG)
    col = int((lng - LON_MIN) // CELL_DEG)

    row = max(0, min(row, GRID_ROW - 1))
    col = max(0, min(col, GRID_COL - 1))
    return row, col

# Helper to create random lat and lng within a grid square
def grid_to_random_location(row, col, rng):
    lat_min_cell = LAT_MIN + row * CELL_DEG
    lat_max_cell = lat_min_cell + CELL_DEG

    lng_min_cell = LON_MIN + col * CELL_DEG
    lng_max_cell = lng_min_cell + CELL_DEG

    lat = float(rng.uniform(lat_min_cell, lat_max_cell))
    lng = float(rng.uniform(lng_min_cell, lng_max_cell))

    return lat, lng

def location_to_price_grid(lat, lng):
    row, col = location_to_grid(lat, lng)
    row = row // AVG_PRICE_GRID_FACTOR
    col = col // AVG_PRICE_GRID_FACTOR
    return row, col

# calculate hour to grid
def hour_to_group(h: int) -> str:
    if   6 <= h < 10:       return "early_morning"
    elif 10 <= h < 15:      return "mid_day"
    elif 15 <= h < 18:      return "afternoon"
    elif 18 <= h < 22:      return "evening"
    elif h >= 22 or h < 2:  return "night"
    else:                   return "late_night"    

def osrm_call(olat, olng, dlat, dlng, retries=3):
    url = OSRM_FMT.format(lon1=olng, lat1=olat,
                          lon2=dlng,   lat2=dlat)
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()

            route      = r.json()["routes"][0]
            sec        = route["duration"]            
            km         = route["distance"] / 1000      
            # OSRM returns (lon, lat)
            polyline   = [(lat, lon) for lon, lat
                          in route["geometry"]["coordinates"]]

            return sec, km, polyline, False
        except Exception:
            time.sleep(0.5)                          
    
    #if osrm_km is not able to be found, then use calculated distance
    backup_km = haversine_vec(olat, olng, dlat, dlng)

    # if osrm_sec is not able to be found, engineer the seconds 
    backup_sec = backup_km * 60 
    return backup_sec, backup_km, [], True

# Helper for Haversine Distance
def haversine_vec(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))  
    return R*c