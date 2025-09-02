
from single_driver_simulated_environment_new.ml_models import predict_distance, predict_duration_ratio, predict_price             
import pandas as pd
from single_driver_simulated_environment_new.config import MONTHS_IN_YEAR, SECONDS_PER_EPOCH
from single_driver_simulated_environment_new.utils import location_to_grid, osrm_call, haversine_vec
import math
from collections import deque


# Driver class that contains information for each drivers
class Driver:
    def __init__(self, driver_id, start_loc, start_date):
        self.id = driver_id                                      # identifier of the driver 
        self.loc = start_loc                                     # current location of driver at start of epoch (lat, lng)
        self.grid = location_to_grid(start_loc[0], start_loc[1]) # current location of driver in grid spots
        self.on_job = False                                      # true if on job, idle if not
        self.job_timer = 0                                       # how much longer the job will last in minutes
        self.current_job = None                                  # what is the current job the driver is on includes job, distance, duration, price, epoch
        self.idle_period = 0                                     # how many epochs the current driver has not been on a job
        self.past_locations = []                                 # past locations the driver has been (not intermediary positions)
        self.rewards = 0                                         # cumulative rewards
        self.pay = 0                                             # the monetary reward earned
        self.current_time = pd.Timestamp(start_date)             # current real time
        self.route_queue = deque()                               # ride segements that include the polyline, seconds left, end location
        self.repositioning = False                               # if in the middle of a repositioning
        self.repositioning_loc = (0,0)                           # do not penalize if going to same spot
        self.repositioning_km = 0.0                              # how far the driver went
        self.current_premium = 0.0                               # what the premium of last ride was
        self.FAST = False                                        # used in SMDP training

    def enque_route_info(self, route, sec, end_loc):
        if not route:                    
                return
        self.route_queue.append((route, sec, end_loc))


    def generate_ride_details(self, olat, olng, dlat, dlng):
        month_index =  self.current_time.year * MONTHS_IN_YEAR + self.current_time.month
        base_trip_info = pd.DataFrame([{
            "begin_lat": olat, "begin_lng": olng,
            "end_lat":   dlat, "end_lng":   dlng,
            "hour": self.current_time.hour, 
            "dow": self.current_time.dayofweek, 
            "month_idx": month_index, 
            "doy":  self.current_time.dayofyear,
        }])
        return base_trip_info
    

    def advance_along_route(self, route_info):
        polyline, sec_left, end_location = route_info

        # if no polyline then treat reached to end
        if not polyline:
            return ([], 0.0, end_location), end_location
        
        sec_left = float(sec_left)
        epochs_left = math.ceil(sec_left / SECONDS_PER_EPOCH)

        # how many segments remain
        segments_left = max(len(polyline) - 1, 0)

        # number to hop
        if(epochs_left == 0):
            verts_this_epoch = 1
        else:
            verts_this_epoch = max(1, math.ceil(segments_left / epochs_left))
        
        # drop
        hop = min(verts_this_epoch, segments_left)
        # keep the vertex we land on as the new start
        polyline = polyline[hop:] if hop else polyline

        # new driver location is the first vertex of the trimmed polyline
        new_loc = polyline[0]

        # deduct one epoch of time
        sec_left = max(sec_left - SECONDS_PER_EPOCH, 0)

        # if that consumed everything, force terminal state
        if len(polyline) == 1:
            new_loc = end_location
            sec_left = 0.0                 

        return (polyline, sec_left, end_location), new_loc

    def move_locations(self, current_lat, current_lng, new_lat, new_lng):
        # generates the features needed to calculate distance
        features_df = self.generate_ride_details(current_lat, current_lng, new_lat, new_lng)

        # calculates the open source distance
        if(not self.FAST):
            osrm_sec, osrm_km, polyline, osrm_prediction_failed = osrm_call(current_lat, current_lng, new_lat, new_lng)
        else:
            osrm_km = haversine_vec(current_lat, current_lng, new_lat, new_lng)
            osrm_sec = osrm_km * 60 
            polyline = []

        # predicts distance in km
        km_pred = predict_distance(features_df)

        # adds the open source to the features df
        features_df["osrm_km"] = osrm_km
        features_df["osrm_sec"] = osrm_sec

        # predicts the duration of ride in sec
        sec_pred = predict_duration_ratio(features_df) * osrm_sec

        return features_df, km_pred, sec_pred, polyline
    
    def reject_move_action(self, new_lat, new_lng):
        self.repositioning = True
        self.repositioning_loc = location_to_grid(new_lat, new_lng)
        self.repositioning_km = 0.0
        # current location of driver - needed to calculate pick up distance
        current_lat, current_lng = self.loc
        # move locations
        _, km_pred, sec_pred, route = self.move_locations(current_lat, current_lng, new_lat, new_lng)
        sec_pred = float(sec_pred.item())
        # enque the route
        self.enque_route_info(route, sec_pred, (new_lat, new_lng))

        # return approx how many km per epoch
        total_minutes_rounded = round(sec_pred/60)
        if total_minutes_rounded == 0: total_minutes_rounded = 1
        self.repositioning_km = float(km_pred.item() / total_minutes_rounded)
        return km_pred
    

    # Function to initiate the drive 
    def job_information(self, ride):
        # current location of driver - needed to calculate pick up distance
        current_lat, current_lng = self.loc

        # start and end point of the ride offered
        origin_lat, origin_lng, dest_lat, dest_lng, _ = ride

        # CALCULATE PICK-UP DISTANCE
        _, km_pred_pickup, sec_pred_pickup, pickup_route = self.move_locations(current_lat, current_lng, origin_lat, origin_lng)
        sec_pred_pickup = float(sec_pred_pickup.item())

        # CALCULATE RIDE DISTANCE
        features_df_ride, km_pred_ride, sec_pred_ride, ride_route = self.move_locations(origin_lat, origin_lng, dest_lat, dest_lng)
        sec_pred_ride = float(sec_pred_ride.item())

        # predicts the take home price offered to the driver
        price_pred = predict_price(features_df_ride)

        pickup_info = (km_pred_pickup, sec_pred_pickup, pickup_route)
        ride_info = (km_pred_ride, sec_pred_ride, ride_route)

        return pickup_info, ride_info, price_pred

    def start_job(self, ride, epoch, pickup_info, ride_info, price_pred):
        # remove all plans currently on
        self.route_queue.clear()
        self.repositioning = False
        self.repositioning_km = 0.0

        # OBTAINS RIDE INFORMATION
        origin_lat, origin_lng, dest_lat, dest_lng, _ = ride

        # OBTAINS JOB INFORMATION
        km_pred_pickup, sec_pred_pickup, pickup_route = pickup_info
        km_pred_ride, sec_pred_ride, ride_route = ride_info

        # CALCULATES TOTAL DISTANCE AND TIME
        total_seconds = sec_pred_pickup + sec_pred_ride
        total_distance = km_pred_pickup + km_pred_ride
        total_minutes_rounded = round(total_seconds/60)

        # ADDS CURRENT JOB
        self.current_job = (ride, total_seconds, total_distance, price_pred, epoch)
        self.enque_route_info(pickup_route, sec_pred_pickup, (origin_lat, origin_lng))
        self.enque_route_info(ride_route, sec_pred_ride, (dest_lat, dest_lng))
        self.job_timer = total_minutes_rounded 
        self.on_job = True
        self.idle_period = 0 


    # Function to update the driver for each epoch
    def update(self):
        while self.route_queue:
            route_info = self.route_queue[0]
            # update the location and grid and advance the route
            advanced_route, self.loc = self.advance_along_route(route_info)
            self.grid = location_to_grid(self.loc[0], self.loc[1])

            # grabs second left
            _, secs, _ = advanced_route

            # replace updated head 
            if secs == 0:
                self.route_queue.popleft()     
            else:
                self.route_queue[0] = advanced_route
                break 
        
        if not self.route_queue and self.repositioning:
            self.repositioning = False
            self.repositioning_km  = 0.0

        if self.on_job == True:
            self.job_timer -= 1 # each epoch is a minute
            
            # If current job is done, reset
            if self.job_timer <= 0:
                # update job information
                self.on_job = False
                self.current_job = None
                self.current_premium = 0.0
                # reset idle period
                self.idle_period = 0
                self.past_locations.append(self.loc)
        else:
            # reset idle period
            self.idle_period += 1
            self.past_locations.append(self.loc)
        
        # increase the current time by 1 minute for each epoch
        self.current_time = self.current_time + pd.Timedelta(minutes=1)