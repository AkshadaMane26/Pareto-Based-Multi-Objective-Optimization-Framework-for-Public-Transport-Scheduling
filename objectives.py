# objectives.py
import numpy as np
from simulation import routes, bus_capacity, cost_per_km, driver_cost_per_hour, avg_speed_kmh, service_hours

def calculate_cost(schedule):
    total_cost = 0
    for i, freq in enumerate(schedule):
        route = list(routes.values())[i]
        trips_per_day = freq * service_hours
        travel_time = route['distance_km'] / avg_speed_kmh
        cost = trips_per_day * route['distance_km'] * cost_per_km + trips_per_day * travel_time * driver_cost_per_hour
        total_cost += cost
    return total_cost

def calculate_waiting_time(schedule):
    waiting_times = []
    for i, freq in enumerate(schedule):
        # Avoid division by zero if freq==0 (shouldn't happen because enforced min=1)
        interval = 60.0 / max(freq, 1)
        avg_wait = interval / 2.0
        waiting_times.append(avg_wait)
    return np.mean(waiting_times)
