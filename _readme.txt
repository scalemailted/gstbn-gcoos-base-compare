gt02 Summary: Concurrency to speed up computation times

gt02 Status: In Progress! -> Use concurrency to perform all trials at once



Changes this version:
[1] Refactor the method --> add_oberver_node() so it manages a pool of trials
[2] define a method that runs a trial 
[3] aggregate the scores and select the best one to use as the new sensor 

Running application:
> cd gt02
> python3.9 gstn_analysis.py 




Code:

from multiprocessing import Pool

# Define a function that runs a single generation of the simulation
def run_generation(inputs):
    lon, lat = get_random_coord(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
    new_sensor = pd.DataFrame([{'Lon': lon, 'Lat': lat}])
    updated_coords = pd.concat([gcoos_coords,new_sensor], ignore_index = True)
    score = analyze(hycom_coords, updated_coords)
    return score

# Create a pool of worker processes
pool = Pool()

# Run the simulations in parallel
results = pool.map(run_generation, inputs)

# Close the pool
pool.close()

# Process the results
optimal_score = float('inf')
optimal_lat = None
optimal_lon = None
for result in results:
    if float(result['average_score']) < optimal_score:
        optimal_score = float(result['average_score'])
        optimal_lon = lon
        optimal_lat = lat

# Add the optimal sensor to the coordinates
new_sensor = pd.DataFrame([{'Lon': optimal_lon, 'Lat': optimal_lat}])
gcoos_cords = pd.concat([gcoos_coords,new_sensor], ignore_index = True)

return gcoos_cords, optimal_score