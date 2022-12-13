import random
import pandas as pd
import hycom_nodes
import gcoss_nodes
import gstn_analysis
from ssted import tnet
#for concurrency
from multiprocessing import Pool
# for timestamp log messages
import time
import tensorflow as tf

def main():
    trial_size=100
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = hycom_nodes.get_gom_bbox()
    #print('bbox: ',LON_MIN, LON_MAX, LAT_MIN, LAT_MAX )
    hycom_coords = hycom_nodes.get_nodelist()
    gcoos_coords = get_gcoos_emptydf()
    #START --> Second sensor suggestion
    # new optimal: -78.7403109976445, 24.385624429875215, score: 160873.8810059777
    #new_sensor = pd.DataFrame([{'Lon': -78.7403109976445, 'Lat': 24.385624429875215}])
    #gcoos_coords = pd.concat([gcoos_coords,new_sensor], ignore_index = True)
    #END --> Second sensor suggestion
    coverage = float('inf')
    target_score = 180222.806856
    while coverage > target_score:
        gcoos_coords, coverage = insert_observer_node(trial_size, gcoos_coords, hycom_coords,LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
        print('current count:', len(gcoos_coords))
    # Write the DataFrame to a file
    gcoos_coords.to_csv('results/gcoos_coords.csv', sep=',', index=False)
    print('node count :', len(gcoos_coords))
    print('coverage :', coverage)


####################################


# Define a function that runs a single trial of the simulation
def run_trial(gpu_id, inputs):
    gpu_id, data = inputs
    # Set the GPU that the worker process will use
    with tf.device(f"/gpu:{gpu_id}"):
        # Run the task here
        lon, lat, gcoos_coords, hycom_coords = data
        new_sensor = pd.DataFrame([{'Lon': lon, 'Lat': lat}])
        updated_coords = pd.concat([gcoos_coords,new_sensor], ignore_index = True)
        score = analyze(hycom_coords, updated_coords)
        return (float(score['average_score']),lon,lat)


def insert_observer_node(trial_size, gcoos_coords,hycom_coords, lon_min, lon_max, lat_min, lat_max):
    # Get the number of GPUs available in Colab
    num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    # Create a pool of worker processes
    #pool = Pool()
    # Generate the input data for the simulations
    inputs = []
    # Use for loops to create a list of tuples containing GPU IDs and task IDs
    for gpu_id in range(num_gpus):
        for i in range(trial_size):
            lon, lat = get_random_coord(lon_min, lon_max, lat_min, lat_max)
            input_data = (lon, lat, gcoos_coords, hycom_coords)
            inputs.append((gpu_id, input_data))
    print("computing new observer...")
    # Create a timer for log message on elapsed time
    timer = make_timer()
    # Run the simulations in parallel
    #results = pool.map(run_trial, inputs)
    # Use the multiprocessing package to run the tasks on multiple GPUs
    with Pool(processes=num_gpus) as pool:
        pool.starmap(run_trial, inputs)
    # Close the pool
    #pool.close()
    # Call the timer function for a log message on elapsed time
    timer()
    # Process the results
    score, lon, lat = min(results, key=lambda x: x[0])
    # Add the optimal sensor to the coordinates
    new_sensor = pd.DataFrame([{'Lon': lon, 'Lat': lat}])
    gcoos_cords = pd.concat([gcoos_coords,new_sensor], ignore_index = True)
    print(f"\nnew optimal: {lon}, {lat}, score: {score}\n")
    return gcoos_cords, score


# Define a function that returns a closure function
def make_timer():
    # Define a local variable to store the previous time
    previous_time = time.perf_counter()
    print(time.strftime("%H:%M:%S", time.localtime()))
    # Define the closure function
    def timer(previous_time=previous_time):
        # Get the current time
        current_time = time.perf_counter()
        # Calculate the elapsed time
        elapsed_time = current_time - previous_time
        # Print the elapsed time
        print(f'Elapsed time: {elapsed_time:.6f} seconds')
        # Update the previous time
        previous_time = current_time
    # Return the closure function
    return timer

###################################


"""
def insert_observer_node( gcoos_coords,hycom_coords, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX):
    optimal_score = float('inf')
    optimal_lat = None
    optimal_lon = None
    generations = 100
    for i in range(generations):
        print(f"Generation: {i+1}/{generations}")
        lon, lat = get_random_coord(LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
        new_sensor = pd.DataFrame([{'Lon': lon, 'Lat': lat}])
        updated_coords = pd.concat([gcoos_coords,new_sensor], ignore_index = True)
        #print('gcoos_coords',gcoos_coords)
        #print('updated_coords',updated_coords)
        score = analyze(hycom_coords, updated_coords)
        #print('score', score)
        if float(score['average_score']) < optimal_score:
            optimal_score = float(score['average_score'])
            optimal_lon = lon
            optimal_lat = lat
            print(f"\nnew optimal: {optimal_lon}, {optimal_lat}, score: {optimal_score}\n")
        else:
            print( f"\tRESULT: {float(score['average_score'])} >= {optimal_score}" )
    new_sensor = pd.DataFrame([{'Lon': optimal_lon, 'Lat': optimal_lat}])
    gcoos_cords = pd.concat([gcoos_coords,new_sensor], ignore_index = True)
    return gcoos_cords, optimal_score
"""

def get_random_coord(min_lon, max_lon, min_lat, max_lat):
        lon = random.uniform(min_lon, max_lon)
        lat = random.uniform(min_lat, max_lat)
        return (lon, lat)


def get_gcoos_emptydf():
    gcoos_df = pd.DataFrame(columns=['Lon', 'Lat'])
    return gcoos_df


def get_gcoos_coords():
    gcoos_df = gcoss_nodes.get_gcoos_dataframe()
    gcoos_coords = gcoos_df[['Lon','Lat']].copy()
    return gcoos_coords

def analyze(hycom_coords, gcoos_nodes):
    #nodes_df_list = [ gcoos_nodes ] * 7 
    edges_df_list = []
    for time in range(7):
        #print(f"\tsnapshot: {time}/{7-1}")
        roi_snapshot = hycom_nodes.get_nodes_roi_at_time(hycom_coords, time)
        edges_df = get_edgelist(roi_snapshot, gcoos_nodes)
        edges_df_list.append(edges_df)
    #tn = tnet.from_list_of_dataframes(edges_df_list, nodes_df_list)
    #analyze(tn)
    coverage_score = gstn_analysis.analyze_temporal_coverage(edges_df_list)
    return coverage_score



def get_edgelist(hycom, gcoos):
    gcoos_df = gcoss_nodes.get_gcoos_dataframe()
    edgelist = [hycom_nodes.find_nearest_node_gcoos(gcoos,lon,lat) for lon, lat in zip(hycom['lon'], hycom['lat'])]
    edges_df = pd.concat(edgelist, axis=0)
    #print(edges_df)
    return edges_df

if __name__ == "__main__":
    main()

