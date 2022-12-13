#Read Me
#Goal: Cleanup & Remove Code
#SUCCESS!! plot lines between nodes
#TODO edge generation --> the closer to death the more links to other nodes
#TODO Future Strategy --> pack all ncvars into one multidim array 2d and process all residuals in one computation 

import os
from typing import _ProtocolMeta
from webbrowser import get
from xml.etree.ElementTree import tostring
import netCDF4 as nc4
import numpy as np
import matplotlib.pyplot as p
import sys
import pandas as pd
from ssted import frame_controller
from ssted import tnet
from ssted import network_writer
from ssted import network_measures
from ssted import draw_measures
from ssted.location import Geolocation
import gcoss_nodes
from mpl_toolkits.basemap import Basemap


import h3
def find_nearest_node_gcoos(df, lon, lat):
    df['Dist'] = df.apply(lambda row: h3.point_dist((row['Lat'], row['Lon']), (lat, lon)), axis=1)
    df['Lon_ROI'] = lon
    df['Lat_ROI'] = lat
    return df[df.Dist==df.Dist.min()]

def get_edgelist(df):
    gcoos_df = gcoss_nodes.get_gcoos_dataframe()
    lonlat_df = gcoos_df[['Lon','Lat']].copy()
    #node_id = find_nearest_node_gcoos(lonlat_df,-90,30)
    #print(node_id)
    edgelist = [find_nearest_node_gcoos(lonlat_df,lon,lat) for lon, lat in zip(df['lon'], df['lat'])]
    edges_df = pd.concat(edgelist, axis=0)
    print(edges_df)
    return edges_df

def draw_edges(df, ax):
    x = df.loc[:, ['Lon', 'Lon_ROI']]
    y = df.loc[:, ['Lat', 'Lat_ROI']]
    for i in range(len(df)):
        p.plot(x.iloc[i,:], y.iloc[i,:],color='red',alpha=0.2)

def show():
    # setup Lambert Conformal basemap.
    m = Basemap(resolution='i', llcrnrlat=16.5,llcrnrlon=-99,urcrnrlat=33,urcrnrlon=-75)
    # draw coastlines.
    #m.drawcoastlines(color='lightgray')
    m.drawlsmask(land_color='lightgray', alpha=0.5)
    #fig = p.gcf()
    #fig.set_size_inches(18.5, 10.5)
    #fig.savefig('net-figure.png', dpi=100)
    p.show()

def get_nodelist():
    #['surf_el', 'time', 'lat', 'lon', 'salinity', 'depth', 'water_temp', 'water_u', 'water_v']
    hycom_variables = ['salinity', 'water_temp', 'water_u', 'water_v']
    roi_thresholds = [0.5, 1] #[0.25, 0.5, 1]
    nc_list  = loadAllNCs()
    node_list = {} 
    for hy_var in hycom_variables:
        for threshold in roi_thresholds:
            points_list = get_roi_by_observation(nc_list, hy_var, threshold)
            for time in range(len(points_list)):
                #print(f"{time} + {hy_var} + {threshold} -> len(points) = {len(points_list[time])}")
                #if time not in node:
                    #node_list[time] = {}
                #node_list[time] +=  [Node_HYCOM( Geolocation(lon,lat) ) for (lon,lat) in points_list[time]] 
                for (lon,lat) in points_list[time]:
                    geo = Geolocation(lon,lat)
                    #if geo not in node_list[time]:
                    if str(geo) not in node_list:
                        #node_list[time][geo] = Node_HYCOM(geo)
                        node_list[str(geo)] = Node_HYCOM(geo)
                    #node_list[time][geo].roi.add_observation(time, hy_var, threshold)
                    node_list[str(geo)].roi.add_observation(time, hy_var, threshold)
    return node_list

def main():
    node_list = get_nodelist()
    roi_snapshot = get_nodes_roi_at_time(node_list, 0)
    print(roi_snapshot)
    edge_df = get_edgelist(roi_snapshot)
    ax = draw_snapshot_nodes_roi(roi_snapshot)
    draw_node_gcoos(ax)
    draw_edges(edge_df, ax)
    show()


def get_nodes_roi_at_time(nodelist, time):
    roi_snapshot = [] 
    for node in nodelist.values():
        if time in node.roi.get_times():
            data = {}
            data['lon'] = node.geolocation.x
            data['lat'] = node.geolocation.y
            observations = node.roi.get_observations(time)
            for ncvar in observations:
                data[ncvar] = node.roi.timesnaps[time][ncvar]
            roi_snapshot.append(data)
    snapshot_df = pd.DataFrame.from_records(roi_snapshot)
    return snapshot_df

def draw_snapshot_nodes_roi(roi_snapshot):
    observations_df = roi_snapshot.loc[:, ~roi_snapshot.columns.isin(['lon', 'lat'])]
    fig, ax = p.subplots()
    ax.scatter(x=roi_snapshot['lon'], y=roi_snapshot['lat'], color='black', alpha=observations_df.max(axis=1) , marker=".", picker=True)
    return ax



def draw_node_gcoos(ax):
    gnodes = get_gcoos_nodes()
    for n in gnodes:
        ax.scatter(n.geolocation.x, n.geolocation.y, color='r', marker=".")


def get_gcoos_nodes():
    gcoos_df = gcoss_nodes.get_gcoos_dataframe()
    nodes = gcoss_nodes.get_gcoos_nodes(gcoos_df)
    return nodes



#Load netcdf files (HYCOM) 
def loadAllNCs(folder='./hycom_dataset'):
    nc_list = []
    dirlist = os.listdir(folder)
    for filename in sorted(dirlist):
        if '.nc4' in filename:
            print('loading... ' + filename)
            nc =  nc4.Dataset(folder+'/'+filename,'r', format='NETCDF4')
            nc_list.append(nc)
    return nc_list


def get_roi_by_observation(nc_list, nc_var, threshold):
    coords2d = get_coords2d(nc_list[0])
    grid2d_list = get_temporal_grid2d(nc_list, nc_var)
    residuals_list = get_all_residuals(grid2d_list)
    points_list = get_all_roi_points(residuals_list, coords2d, threshold)
    return points_list


#main - load all netcdf files
def main_old():
    hycom_variables = ['surf_el', 'time', 'lat', 'lon', 'salinity', 'depth', 'water_temp', 'water_u', 'water_v']
    nc_var = sys.argv[1] if len(sys.argv) > 1 else 'water_temp'
    nc_list, fnames = loadAllNCs()
    bbox = get_bounding_box(nc_list[0])
    coords2d = get_coords2d(nc_list[0])
    grid2d_list = get_temporal_grid2d(nc_list, nc_var)
    residuals_list = get_all_residuals(grid2d_list,nc_var)
    points_list = get_all_roi_points(residuals_list, coords2d)
    tnet.add_tnet_nodelist(points_list)
    #tnet.cluster_nodes()
    #tnet.add_tnet_edgelist() #TODO
    print('nodes:', len(tnet.nodes))
    print('edges:', len(tnet.edges))
    draw_nodes()
    #tn = tnet.get_tnet()
    #network_writer.save_json(tn,name='tnet',start=0,end=len(tn))
    #analyze(tn)




def compute_residual(this_grid, next_grid):
    residual = (this_grid - next_grid)**2
    return residual


def get_all_residuals(grid2d_list):
    residuals_list = []
    for i in range(len(grid2d_list)-1):
        this_grid, next_grid = grid2d_list[i], grid2d_list[i+1]
        residual = compute_residual(this_grid, next_grid)
        residuals_list.append(residual)
    return residuals_list


def get_all_roi_points(residuals_list, coords2d, threshold=0.25):
    roi_list = []
    for residual in residuals_list:
        roi = get_roi_coords(coords2d, residual, threshold)
        roi_list.append(roi)
    return roi_list


def get_roi_coords(coords, residual, threshold):
    index_lists = np.ma.where(residual > threshold) 
    grid_indexes = np.transpose(index_lists)
    points = [tuple(coords[i,j]) for i,j in grid_indexes]
    return points

def get_coords2d(nc):
    lon = nc.variables['lon']
    lat = nc.variables['lat']
    lons,lats= np.meshgrid(lon,lat)
    coords2d = np.dstack( (lons ,lats) )
    return coords2d


def generate_update(grid_list,img):
    return lambda i: img.set_data(grid_list[i]) 

def draw_colormaps(grid2d_list, bbox):
    fig, ax = p.subplots()
    #Create color map
    cmap = p.cm.get_cmap("jet").copy()
    cmap.set_bad('w',1.)
    maskedmap = grid2d_list[0]
    img = ax.imshow(maskedmap, interpolation='kaiser', cmap=cmap, aspect='auto', extent=bbox, origin='lower')
    update = generate_update(grid2d_list, img)
    ani = frame_controller.Player(fig, update, maxi=len(grid2d_list)-1)
    p.show()

def draw_points(points_list):
    fig, ax = p.subplots()
    points = np.array(points_list[0])
    path_col = ax.scatter(points[:,0], points[:,1], c='r', marker="o", picker=True)
    update = update_points(points_list, path_col)
    ani = frame_controller.Player(fig, update, maxi=len(points_list)-1)
    p.show()

def update_points(points_list, path_col):
    return lambda i: path_col.set_offsets(points_list[i])

def update_nodes(nodelist, ax):
    def update(i):
        clusters = nodelist[i]
        ax.clear()
        #colors = ['purple', 'blue', 'red', 'green', 'orange', 'yellow', 'gray', 'cyan']
        for label in clusters:
            points = np.array(clusters[label])
            ax.scatter(points[:,0], points[:,1], color='k', marker=".", picker=True)
            #ax.scatter(points[:,0], points[:,1], color=colors[label], marker=".", picker=True)
            #ax.plot(points[:,0], points[:,1], color='black', alpha=0.25)
        #centroids = np.array(tnet.centroids)
        #ax.scatter(centroids[:,0], centroids[:,1], color='black', marker="*", picker=True)
        #intraedges = np.array(tnet.intraedges)
        #ax.plot(intraedges[:,0], intraedges[:,1], color='black', alpha=0.5)
        #gcoos nodes
        global gnodes
        for n in gnodes:
            ax.scatter(n.geolocation.x, n.geolocation.y, color='r', marker=".")
    return update

def draw_nodes():
    nodelist = tnet.get_tnet_nodelist()
    fig, ax = p.subplots()
    clusters = nodelist[0]
    #colors = ['purple', 'blue', 'red', 'green', 'orange', 'yellow', 'gray', 'cyan']
    for label in clusters:
        points = np.array(clusters[label])
        ax.scatter(points[:,0], points[:,1], color='k', marker=".", picker=True)
        #ax.scatter(points[:,0], points[:,1], color=colors[label], marker=".", picker=True)
        #ax.plot(points[:,0], points[:,1], color='black', alpha=0.25)
    #path_col = ax.scatter(points[:,0], points[:,1], label=0, marker=".", picker=True)
    centroids = np.array(tnet.centroids)
    #ax.scatter(centroids[:,0], centroids[:,1], color='black', marker="*", picker=True)
    #intraedges = np.array(tnet.intraedges)
    #ax.plot(intraedges[:,0], intraedges[:,1], color='black', alpha=0.5)
    update = update_nodes(nodelist, ax)
    ani = frame_controller.Player(fig, update, maxi=len(nodelist)-1)
    #gcoos nodes
    global gnodes
    for n in gnodes:
        ax.scatter(n.geolocation.x, n.geolocation.y, color='r', marker=".")
    p.show()



def get_temporal_grid2d(nc_list, nc_var):
    temporal_grid2d = []
    for nc in nc_list:
        data_2d = get_netcdf_grid2d(nc, nc_var)
        temporal_grid2d.append(data_2d)
    return temporal_grid2d




def get_bounding_box(ds):
    LON_MIN = ds.geospatial_lon_min
    LON_MAX = ds.geospatial_lon_max
    LAT_MIN = ds.geospatial_lat_min
    LAT_MAX = ds.geospatial_lat_max
    return (LON_MIN, LON_MAX, LAT_MIN, LAT_MAX) 


def draw_colormap(masked_array, bounding_box):
   #Create color map
   cmap = p.cm.get_cmap("jet").copy()
   cmap.set_bad('w',1.)
   p.imshow(masked_array, interpolation='kaiser', cmap=cmap, aspect='auto', extent=bounding_box, origin='lower')
   p.show()


def get_netcdf_grid2d(nc_data, nc_var, time=0, depth=0):
   val2d = nc_data.variables[nc_var][time,depth]               #generate lon/lat values
   masked_array = np.ma.array (val2d, mask=np.isnan(val2d))    #mask the land values
   return masked_array
   


#merging masked arrays --> use for merging all variables

def analyze(tn): 
    degrees = temporal_degree_centrality(tn)
    draw_temporal_degree_centrality(degrees)
    print('degrees')
    degree_totals = temporal_degree_centrality_overall(degrees)
    draw_temporal_degree_centrality_overall(degree_totals)
    print('degree totals')
    overlaps_network = topological_overlap_overall(tn)
    draw_topological_overlap_overall(overlaps_network)
    print('overlaps-network')
    overlaps_nodes = topological_overlap(tn)
    draw_topological_overlap(overlaps_nodes)
    print('overlaps-nodes')
    overlap_averages = topological_overlap_average(overlaps_nodes)
    draw_topological_overlap_average(overlap_averages)
    print('overlap averages')
    tcc = temporal_correlation_coefficient(overlap_averages)
    draw_temporal_correlation_coefficient(tcc)
    print('tcc', tcc)
    #New Analysis
    icts = intercontact_times(tn)
    draw_icts_avg_lag(icts)
    draw_icts_max_lag(icts)
    print('icts',icts)
    edge_coeff = bursty_coeff(icts)
    draw_bursty_coeff(edge_coeff)
    draw_bursty_coeff_avg(edge_coeff)
    print('bursty coeff',edge_coeff)
    lvs = local_variation(icts)
    draw_lvs(lvs)
    draw_lvs_avg(lvs)
    print('lvs', lvs)
    paths = get_shortest_paths(tn)
    draw_shortest_paths(paths)
    print('paths',paths,0,len(tn))
    tcc = closeness_centrality(paths)
    draw_closeness(tcc)
    print('tcc', tcc)
    tbc = betweenness_centrality(paths)
    draw_betweenness(tbc)
    print('tbc', tbc)


#[ CLASS: Node_HYCOM ]#######################################################################

'''
def add_node(lon, lat, time):
    location = Geolocation(lon, lat)
    id = str(location)
    if id in nodes:
        nodes[id].occurrences.add(time)
        nodes[id].alive = True
        nodes[id].ttl = 2
    else:
        nodes[id] = Node(id, location, time)
    return nodes[id]
'''

def get_hycom_args(data):
    params = {}
    params['geolocation'] = Geolocation(data['lon'],data['lat'])
    params['id'] = str(params['geolocation'])
    params['roi_strength']
    params['timeframes']
    params['observations']
    return params


'''
def make_hycom_node(lon, lat, time, observation, strength):
    geo = Geolocation(lon, lat)
    node = Node_HYCOM(geo)
    node.roi.add_observation(time, observation, strength)
'''

# timeframe --> observation_set --> foreach:obs  --> roi_str 

#TODO: reset roi_strength each frame or decrement by delta value if not appear until roi_strength is 0
class Node_HYCOM:
        def __init__(self, geolocation):
            self.geolocation = geolocation
            self.id = str(self.geolocation)
            self.roi = ROI()
            #self.id = params['id']
            #self.roi_strength = params['roi_strength']
            #self.timeframes = params['timeframes']
            #self.observations = params['observations']
            
        def __str__(self):
            toString = "[Node-HYCOM]: { "
            toString += f"id:{self.id}"
            toString += f", geolocation:{self.geolocation}"
            toString += f", roi:{self.roi}"
            #toString += f"roi_strength:{self.roi_strength}, "
            #toString += f"timeframes:{self.timeframes}, "
            #toString += f"observations:{self.observations}"
            toString += " }"
            return toString
        
        def __eq__(self, other):
            return isinstance(other, Node_HYCOM) and self.geolocation == other.geolocation


class ROI:
    def __init__(self):
        self.timesnaps = {}
    #
    def add_observation(self, time, hycom_variable, roi_strength):
        if time not in self.timesnaps:
            self.timesnaps[time] = {}
        self.timesnaps[time][hycom_variable] = roi_strength
    #
    def get_times(self):
        return self.timesnaps.keys()
    #
    def get_observations(self, time):
        return self.timesnaps[time].keys() if (time in self.timesnaps) else None
    #
    def get_all_observations(self):
        observations = set()
        for time in self.timesnaps:
            observations |= self.timesnaps[time].keys()
        return observations
    #
    def __str__(self):
        toString = "[ROI]: { "
        toString += f"times:{list(self.timesnaps.keys())}, "
        toString += f"observations: {self.get_all_observations()}"
        toString += " }"
        return toString
    





#[ END: Node_HYCOM ]###########################################################################



#[MONET CARLO SIM] ###########################################################################
#v11
def get_gom_bbox():
    nc_list = loadAllNCs()
    bbox = get_bounding_box(nc_list[0])
    return bbox


if __name__ == "__main__":
    main()






