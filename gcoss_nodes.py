from calendar import c
import os
import netCDF4 as nc4
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.cm as cm
from ssted import tnet
from ssted.location import Geolocation





def main():
    gcoos_df = get_gcoos_dataframe()
    nodes = get_gcoos_nodes(gcoos_df)
    for n in nodes:
        print(n)
    draw_gcoos_nodes(nodes)
    plt.show()
    #draw_nodes(gcoos_df)



def draw_gcoos_nodes(nodes):
    for n in nodes:
        plt.scatter(n.geolocation.x, n.geolocation.y, color='r', marker=".")
    #plt.show()


def draw_nodes(df):
    x_name = 'Lon'
    y_name = 'Lat'
    tooltip_name = 'Platform/Station'
    x = df[x_name]
    y = df[y_name]
    tt = df[tooltip_name].values

    indicators = {
        'membership':{
            'name':'Membership',
            'values':df['Membership'].values,
            'poly':Polygon([(0, 1.4), (0, 1.6), (2, 1.6), (2, 1.4)])
        },
        'datasource':{
            'name':'Data source',
            'values':df['Data Source'].values,
            'poly':Polygon([(0, 1.2), (0, 1.4), (2, 1.4), (2, 1.2)]) 
        },
        'platform':{
            'name':'Platform/Station',
            'values':df['Platform/Station'].values,
            'poly':Polygon([(0, 1.), (0, 1.2), (2, 1.2), (2, 1.)])
        },
        'observations':{
            'name':'Observation(s)',
            'values':df['Observation(s)'].values,
            'poly':Polygon([(0, 0.8), (0, 1.), (2, 1.), (2, 0.8)]) 
        }
    }

    # define figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4), gridspec_kw={'width_ratios': [3, 2]}, facecolor='#393939')
    ax1.tick_params(axis='both', colors='w')
    cmap = plt.get_cmap("viridis")
    plt.suptitle('GCOOS Sensor Array', color='w')

    # scatter plot
    colors = [float(hash(s) % 256) / 256 for s in df["Data Source"]] 
    #sc = ax1.scatter(x, y, c=df["Data Source"])
    sc = ax1.scatter(x, y, c=colors)

    # axis 1 labels
    ax1.set_xlabel(x_name, color='w')
    ax1.set_ylabel(y_name, color='w')

    # axis 2 ticks and limits
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlim(0,2)
    ax2.set_ylim(0,2)

    # place holder for country name in axis 2
    cnt = ax2.text(1, 1.8, '', ha='center', fontsize=8)

    # indicator texts in axis 2
    txt_x = 1.8
    txt_y = 1.5
    for ind in indicators.keys():
        n = indicators[ind]['name']
        indicators[ind]['txt'] = ax2.text(txt_x, txt_y, n.ljust(len(n)+13), ha='right', fontsize=8)
        txt_y -= 0.2

    # line break in axis 2
    ax2.plot([0,2],[txt_y-0.1, txt_y-0.1], lw=3, color='#393939')

    # annotation / tooltip
    annot = ax1.annotate("", xy=(0,0), xytext=(5,5),textcoords="offset points", bbox=dict(boxstyle="round,pad=0.3", fc="w", lw=2))
    annot.set_visible(False)

    # notes axis 2
    note ="""NCOOS nodes"""

    ax2.text(0.05, 0.48, note, ha='left', va='top', fontsize=8)

    # change color map of axis 1
    def change_cmap(values, annotation):
        clean_ax2()
        sc.set_norm(plt.Normalize(np.nanmin(values), np.nan_to_num(values).max()))
        annotation.set_color('#2A74A2')
        sc.set_array(values)

    # clean text in axis 2 and reset color of axis 1
    def clean_ax2():
        for ind in indicators.keys():
            indicators[ind]['txt'].set_color('black')
        sc.set_color('black')

    # cursor hover
    def hover(event):
        # check if event was in axis 1
        if event.inaxes == ax1:        
            clean_ax2()
            # get the points contained in the event
            cont, ind = sc.contains(event)
            if cont:
                # change annotation position
                annot.xy = (event.xdata, event.ydata)
                # write the name of every point contained in the event
                station = "{}".format(', '.join([tt[n] for n in ind["ind"]]))
                print("station: ", station)
                annot.set_text(station)
                annot.set_visible(True)
                # get id of selected country
                station_id = ind["ind"][0]
                # set axis 2 country label
                cnt.set_text(tt[station_id])
                # set axis 2 indicators values
                for ind in indicators.keys():
                    n = indicators[ind]['name']
                    txt = indicators[ind]['txt']
                    val = indicators[ind]['values'][station_id]
                    txt.set_text('{}:\n{}'.format(n, val))
            # when stop hovering a point hide annotation
            else:
                annot.set_visible(False)
        # check if event was in axis 2
        elif event.inaxes == ax2:
            # bool to detect when mouse is not over text space
            reset_flag = False
            for ind in indicators.keys():
                # check if cursor position is in text space
                if indicators[ind]['poly'].contains(Point(event.xdata, event.ydata)):
                    # clean axis 2 and change color map
                    clean_ax2()
                    change_cmap(indicators[ind]['values'], indicators[ind]['txt'])
                    reset_flag = False
                    break
                else:
                    reset_flag = True
            # If cursor not over any text clean axis 2 
            if reset_flag:
                clean_ax2()
        fig.canvas.draw_idle()   

    # when leaving any axis clean axis 2 and hide annotation
    def leave_axes(event):
        clean_ax2()
        annot.set_visible(False)

    fig.canvas.mpl_connect("motion_notify_event", hover)
    fig.canvas.mpl_connect('axes_leave_event', leave_axes)
    plt.show()

    
def get_gcoos_points(df):
    return np.array([df['Lon'], df['Lat']]).T

#add mobility column to gcoos dataframe
def add_mobilty_column(df):
    #create new mobility column based on regex on {mobile} in lon/lat cols
    conditions = [
        df['Lon'].astype(str).str.contains('{mobile}', regex= True, na=False) == True,
        df['Lon'].astype(str).str.contains('{mobile}', regex= True, na=False) == False
    ]
    values = ['mobile', 'stationary']
    df['Mobility'] = np.select(conditions, values)
    #
    #remove {mobile} from lon/lat columns & cast into float
    df['Lon'] = df['Lon'].astype(str).str.replace(' {mobile}', '', regex=False).astype(float)
    df['Lat'] = df['Lat'].astype(str).str.replace(' {mobile}', '', regex=False).astype(float) 

def merge_gcoos_frames(federal, ldn):
    federal['Membership'] = 'federal'
    ldn['Membership'] = 'ldn'
    gcoos = pd.concat([federal, ldn])
    return gcoos


#load all csv files
def get_gcoos_dataframe(folder='../../gcoos_dataset'):
    federal = pd.read_csv(f"{folder}/gcoos-federal-sensors.csv")
    ldn = pd.read_csv(f"{folder}/gcoos-ldn-sensors.csv")
    gcoos = merge_gcoos_frames(federal, ldn)
    add_mobilty_column(gcoos)
    return gcoos


def get_gcoos_args(row):
    row_id, data_source, platform, lat, lon, observations, membership, mobility = row 
    params = {}
    params['id'] = row_id
    params['data_source'] = data_source
    params['platform'] = platform
    params['geolocation'] = Geolocation(lon,lat)
    params['observations'] = observations.replace(" ", "").split(",")
    params['membership'] = membership
    params['mobility'] = mobility
    return params


#get Node objects from GCOOS Dataframe
def get_gcoos_nodes(df):
    nodes = [Node_GCOOS(get_gcoos_args(row)) for row in df.itertuples()]
    return nodes



#load all ncs
def loadAllNCs(folder='../gcoos_dataset'):
    nc_list = {'ldn':[], 'federal':[] }
    for key in nc_list:
        nc_path = f"{folder}/{key}/"
        dirlist = os.listdir(nc_path)
        for filename in sorted(dirlist):
            if '.nc' in filename:
                print(f"loading... [{key}]: {filename}")
                nc =  nc4.Dataset(f"{nc_path}/{filename}",'r', format='NETCDF4') 
                nc_list[key].append(nc)
    return nc_list

def get_platform(nc_data):
    return nc_data.__dict__['platform']

def get_instrument(nc_data):
    return nc_data.__dict__['instrument']

def get_coords(nc_data):
    lon = float(nc_data.variables['lon'][:].data )
    lat = float(nc_data.variables['lat'][:].data )
    return (lon, lat)

def get_points_list(nc_list):
    points = [ get_coords(nc) for nc in nc_list ]
    return points

def get_bounding_box(nc_list):
    lon_min = min([ nc.geospatial_lon_min for nc in nc_list])
    lon_max = max([ nc.geospatial_lon_max for nc in nc_list])
    lat_min = min([ nc.geospatial_lat_min for nc in nc_list])
    lat_max = max([ nc.geospatial_lat_max for nc in nc_list])
    return (lon_min, lon_max, lat_min, lat_max) 

def draw_points(points_list):
    fig, ax = p.subplots()
    points = np.array(points_list)
    path_col = ax.scatter(points[:,0], points[:,1], c='r', marker="o", picker=True)
    p.show()



#[ CLASS: Node_GCOOS ]#######################################################################

class Node_GCOOS:
        def __init__(self, params):
            #Props:  Data Source,Platform/Station,Lat,Lon,Observation(s)
            self.id = params['id']
            self.membership = params['membership']
            self.data_source = params['data_source']
            self.platform = params['platform']
            self.mobility = params['mobility']
            self.geolocation = params['geolocation']
            self.observations = params['observations']
        
        def __str__(self):
            toString = "[Node-GCOOS]: { "
            toString += f"id:{self.id}, "
            toString += f"membership:\"{self.membership}\", "
            toString += f"data_source:\"{self.data_source}\", "
            toString += f"platform:\"{self.platform}\", "
            toString += f"mobility:\"{self.mobility}\", "
            toString += f"geolocation:{self.geolocation}, "
            toString += f"observations:{self.observations} "
            toString += "}"
            return toString

#[ END: Node_GCOOS ]###########################################################################

#[ Edgelist ]##################################################################################
edge_list = []

#[ END: Edgelist ]#############################################################################


if __name__ == "__main__":
    main()
