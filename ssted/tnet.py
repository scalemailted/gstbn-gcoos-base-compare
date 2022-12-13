from sklearn.cluster import KMeans
import numpy as np

from ssted.location import Geolocation as Geolocation


nodes = {}
edges = {}
snapshots = {}
centroids = []
intraedges = []

'''
def add_tnet_edgelist():
    nodelist = get_tnet_nodelist()
    num_times = len(nodelist)
    num_clusters = len(nodelist[0])
    for t in range(num_times):
        snapshots[t]['edges'] = []
        for c in range(num_clusters):
            cluster = nodelist[t][c]
            for i,j in zip(cluster[:-1],cluster[1:]):
                src = nodes[get_node_id(i[0],i[1])]
                dst = nodes[get_node_id(j[0],j[1])]
                edge_id = get_edge_id(src,dst)
                if edge_id in edges:
                    edges[edge_id].add(t)
                else:
                    te = TemporalEdge(src, dst,t)
                    edges[te.id] = te
                    snapshots[t]['edges'].append(te)
'''





def from_list_of_dataframes(edges_df_list, nodes_df_list):
    tn = TemporalNetwork()
    #time_length = len(df_list)
    for index in range(len(nodes_df_list)):
        df = nodes_df_list[index]
        for row in df.itertuples():
            coord = Geolocation(x=row.Lon, y=row.Lat, precision=6)
            id = f"GCOOS-{row.Index}-{str(coord)}"
            node = Node(id,coord,index)
            tn.add_node(node)
            #print(f"Time[{index}]: add Nodes from NodeList: {len(tn.nodes)} /{len(df)} --> {str(node)}")
    for index in range(len(edges_df_list)):
         #src_nodeslist = df.loc[:, src_node].values.tolist()
        df = edges_df_list[index]
        for row in df.itertuples():
            src_coord = Geolocation(x=row.Lon, y=row.Lat, precision=6)
            src_id = f"GCOOS-{row.Index}-{str(coord)}"
            src_node = Node(src_id, src_coord, index)
            dst_coord = Geolocation(row.Lon_ROI, row.Lat_ROI)
            dst_id = f"HYCOM-{str(dst_coord)}"
            dst_node = Node(dst_id, dst_coord, index)
            edge = TemporalEdge(src_node, dst_node, index)
            tn.add_edge(edge,[index])
            #print(f"Time[{index}]: add Nodes from EdgeList: {len(tn.nodes)}")
            #print(f"Time[{index}]: add Edge from EdgeList: {len(tn.edges)}")
    #return tn
    return tn


def get_tnet_nodelist():
    tnet_nodelist = []
    for time in snapshots:
        nodes = {}
        for node in snapshots[time]["nodes"]:
            p = (node.location.x, node.location.y)
            if node.label not in nodes:
                nodes[node.label] = []
            nodes[node.label].append(p)
        tnet_nodelist.append(nodes)
    return tnet_nodelist


def add_tnet_nodelist(tnet_node_list):
    for t in range(len(tnet_node_list)):
        node_list = tnet_node_list[t]
        for node in node_list:
            lon, lat = node
            node = add_node(lon, lat, t)
            i = 0
            while (node.alive and t+i < len(tnet_node_list)):
                add_snapshot(t+i,node=node)
                node.decay()
                i+=1

def add_snapshot(time,node=None, edge=None):
    if time not in snapshots:
        snapshots[time] = {"nodes":set(), "edges":set()}
    if node:
        snapshots[time]["nodes"].add(node)
    if edge:
        snapshots[time]["edges"].add(edge)


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

def get_node_id(lon, lat):
    return str(Geolocation(lon,lat))

def get_edge_id(src ,dst):
    return f'({src.id},{dst.id})'


class Node:
    def __init__(self, id, loc, time, label=0, ttl=float('inf'), group=1):
        self.id = id
        self.location = loc
        self.ttl = ttl
        self.alive = True
        self.links = set()
        self.occurrences = set([time])
        self.cluster = label
        self.group = group

    def add_occurrence(self, time):
        self.occurrences.add(time)
    
    def decay(self):
        self.ttl -= 1
        if self.ttl < 0:
            self.alive = False
    
    def __str__(self):
        pos_str = f' "fx": {self.location.x}, "fy": {self.location.y},' if self.location else ""
        return f'{{"id": "{self.id}",{pos_str} "group": {self.group}}}'
    
    def __hash__(self):
        return hash(self.id)


def cluster_nodes():
    points = [(nodes[id].location.x, nodes[id].location.y) for id in nodes]
    points = np.array(points)
    #Kmeans cluster
    print("computing kmeans...")
    kmeans = KMeans(init='k-means++', n_clusters=8, n_init=10 )
    #predict the labels of clusters.
    label = kmeans.fit_predict(points)
    #Getting unique labels
    u_labels = np.unique(label)
    print("coloring clusters...")
    #plotting the results:
    #for i in u_labels:
        #p.scatter(points[label == i , 0] , points[label == i , 1] , label = i)
    #for i in u_labels:
    #    clusters[i] = (points[label == i , 0] , points[label == i , 1])
    for (x,y),i in zip(points,label):
        id = get_node_id(x,y)
        nodes[id].label = i
    
    kmeans.fit(points)
    for i in u_labels:
        #centroids[i] = kmeans.cluster_centers_[i]
        centroids.append(kmeans.cluster_centers_[i])
        x,y = kmeans.cluster_centers_[i]
        for t in snapshots:
            add_node(x,y,t)
        id = get_node_id(x,y)
        nodes[id].label = i

    #p.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='w', marker="o", picker=True) 
    print('ulabels: ',u_labels)
    print('label', label)
    print('centers: ',kmeans.cluster_centers_) 
    #print(clusters.keys())
    #print(centroids)
    add_intra_edges()


def add_intra_edges():
    for i in centroids[:-1]:
        for j in centroids[1:]:
            intraedges.append((i[0],i[1]))
            intraedges.append((j[0], j[1]))

'''
def add_tnet_edgelist():
    nodelist = get_tnet_nodelist()
    num_times = len(nodelist)
    num_clusters = len(nodelist[0])
    for t in range(num_times):
        snapshots[t]['edges'] = []
        for c in range(num_clusters):
            cluster = nodelist[t][c]
            for i,j in zip(cluster[:-1],cluster[1:]):
                src = nodes[get_node_id(i[0],i[1])]
                dst = nodes[get_node_id(j[0],j[1])]
                edge_id = get_edge_id(src,dst)
                if edge_id in edges:
                    edges[edge_id].add(t)
                else:
                    te = TemporalEdge(src, dst,t)
                    edges[te.id] = te
                    snapshots[t]['edges'].append(te)
'''
                

def get_tnet():
    tnet = TemporalNetwork()
    for n in nodes.values():
        tnet.add_node(n)
    for e in edges.values():
        tnet.add_edge_without_nodes(e)
    return tnet





###########################################################



#TODO Add edges --> the below code is from ssted
class TemporalEdge:
    def __init__(self, src, dst, *times):
        if isinstance(src, Node):
            self.src = src
        else:
            self.src = Node(src)
        if isinstance(dst, Node):
            self.dst = dst
        else:
            self.dst = Node(dst)
        self.occurences = frozenset(times)
        self.weight = 1
        self.id = f'({self.src.id},{self.dst.id})'
    
    def add(self, other):
        if isinstance(other, TemporalEdge) and self == other:
            self.occurences |= other.occurences
        elif isinstance(other, (int, float, complex)):
            self.occurences =  frozenset(list(self.occurences) + [other])
        elif isinstance(other, (tuple,list,set,frozenset)): #ADDED [NO TEST]
            self.occurences |= set(other) 
    
    def contains(self, node):
        if isinstance(node, (str,int)):
            return node in (self.src.id, self.dst.id)
        else:
            return node in (self.src, self.dst)
            
    def __add__(self, other):
        if isinstance(other, TemporalEdge) and self == other:
            times = self.occurences | other.occurences 
            return TemporalEdge(self.src, self.dst, *times)
        elif isinstance(other, (int, float, complex)):
            times = self.occurences | set([other]) 
            return TemporalEdge(self.src, self.dst, *times)
        elif isinstance(other, (tuple,list,set,frozenset)): #ADDED [NO TEST]
            times = self.occurences | set(other) 
            return TemporalEdge(self.src, self.dst, *times)
    
    def __eq__(self, other):
        return isinstance(other, TemporalEdge) and self.src == other.src and self.dst == other.dst
    
    def __len__(self):
        return len(self.occurences)
    
    def __str__(self):
        return f'{{"source": "{self.src.id}", "target": "{self.dst.id}", "value": {self.weight}}}'
    
    def __hash__(self):
        return id(self)

###########################################################

from collections import defaultdict
class TemporalNetwork:
    def __init__(self):
        self.nodes = dict()
        self.edges = dict()
        self.timestamps = defaultdict(set) 
        #self.timestamps = dict()
        #self.timestamps = set()
        
    def get(self, start=0, end=1):
        edgelist = { edge for (time,edges) in self.timestamps.items() for edge in edges if start <= time < end }
        #edgelist = set()
        #for time,edges in self.timestamps.items():
        #    if start <= time <= end:
        #        edgelist |= edges
        #    if time >= end:
        #        break
        return (self.nodes.values(), edgelist)
    
    def add_node(self, node=None, pos=None, group=1):
        if isinstance(node, Node):
            self.nodes[node.id] = node
        elif node:
            self.nodes[node] = Node(node, pos, group)

    def add_tnode(self, node=None, pos=None, time=None, group=1):
        if isinstance(node, TemporalNode):
            if not self.nodes.get(node.id):
                self.nodes[node.id] = node
            self.nodes[node.id].set_position(time,pos) 
        elif node:
            print('node',node)
            if not self.nodes.get(node):
                self.nodes[node] = TemporalNode(node, group)
            self.nodes[node].set_position(time,pos) 
    
    def add_nodes(self, nodes):
        if isinstance(nodes, dict):
            print("untested")
            self.nodes.update(nodes)
        elif isinstance(nodes, (list, set, frozenset, tuple)):
            self.nodes.update({n.id: n for n in nodes if not n.id in nodes})
    
    def add_edge(self, edge, *times):
        self.add_nodes( [edge.src, edge.dst] )
        if times:
            edge.occurences |= set(*times)
        #self.timestamps |= { time: edge for time in edge.occurences} 
        for time in edge.occurences:
            self.timestamps[time].add(edge)
        if edge in self.edges:
            self.edges[edge.id] += edge
        else:
            self.edges[edge.id] = edge
    
    #refactor this with above method
    def add_edge_without_nodes(self, edge, *times):
        if times:
            edge.occurences |= set(*times)
        #self.timestamps |= { time: edge for time in edge.occurences} 
        for time in edge.occurences:
            self.timestamps[time].add(edge)
        if edge in self.edges:
            self.edges[edge.id] += edge
        else:
            self.edges[edge.id] = edge
    
    def get_neighbor_edges(self, node):
        #return [ edge.id for edge in self.edges.values() if node in (edge.src.id, edge.dst.id) ]
        return { edge.id for edge in self.edges.values() if edge.contains(node) }
    
    def __len__(self):
        return len(self.timestamps)
        
    def __str__(self):
        return f'{{"snapshots":  {len(self.timestamps)}, "nodes": {len(self.nodes)}, "edges:" {len(self.edges)}}}'





