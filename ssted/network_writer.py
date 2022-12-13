from ssted.tnet import *

import os


#TODO: Refactor to only having writing logic

def save_json(tn, name='tnet', start=0, end=30):
    print(tn)
    dirName = f"{name}-{start}-{end}"
    os.mkdir(dirName)
    for time in range(start, end):
        path = os.path.join(dirName, f"{time:02d}.json")
        json = open(path,'w')
        nodes, edges = tn.get(time,time+1)
        g = StaticNetwork(nodes, edges)
        g_string = str(g)
        print("Writing file")
        json.write(g_string)
        json.flush()
        json.close()
        print("File written")

#Updated 2021-3-5 for tnodes
def get_jsons(tn, name='tnet', start=0, end=30):
    jsons = []
    for time in range(start, end):
        nodes, edges = tn.get(time,time+1)
        print('nodes',nodes)
        # if isinstance(list(nodes)[0], TemporalNode):
        #     tnodes = []
        #     for n in nodes:
        #         tnodes.append(n.at_time(time))
        #     nodes = tnodes
        g = StaticNetwork(nodes, edges)
        g_string = str(g)
        jsons.append(g_string)
    return jsons




class StaticNetwork:
    def __init__(self, nodes=None, edges=None):
        self.nodes = set()
        self.edges = set()
        for n in nodes:
            self.add_node(n)
        for e in edges:
            self.add_edge(e)
    
    def add_edge(self, edge):
        #self.add_node(edge.src)
        #self.add_node(edge.dst)
        self.edges.add(edge)
    
    def add_node(self, node):
        self.nodes.add(node)
    
    def __str__(self):
        nodes =  ",".join( map(str,self.nodes))
        links =  ",".join( map(str,self.edges))
        return f'{{"nodes": [{nodes}], "links": [{links}]}}'