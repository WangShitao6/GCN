#interface for graph
import networkx as nx

class Graph:

    def __init__(self,):
        self.name = None
        self.graph = nx.Graph()
        pass

    def __repr__(self):
        return "graph name:{}\ngraph:{}\n".format(self.name,self.graph)
    
    def add_node(self,node):
        self.graph.add_node(node)

    def add_nodes_from(self,nodes):
        self.graph.add_nodes_from(nodes)

    def add_edge(self,*edge):#edge should be a tuple
        self.graph.add_edge(*edge)
    
    def add_edges(self,edges):
        self.graph.add_edges_from(edges)

    def add_node_attribute(self,node,attr,attr_name='attribute'):
        self.graph.add_node(node,attr_name = attr)
    
    def add_nodes_attribure(self,attr_dict):
        for node,attr in attr_dict.items():
            self.add_node_attribute(node,attr)

    def add_edge_attribute(self,*edge,attr,attr_name='attribute'):
        self.graph.add_edge(*edge,attr_name=attr)

    def add_edges_attribute(self,attr_dict):
        for edge,attr in attr_dict:
            self.graph.add_edge(*edge,attr)

    def get_node_attributes(self,node):
        return self.graph.nodes[node]

    def get_edge_attributes(self,edge):#edge is a tuple
        return self.graph.edges[edge]

    # def get_nodes_attributes(self):
    #     return self.graph.nodes