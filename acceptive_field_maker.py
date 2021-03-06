from pynauty.graph import Graph,canonical_labeling
import networkx as nx

class acceptive_field_maker:

    def __init__(self,graph,size):
        self.graph = graph
        self.size = size

    def assemble_neighbors(self,node):
        node_neighbors = {node}
        new_neighbors = {node}
        while len(node_neighbors)<self.size or len(new_neighbors)!=0):
            tmp = set()
            for node in new_neighbors:
                tmp = tmp|nx.neighbors(self.graph,node)   
                     
            new_neighbors = tmp - node_neighbors
            node_neighbors |= new_neighbors

        return self.graph.subgraph(node_neighbors)

    def distance_labeling_produce(self,subgraph,node):
        G = nx.Graph(subgraph)
        distance_mapping = nx.single_source_shortest_path_length(subgraph,node)
        nx.set_node_attributes(subgraph,distance_mapping,'distance_labeling')

        return subgraph

    def pagerank_labeling_produce(self,subgraph):
        pagerank_mapping = nx.pagerank(subgraph,alpha=0.85)
        nx.set_node_attributes(subgraph,pagerank_mapping,'pagerank_labeling')
        return subgraph
        
    def nauty_labeling_produce(self,subgraph):
        relabel_subgraph = nx.convert_node_labels_to_integers(subgraph)

        nauty_graph = Graph(len(list(relabel_subgraph.nodes)),directed=True)
        nauty_graph.set_adjacency_dict({n:list(ndict) for n,ndict in relabel_subgraph.adjacency()})
        
        nauty_mapping = dict()
        subgraph_nodes = list(subgraph.nodes())
        nauty_mapping_list = canonical_labeling(nauty_graph)
        #key subgraph_nodes[i] is node;value nauty_mapping_list[i] is the order in canonical_labeling
        nauty_mapping = {subgraph_nodes[i]:nauty_mapping_list[i] for i in range(len(nauty_mapping_list))}

        nx.set_node_attributes(subgraph,nauty_mapping,'nauty_labeling')
        return subgraph

    def create_acceptive_field(self,subgraph,node)
