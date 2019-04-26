
import networkx as nx

class data_process:
    
    def __init__(self,size,attributes_num):
        self.__data_path = "~/dataset/cora"
        self.__aceptive_field_list = list()
        self.__aceptive_field_size = size
        self.attributes_num = attributes_num
        
    def get_graph_dataset(self):
        graph = nx.DiGraph()
        edges = list()
        with open(self.__data_path+"cora.cites",'r') as f:
            for line in f.readlines():
                line = line.split()
                edges.append(line[1],line[0])
        graph.add_edges_from(edges)
        return graph

    def get_nodes_attribute_dataset(self):
        nodes_attribute = dict()
        with open(self.__data_path+"cora.content",'r') as f:
            for line in f.readlines():
                line = line.split()
                #create a nodes_attribute_dict for nx.set_node_attributes()
                #index 0 denote node and index len(line)-1 denote the category of node;
                #index 1~len(line)-2 denote the attributes of node
                nodes_attribute[line[0]] = {i:line[i] for i in range(1,len(line)-1)}
                nodes_attribute[line[0]]['category'] = line[len(line)-1]
        return nodes_attribute

    def set_nodes_attribute_graph(self,graph,nodes_attribute):
        attr_keys = graph.nodes(data=True)[0][1].keys()
        attr_dict = dict()
        #graph.nodes(data=True) will return a list,the elemt of list is tuple
        #tuple[0] is node;tuple[1] is node's attr_dict
        for node in list(graph.nodes()):
            if "f" in node:
                attr_dict[node] = {x:0 for x in attr_keys}
            else:
                attr_dict[node] = nodes_attribute[node]
        nx.set_node_attributes(graph,attr_dict)
        return graph

        