
import networkx as nx

class data_process:
    
    # def __init__(self,size,attributes_num):
    #     self.__data_path = "~/dataset/cora"
    #     self.__aceptive_field_list = list()
    #     self.__aceptive_field_size = size
    #     self.attributes_num = attributes_num

    def __init__(self,data_path,attr_num,category_dict):
        self.__data_path = data_path
        self.attr_num = attr_num
        self.category = category_dict
        
    def get_graph_dataset(self):
        graph = nx.DiGraph()
        edges = list()
        with open(self.__data_path+"cora.cites",'r') as f:
            for line in f.readlines():
                line = line.split()
                edges.append((line[1],line[0]))
        graph.add_edges_from(edges)
        return graph

    def get_nodes_attributes_dataset(self):
        nodes_attribute = dict()
        with open(self.__data_path+"cora.content",'r') as f:
            for line in f.readlines():
                line = line.split()
                #create a nodes_attribute_dict for nx.set_node_attributes()
                #index 0 denote node and index len(line)-1 denote the category of node;
                #index 1~len(line)-2 denote the attributes of node
                nodes_attribute[line[0]] = {i:line[i] for i in range(1,len(line)-1)}
                nodes_attribute[line[0]]['category'] = self.category[line[len(line)-1]]
        return nodes_attribute

    def set_nodes_attributes_graph(self,graph,nodes_attribute):
        attr_dict = dict()
        #graph.nodes(data=True) will return a list,the elemt of list is tuple
        #tuple[0] is node;tuple[1] is node's attr_dict
        for node in list(graph.nodes()):
            if "f" in node:
                attr_dict[node] = {x:0 for x in range(1,self.attr_num+1)}
            else:
                attr_dict[node] = nodes_attribute[node]
        nx.set_node_attributes(graph,attr_dict)
        return graph

        