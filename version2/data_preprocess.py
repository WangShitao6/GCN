#作为数据预处理的，将所有构建出的接受域应用在输入通道上去。
class data_preprocess:

    def __init__(self,nodes_attrs,edges_attrs,one_hot=False):
        if one_hot:
            self.nodes_attrs = self.convert_to_one_hot(nodes_attrs)#waiting
            self.edges_attrs = self.convert_to_one_hot(edges_attrs)#waiting
        else:
            self.nodes_attrs = nodes_attrs
            self.edges_attrs = edges_attrs
        pass


    def convert_to_one_hot

    def get_node_edge_tensor(self,graph,one_hot=self.one_hot):
        nodes_list = list(graph.nodes)
        node_tensor = list()
        edge_tensor = list()
        for node in nodes_list:
            if 
            node_tensor.append(list(nodes_attrs[node]))
            for target in nodes_list:

                node_edge_attr = list()
                if (node,target) in edges_attrs:
                    node_edge_attr.append()
    