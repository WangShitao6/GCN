#作为数据预处理的，将所有构建出的接受域应用在输入通道上去。
from sklearn import preprocessing
class data_preprocess:

    def __init__(self,nodes_attrs,edges_attrs,one_hot=False,fake_value=0):#这里的节点和边的属性是所有图的
    #属性的格式:{node1:[attr_list],...,noden:[attr_list]}
    #属性的格式:{edge1:[attr_list],...,edgen:[attr_list]}
        
        if one_hot:
            self.nodes_attrs = self.convert_to_one_hot(nodes_attrs)#waiting
            self.nodes_attrs['fake']=[0]*self.nodes_attrs.values()[0].shape[1]

            self.edges_attrs = self.convert_to_one_hot(edges_attrs)#waiting
            self.edges_attrs['no_edge']=[0]*self.edges_attrs.values()[0].shape[1]
        else:
            nodes_attrs['fake']=[fake_value]*len(list(nodes_attrs.values())[0])#这里可以做一个异常处理，如果传进来的属性格式不对,或者字典为空
            edges_attrs['no_edge']=[fake_value]*len(list(edges_attrs.values())[0])

            self.nodes_attrs = nodes_attrs
            self.edges_attrs = edges_attrs
        pass


    def convert_to_one_hot(self,attrs):#可以改进的地方：可以以字符串的形式存储读取到的属性，然后通过LabelEncoder转换类型，然后在进行one_hot编码
        one_hot = preprocessing.OneHotEncoder(categories='auto',sparse=False)
        values = one_hot.fit_transform(list(attrs.values()))
        for index,key in enumerate(attr.keys()):
            attrs[key]=values[index]
        return attrs

    def get_node_edge_tensor(self,graph):
        nodes_list = list(graph.nodes)

        node_tensor = list()
        edge_tensor = list()
        for node in nodes_list:
            if 'f' in node:
                node_tensor.append(self.nodes_attrs['fake'])
                edge_tensor.append([self.edges_attrs['no_edge'] for i in range(len(nodes_list))])
            else:
                node_tensor.append(self.nodes_attrs[node])
                node_edge_attr_list=list()
                for target in nodes_list:#这个地方可以优化，先把每个graph的属性拿出来，在进行判断
                    if (node,target) in self.edges_attrs.keys():
                        node_edge_attr_list.append(self.edges_attrs[(node,target)])
                    else:
                        node_edge_attr_list.append(self.edges_attrs['no_edge'])
                edge_tensor.append(node_edge_attr_list)
        return node_tensor,edge_tensor

    