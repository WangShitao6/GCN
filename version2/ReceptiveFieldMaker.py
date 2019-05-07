from nauty.graph import Graph,canonical_labeling

class Maker:
    
    def __init__(self,
                graph,
                width,#the number of choicing
                k_size,#acceptive field size
                strides,#travrse strides
                one_hot=False,#one_hot encode
                fake_value=-1,#
                label = 'pagerank',#sort nodes,choice 'width' number node
                ):
        self.graph = graph
        self.width = width
        self.k_size = k_size
        self.strides = strides
        self.one_hot = one_hot
        if label=='pagerank':
            self.initial_label = self.pagerank_label_produce(self.graph)#waiting
        elif label == 'betweenness_centrality':
            self.initial_label = self.betweenness_label_produce(self.graph)#waiting
        else:
            pass


    def select_node_sequence(self):
        node_sequence = self.initial_label[0:self.k_size]
        train_list = list()
        i = 0
        j = 0
        while j<self.width:
            if i<len(node_sequence):
                acceptive_field = self.receptive_field(node_sequence[i])
            else:
                acceptive_field = self.zero_receptive_field()#waiting
            train_example = self.apply_to_input_channels(acceptive_field)#waiting
            train_list.append(acceptive_field)
            i+=self.strides
            j+=1
        return train_list

    def zero_receptive_field(self,):
        graph = nx.Graph()
        for i in range(self.k_size):
            graph.add_node('f'+str(i))
        return graph

    def receptive_field(self,node):
        neighbors = self.neighbors_assemb(node)
        norm_graph = self.normalize_graph(neighbors,node)
        return norm_graph

    def apply_to_input_channels(self,)


    def neighbors_assemb(self,node):
        N = {node}
        L = {node}
        while len(N)<self.k_size and len(L)>0:
            tmp = set()
            for item in L:
                tmp = tmp|self.graph.neighbors(node)
            L = tmp - N
            N = N|L
        return N

    def normalize_graph(self,neighbors,node):
        rank_r,coloring = self.labeling_produce(neighbors,node)#waiting
        fake_node_num = 0
        if len(neighbors)>self.k_size:
            N = rank_r[0:self.k_size]
            rank_r,coloring = self.labeling_produce(N,node)#waiting
        elif len(neighbors)<self.k_size:
            fake_node_num = self.k_size-len(neighbors)
            N,coloring = self.add_fake_node(neighbors,coloring,fake_node_num)#waiting
        else:
            N = neighbors
        graph_n = self.construct_subgraph(N,fake_node_num)#waiting
        canonical_graph = self.canonicalize(graph_n,coloring)#waiting
        return canonical_graph
'''
在canonical过程中可以将label的划分结果传入，然后作label。
'''
    def labeling_produce(self,neighbors,node):
        coloring=list()
        level_coloring = list()
        subgraph = self.graph.subgraph(neighbors)
        if self.label=='pagerank':
            init_dict = nx.pagerank(subgraph,alpha=0.85)
        else:
            init_dict = nx.betweenness_centrality(subgraph)
        level_dict = nx.single_source_dijkstra_path_length(subgraph,node)
      
        for value in set(level_dict.value()):#set自动从小到大排序
            level_coloring.append({node for node in level_dict.keys() if level_dict[node]==value})
        for color in level_coloring:
            value_set = set()
            for node in color:
                value_set.add(-init_dict[node])
            for value in value_set:#里面的值都是负的，也就是原本值是从大到小排序
                coloring.append(node for node in color if init_dict[node]==-value)

        label = sorted(neighbors,key=lambda item:(level_dict[item],-init_dict[item]))
        return label,coloring

    def pagerank_label_produce(self,graph):
        label = nx.pagerank(graph)
        label = sorted(label.keys(),key=lambda item:-label[item])
        return label

    def betweenness_label_produce(self,graph):
        label = nx.betweenness_centrality(graph)
        label = sorted(label.keys(),key=lambda item:-label[item])
        return label

    def add_fake_node(self,neighbors,coloring,number):
        fake_set = set()
        for index in range(number):
            fake_node_name = 'f'+str(index)
            neighbors.append(fake_node_name)
            fake_set.add(fake_node_name)
        coloring.append(fake_set)
        return neighbors,coloring

    def construct_subgraph(self,N,fake_node_num):#需要改进的地方，要将所有的属性输入留在接受域构建完成后进行，在load data里面改返回的数据内容
        subgraph = self.graph.subgraph(N[0:len(N)-fake_node_num])
        for node in N[len(N)-fake_node_num:len(N)]:
            subgraph.add_node(node)
        return subgraph

    def canonicalize(self,graph,coloring):
        nodes = list(graph.nodes)

        graph = nx.convert_node_labels_to_integers(graph)

        relabel_coloring = list()
        for color in coloring:
            tmp = set()
            for node in color:
                tmp.add(nodes.index(node))
            relabel_coloring.append(tmp)
    
        nauty_graph=Graph(number_of_vertices=len(graph.nodes),
                          directed=False,
                          adjacency_dict={k:list(ndict) for k,ndict in graph.adjacency()},
                          vertex_coloring=relabel_coloring,)

        nauty_label=canonical_labeling(nauty_graph)#列表里面是有序的节点序列
        relabel_dict={key:nodes[nauty_label[key]] for key in range(len(nauty_label))}
        graph = nx.relabel_nodes(graph,relabel_dict)
        return graph
        

        


        



    