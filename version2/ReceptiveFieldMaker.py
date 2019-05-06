from nauty.graph import Graph,canonical_labeling

class Maker:
    
    def __init__(self,
                graph,
                width,#the number of choicing
                k_size,#acceptive field size
                strides,#travrse strides
                one_hot=False,#one_hot encode
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
        train_example = list()
        i = 0
        j = 0
        while j<self.width:
            if i<len(node_sequence):
                acceptive_field = self.receptive_field(node_sequence[i])
            else:
                acceptive_field = self.zero_receptive_field()
            train_example.appen(acceptive_field)
            i+=self.strides
            j+=1
        return train_example

    def receptive_field(self,node):
        neighbors = self.neighbors_assemb(node)
        norm_graph = self.normalize_graph(neighbors,node)
        return norm_graph


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
        rank_r,_ = self.labeling_produce(neighbors,node)#waiting
        fake_node = False
        if len(neighbors)>self.k_size:
            N = sorted(neighbors,key=rank_r)[0:self.k_size]
            rank_r = self.labeling_produce(N,node)#waiting
        elif len(neighbors)<self.k_size:
            N = self.add_fake_node(self.k_size-len(neighbors))#waiting
            fake_node=True
        else:
            N = neighbors
        graph_n = self.construct_subgraph(N,fake_node)#waiting
        canonical_graph = self.canonicalize(graph_n)#waiting
        return canonical_graph
'''
在canonical过程中可以将label的划分结果传入，然后作label。
'''
    def labeling_produce(self,neighbors,node):
        subgraph = self.graph.subgraph(neighbors)
        if self.label=='pagerank':
            label_1 = nx.pagerank(subgraph,alpha=0.85)
        else:
            label_1 = nx.betweenness_centrality(subgraph)

        label_2 = nx.single_source_dijkstra_path_length(subgraph,node)
        label = sorted(neighbors,key=lambda item:(label_2[item],-label_1[item]))
        return label

    def pagerank_label_produce(self,graph):
        label = nx.pagerank(graph)
        label = sorted(label.keys(),key=lambda item:-label[item])
        return label

    def betweenness_label_produce(self,graph):
        label = nx.betweenness_centrality(graph)
        label = sorted(label.keys(),key=lambda item:-label[item])
        return label

    def add_fake_node(self,number):
        


        



    