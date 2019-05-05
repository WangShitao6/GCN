#load dataset
def load_dataset(path,name,one_hot=False):
    if name=="mutag":
        dataset = create_mutag(path,one_hot)


    return dataset


def create_mutag(path,one_hot):
    graph_label = get_graph_label(path,'MUTAG_graph_labels.txt')
    adjancy = get_adjancy(path,'MUTAG.txt')
    node_dict = get_node_dict(path,'MUTAG_node_labels.txt')
    edge_dict = get_edge_dict(path,'MUTAG_edge_labels.txt')
    indicator = get_graph_indicator(path,'MUTAG_graph_indicator.txt')
    pass


def get_graph_label(path,name):
    graph_label = list()
    with open(path+name,'r') as f:
        graph_label = list(filter(lambda x:x.split() if x in f.readlines()))

    return graph_label

get_graph_label('./data','MUTAG_graph_labels.txt')
