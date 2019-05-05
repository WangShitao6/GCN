#load dataset
from collections import defaultdict
import os
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
        for index,line in enumerate(f):
            graph_label.append((index+1,int(line.split()[0])))
    return graph_label


def get_adjancy(path,name):
    adjancy = defaultdict(list)
    with open(path+name,'r') as f:
        for line in f:
            adjancy[int(line.split(',')[0])].append(int(line.split(',')[1]))
    return adjancy

def get_node_dict(path,name):
    node_dict = dict()
    with open(path+name,'r') as f:
        for index,line in enumerate(f):
            node[index+1]=int(line.split()[0])
    return node_dict

def get_graph_indicator(path,name):
    indecator = defaultdict(list)
    with open(path+name,'r') as f:
        for index,line in enumerate(f):
            indecator[int(line.split()[0])].append(index+1)
    return indecator

path = os.getcwd()+'/data/MUTAG_2/'



