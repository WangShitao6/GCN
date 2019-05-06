#load dataset
from collections import defaultdict
import os
from graph import Graph
import numpy as np


def load_dataset(path,name,one_hot=False):
    if name=="mutag":
        dataset = create_mutag_data(path,one_hot)
    return dataset


def create_mutag_data(path,one_hot):
    graph_label = get_graph_label(path,'MUTAG_graph_labels.txt')#return a tuple (graph name,label )
    adjancy = get_adjancy(path,'MUTAG_A.txt')
    #edges = get_edges(path,'MUTAG_A.txt')
    node_dict = get_nodes_dict(path,'MUTAG_node_labels.txt')
    edge_dict = get_edges_dict(path,'MUTAG_edge_labels.txt')
    graph_indicator = get_graph_indicator(path,'MUTAG_graph_indicator.txt')
    train = list()
    for item in graph_label:
        #print('current is fisrt')
        g = Graph()
        g.name = item[0]
        for node in graph_indicator[item[0]]:
            #print('current is second')
            g.add_node(node)
            if one_hot:
                attr = convert_to_one_hot(node_dict[node],7)
                g.add_node_attribute(node,attr)
            else:
                g.add_node_attribute(node,node_dict[node])
            for target in adjancy[node]:
                if one_hot:
                    attr = convert_to_one_hot(edge_dict[(node,target)],4)
                else:
                    attr = edge_dict[(node,target)]
                g.add_edge(node,target)
                g.add_edge_attribute(node,target,attr=attr)
        train.append((g,item[1]))
    return train


def convert_to_one_hot(source,category,label_fake=-1):
    if source == label_fake:
        return np.zeros(category)
    else:
        one_hot = np.zeros(num)
        one_hot[source]=1
        return one_hot


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

def get_edges(path,name):
    edge = list()
    with open(path+name,'r') as f:
        for line in f:
            edge.append((int(line.split(',')[0]),int(line.split(',')[1])))
    return edge

def get_nodes_dict(path,name):
    node_dict = dict()
    with open(path+name,'r') as f:
        for index,line in enumerate(f):
            node_dict[index+1]=int(line.split()[0])
    return node_dict

def get_graph_indicator(path,name):
    indecator = defaultdict(list)
    with open(path+name,'r') as f:
        for index,line in enumerate(f):
            indecator[int(line.split()[0])].append(index+1)
    return indecator

def get_edges_dict(path,name):
    edge_dict = dict()
    edges = get_edges(path,'MUTAG_A.txt')
    with open(path+name,'r') as f:
        for index,line in enumerate(f):
            edge_dict[edges[index]]=int(line.split()[0])
    return edge_dict

path = os.getcwd()+'/data/MUTAG_2/'
test = load_dataset(path,name = 'mutag',one_hot=False)
#test = get_graph_indicator(path,'MUTAG_graph_indicator.txt')
#test = get_graph_label(path,'MUTAG_graph_labels.txt')#return a tuple (graph name,label )
#test = get_adjancy(path,'MUTAG_A.txt')
#test = get_edges(path,'MUTAG_A.txt')
#test = get_nodes_dict(path,'MUTAG_node_labels.txt')
#test = get_edges_dict(path,'MUTAG_edge_labels.txt')
#graph_indicator = get_graph_indicator(path,'MUTAG_graph_indicator.txt')
print(test[0][1])

