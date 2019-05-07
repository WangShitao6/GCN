#load dataset
from collections import defaultdict
import os
from graph import Graph
import numpy as np


def load_dataset(path,name,one_hot=False):
    if name=="mutag":
        dataset = create_mutag_data(path)
    return dataset


def create_mutag_data(path):#将所有的属性留在接受域构建完成之后进行
    graph_label = get_graph_label(path,'MUTAG_graph_labels.txt')#return a tuple (graph name,label )
    adjancy = get_adjancy(path,'MUTAG_A.txt')
    #edges = get_edges(path,'MUTAG_A.txt')
    #node_dict = get_nodes_dict(path,'MUTAG_node_labels.txt')
    #edge_dict = get_edges_dict(path,'MUTAG_edge_labels.txt')
    graph_indicator = get_graph_indicator(path,'MUTAG_graph_indicator.txt')
    graph_list = list()
    for item in graph_label:
        g = Graph()
        g.name = item[0]
        for node in graph_indicator[item[0]]:
            g.add_node(node)
            for target in adjancy[node]:
                g.add_edge(node,target)
        graph_list.append((g,item[1]))
    return graph_list


def convert_to_one_hot(source,category,fake_value=-1):
    if source == fake_value:
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
            edge_dict[edges[index]]=int(line.split()[0])+1
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

