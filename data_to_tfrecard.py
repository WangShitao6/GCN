#in this moudle,acceptive_field will be builded to tfrecard.
import networkx as nx
import tensorflow as tf
import numpy as np
from acceptive_field_maker import acceptive_field_maker as maker
from data_processor import data_process 

data_path = '/home/wangshitao/file/dataset/cora/'
category_mapping = {'Neural_Networks':0,
                    'Rule_Learning':1,
                    'Reinforcement_Learning':2,
                    'Probabilistic_Methods':3,
                    'Theory':4,
                    'Genetic_Algorithms':5,
                    'Case_Based':6}
node_attributes_num = 1433

data = data_process(data_path,node_attributes_num,category_mapping)
all_nodes_attrs = data.get_nodes_attributes_dataset()

graph = data.get_graph_dataset()
nodes = list(graph.nodes())

acceptive_maker = maker(graph,10)

#--------------------------------------------------------------
field = acceptive_maker.create_acceptive_field(nodes[0])
#field = data.set_nodes_attributes_graph(field,all_nodes_attrs)
#should create node tensor and edge tensor
#assume two methods
#one:graph = nx.relabel_nodes() then create a new nx.DiGraph() and the adegs is graph.edges()
mapping = nx.get_node_attributes(field,'label')
relabel_field = nx.relabel_nodes(field,mapping)
mapping = sorted(mapping,key =lambda item:mapping[item])

node_tensor = list()
edge_tensor = np.zeros((acceptive_maker.size,acceptive_maker.size))
for node in mapping:
    if 'f' in node:
        node_tensor.append([0]*node_attributes_num)
    else:
        node_tensor.append([all_nodes_attrs[node][x] for x in range(1,node_attributes_num+1)])

for edge in list(relabel_field.edges()):
    edge_tensor[edge[0]][edge[1]] = 1

nodes_list=list()
edges_list = list()
category_label = list()
nodes_list.append(np.array(node_tensor))
a = np.array(node_tensor)
print(a.shape)
edges_list.append(edge_tensor)
category_label.append(all_nodes_attrs[nodes[0]]['category'])
#--------------------------------------------------------------------------





def _int64_feature(value):  
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def save_tfrecords(node,edge,label,desfile):
    with tf.python_io.TFRecordWriter(desfile) as writer:
        for i in range(len(label)):
            features = tf.train.Features(
                feature = {
                    "node":_bytes_feature(value=node[i].astype(np.float64).tostring()),
                    "edge":_bytes_feature(value=edge[i].astype(np.float64).tostring()),
                    "label":_int64_feature(value=label[i])
                }
            )
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())

desfile = "./data.tfrecords"

save_tfrecords(nodes_list,edges_list,category_label,desfile)