#in this moudle,acceptive_field will be builded to tfrec0rd.
import networkx as nx
import tensorflow as tf
import numpy as np
from acceptive_field_maker import acceptive_field_maker as maker
from data_processor import data_process 
import os

corrent_path = os.getcwd()
parent_path = os.path.abspath(os.path.dirname(corrent_path))
print(parent_path)
data_path = parent_path+"/dataset/cora/"
tfrecords_path='./data.tfrecords'
category_mapping = {'Neural_Networks':0,
                    'Rule_Learning':1,
                    'Reinforcement_Learning':2,
                    'Probabilistic_Methods':3,
                    'Theory':4,
                    'Genetic_Algorithms':5,
                    'Case_Based':6}
node_attributes_num = 1433
field_size = 8

processor = data_process(data_path=data_path,attr_num=node_attributes_num,category_dict=category_mapping,size=field_size)
print("create processor success")
processor.save_tfrecords(tfrecords_path)


# data = data_process(data_path,node_attributes_num,category_mapping)

# all_nodes_attrs = data.get_nodes_attributes_dataset()

# graph = data.get_graph_dataset()

# nodes = list(graph.nodes())
# def get_node_data(graph,node):
#     #print("%s tart make field"%nodes[i])
#     #--------------------------------------------------------------
#     acceptive_maker = maker(graph,12)
#     field = acceptive_maker.create_acceptive_field(node)
#     # print("field make success")
#     # print(field.nodes())
#     #field = data.set_nodes_attributes_graph(field,all_nodes_attrs)
#     #should create node tensor and edge tensor
#     #assume two methods
#     #one:graph = nx.relabel_nodes() then create a new nx.DiGraph() and the adegs is graph.edges()
#     mapping = nx.get_node_attributes(field,'label')
#     relabel_field = nx.relabel_nodes(field,mapping)
#     mapping = sorted(mapping,key =lambda item:mapping[item])

#     node_tensor = list()
#     edge_tensor = np.zeros((acceptive_maker.size,acceptive_maker.size))
#     for Node in mapping:
#         if 'f' in Node:
#             node_tensor.append([0]*node_attributes_num)
#         else:
#             node_tensor.append([all_nodes_attrs[Node][x] for x in range(1,node_attributes_num+1)])

#     for edge in list(relabel_field.edges()):
#         edge_tensor[edge[0]][edge[1]] = 1

#     node_tensor=np.array(node_tensor)
#     #print(a.shape)

#     label_tensor = np.zeros(7)
#     label_tensor[all_nodes_attrs[node]['category']]=1
#     #print(array)
#     return node_tensor,edge_tensor,label_tensor
# #--------------------------------------------------------------------------

# def _int64_feature(value):  
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# def _bytes_feature(value):
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# def _float_feature(value):
#     return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

# def save_tfrecords(graph,desfile):
#     order = 1
#     with tf.python_io.TFRecordWriter(desfile) as writer:
#         for Node in list(graph.nodes()):
#             node,edge,label = get_node_data(graph,Node)
#             features = tf.train.Features(
#                 feature = {
#                     "node":_bytes_feature(value=node.astype(np.float64).tostring()),
#                     "edge":_bytes_feature(value=edge.astype(np.float64).tostring()),
#                     "label":_bytes_feature(value=label.astype(np.float64).tostring())
#                 }
#             )
#             example = tf.train.Example(features=features)
#             writer.write(example.SerializeToString())
#             order+=1
#             print(order,":",Node," is done",end='\n')

# desfile = "./data.tfrecords"

# save_tfrecords(graph,desfile)