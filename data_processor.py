import networkx as nx
import tensorflow as tf
import numpy as np
from acceptive_field_maker import acceptive_field_maker
from data_processor import data_process 

class data_process:

    def __init__(self,
                attr_num,#the node attributes number
                category_dict,#key:category,value:a int values and must be continuous
                size#which size do you want to create for acceptive field
                ):
        self.attr_num = attr_num
        self.category = category_dict
        self.size = size
        
    def get_graph_dataset(self,src_data_path):
        graph = nx.DiGraph()
        edges = list()
        with open(src_data_path+"cora.cites",'r') as f:
            for line in f.readlines():
                line = line.split()
                edges.append((line[1],line[0]))
        graph.add_edges_from(edges)
        return graph

    def get_nodes_attributes_dataset(self):
        nodes_attribute = dict()
        with open(src_data_path+"cora.content",'r') as f:
            for line in f.readlines():
                line = line.split()
                #create a nodes_attribute_dict for nx.set_node_attributes()
                #index 0 denote node and index len(line)-1 denote the category of node;
                #index 1~len(line)-2 denote the attributes of node
                nodes_attribute[line[0]] = {i:line[i] for i in range(1,len(line)-1)}
                nodes_attribute[line[0]]['category'] = self.category[line[len(line)-1]]
        return nodes_attribute


    def _bytes_feature(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def get_format_data(self,graph,node):
        '''
        get the node's acceptive field
        build node,edge,lable np.array() 
        which will be saved as tfrecords
        '''
        all_nodes_attributes = self.get_nodes_attributes_dataset()
        maker = acceptive_field_maker(graph,self.size)
        field = maker.create_acceptive_field(node)

        node_order_mapping = nx.get_node_attributes(field,'label')
        relabel_field = nx.relabel_nodes(field,node_order_mapping)

        node_order = sorted(node_order_mapping,key = lambda Node:node_order_mapping[Node])

        node_tensor = list()
        edge_tensor = np.zeros(self.size,self.size)

        for Node in node_order:
            if 'f' in Node:
                node_tensor.append([0]*self.attr_num)
            else:
                node_tensor.append([all_nodes_attrs[Node][x] for x in range(1,node_attributes_num+1)])
            
        for edge in list(relabel_field.edges()):
            edge_tensor[edge[0]][edge[1]] = 1
        
        node_tensor = np.array(node_tensor)

        label_tensor = np.zeros(7)
        label_tensor[all_nodes_attributes[node]['category']] = 1

        return node_tensor,edge_tensor,label_tensor


    def save_tfrecords(self,desfile):
        try:
            f = open(desfile,'rw')
            f.close()
        except FileNotFoundError:
            print("File is not found.")
            return False
        except PersmissionError:
            print("You don't have permission to access this file.")
            return False

        order=0
        graph = self.get_graph_dataset()
        with tf.python_io.TFRecordWriter(desfile) as writer:
            for Node in list(graph.nodes()):
                node,edge,label = self.get_format_data(graph,Node)
                features = tf.train.Features(
                    feature = {
                        "node":_bytes_feature(value=node.astype(np.float64).tostring()),
                        "edge":_bytes_feature(value=edge.astype(np.float64).tostring()),
                        "label":_bytes_feature(value=label.astype(np.float64).tostring())
                    }
                )
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())
            order+=1
            if order%200==0:
                print('=>')
            else:
                pass
        print('complete\n%s files were saved successfully in%s'%(order,desfile))
        return True

    def _parse_fuction(self,example):
        features = {"node":tf.FixedLenFeature((),tf.string),
                "edge":tf.FixedLenFeature((),tf.string),
                "label":tf.FixedLenFeature((),tf.string)}
        parse_features = tf.parse_single_example(example,features)

        node = tf.decode_raw(parse_features["node"],tf.float64)
        node = tf.reshape(node,(self.size,self.attr_num))

        edge = tf.decode_raw(parse_features['edge'],tf.float64)
        edge = tf.reshape(edge,(self.size,self.size))
        #should Stitching node tensor and edge tensor together in column
        data = tf.concat([node,edge],1)

        label = tf.decode_raw(parse_features['label'],tf.float64)
        label = tf.reshape(label,(1,7))
        return data,label

    def load_tfrecords(self,src_tfrecordfiles,batch)
        try:
            f = open(src_tfrecordfiles,'rw')
            f.close()
        except FileNotFoundError:
            print("File is not found.")
            return False
        except PersmissionError:
            print("You don't have permission to access this file.")
            return False

        dataset = tf.data.TFRecordDataset(src_tfrecordfiles)
        dataset = data.map(_parse_fuction)
        dataset = dataset.batch(batch)

        return dataset