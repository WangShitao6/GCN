#测试ReceptiveFieldMaker.py 和 data_preprocess.py
from load_dataset import *
from data_preprocess import *
from AcceptiveFieldMaker import *
import numpy as np
from sys import getsizeof
from PSCN_model import PSCN

data,nodes_attrs,edges_attrs = load_dataset('./data','mutag')

graph,label = zip(*data)



data_processor = data_preprocess(nodes_attrs,edges_attrs,one_hot=True,fake_value=-1)#careful

width = 18
k = 5
strides = 1

# for item in graph[0:1]:
#     make = Maker(item,data_processor,width,k,strides)
#     node,edge = make.select_node_sequence()
#     node_array = np.array(node)
#     edge_array = np.array(edge)

#     print('node array shape:{} ;memery:{}'.format(node_array.shape,getsizeof(node_array)))
#     print('edge array shape:{} ;memery:{}'.format(edge_array.shape,getsizeof(edge_array)))
#     print(188*(getsizeof(node_array)+getsizeof(edge_array))/(1024*1024))

#     print(make)

# for index,G in enumerate(graph):
#     print('index:{}\ngraph:{}\n'.format(label[index],G))

pscn=PSCN(data_processor=data_processor,width=width,category=2,k_size=k)
print(pscn)
