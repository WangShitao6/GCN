from load_dataset import *
from data_preprocess import *
from AcceptiveFieldMaker import *
import numpy as np
from sys import getsizeof
from PSCN_model import PSCN
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import plot_model


graph_data,nodes_attrs,edges_attrs = load_dataset('./data','mutag')
data_processor = data_preprocess(nodes_attrs,edges_attrs,one_hot=False,fake_value=-1)
pscn = PSCN(data_processor=data_processor,width=18,category=2,k_size=10,epochs=100,conv1D_output1=16,conv1D_output2=8)

data,label = pscn.data_generator(graph_data)
node,edge = zip(*data)


x_train,x_test,y_train,y_test = train_test_split(data,label,test_size=0.33, random_state=42)
node_train,edge_train = zip(*x_train)
node_test,edge_test = zip(*x_test)
print('x train:{}\nx test:{}\ny train:{}\ny test:{}'.format(len(x_train),len(x_test),len(y_train),len(y_test)))
print('node train:{}\nedge train:{}'.format(len(node_train),len(edge_train)))
print(node_train[0].shape)
print(edge_train[0].shape)

#for index,graph in enumerate()
history = pscn.train([node,edge],label)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Test'],loc='upper left')
plt.show()
#plot_model(pscn.model_shape,to_file='/mnt/hgfs/root/model.png',show_shapes=True)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('Model loss')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Test'],loc='upper left')
plt.show()


#print('result:',result)
preds=pscn.prediction([node_test,edge_test])
# print(preds)
# print(y_test)
print(np.sum(preds==y_test)/len(y_test))




