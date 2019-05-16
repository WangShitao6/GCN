#PSCN
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input,Dense,Conv1D,Conv2D,Add,add,Dropout,Flatten,MaxPooling2D
from tensorflow.keras.initializers import RandomNormal,RandomUniform,Constant
from tensorflow.keras import backend as back
from tensorflow.keras.optimizers import SGD
import numpy as np
import time
from keras.wrappers.scikit_learn import KerasClassifier
from AcceptiveFieldMaker import Maker

class PSCN:

    def __init__(self,
                 width,#the length of nodes sequence
                 category,#the number of all example category
                 data_processor,#process node and edge attributes
                 k_size=5,#acceptive field size
                 strides=1,#the traverse on nodes sequence
                 conv1D_output1=10,
                 conv1D_output2=10,
                 batch_size=32,
                 epochs=10,
                 verbose=2,
                 label='pagerank',#['pagerank','betweenness_centrality']
                 ):

        self.batch_size=batch_size
        self.width=width
        self.strides=strides
        self.k=k_size
        self.category=category
        self.conv1D_output1=conv1D_output1
        self.conv1D_output2=conv1D_output2
        self.data_processor=data_processor
        self.node_attrs_num = self.data_processor.get_node_attrs_num()
        self.edge_attrs_num = self.data_processor.get_edge_attrs_num()
        self.epochs=epochs
        self.verbose=verbose
        self.model_shape = self.pscn_model()
        self.model = KerasClassifier(build_fn=self.pscn_model,
                                     epochs=self.epochs, 
                                     batch_size=self.batch_size, verbose=self.verbose)
        
        self.label=label
        pass
    def __repr__(self):
        return '''class PSCN:
        width size:{}
        kernel size:{}
        category num:{}
        conv1D output depth:{}
        conv2D output depth:{}
        batch size:{}
        epochs:{}
        verbose:{}
        '''.format(self.width,self.k,self.category,self.conv1D_output1,self.conv1D_output2,self.batch_size,self.epochs,self.verbose)

    def __del__(self):
        '''
        将所有属性占用内存释放
        '''
        pass

    def pscn_model(self,):
        node_input = Input(shape=(self.width*self.k,self.node_attrs_num),name='node')
        edge_input = Input(shape=(self.width*self.k*self.k,self.edge_attrs_num),name='edge')

        conv1d_node=Conv1D(filters=self.conv1D_output1,
                           kernel_size=self.k,
                           strides=self.k,
                           #kernel_initializer=RandomNormal(mean=0.0,stddev=0.05,seed=None)
                           )(node_input)
        
        conv1d_edge=Conv1D(filters=self.conv1D_output1,
                           kernel_size=self.k*self.k,
                           strides=self.k*self.k,
                           #kernel_initializer=RandomNormal(mean=0.0,stddev=0.05,seed=None)
                           )(edge_input)

        #print('node:{};edge:{}'.format(conv1d_node,conv1d_edge))

        merge = add([conv1d_node,conv1d_edge],name='merge')

        #print('merge:{}'.format(merge))

        conv2 = Conv1D(filters = self.conv1D_output2,
                       kernel_size=10,
                       #kernel_initializer=RandomNormal(mean=0.0,stddev=0.05,seed=None),
                       #use_bias=True,
                       #bias_initializer=Constant(value=1.0),
                       strides=1,
                       #padding='valid'
                       )(merge)

        #max_pool = MaxPooling2D(pooling_size=(2,2))(conv2)

        flatten=Flatten()(conv2)
        dense1 = Dense(128,activation='relu')(flatten)
        dropout = Dropout(0.5)(dense1)
        output = Dense(self.category,activation='softmax')(dropout)
        model = Model(inputs=[node_input,edge_input],outputs=[output])
        model.compile(loss="sparse_categorical_crossentropy",optimizer="rmsprop",metrics=["accuracy"])
        return model

    def create_model(self):
        model=Sequential()
        model.add(Conv1D(filters=16,kernel_size=self.k,strides=self.k,input_shape=(self.width*self.k,self.node_attrs_num)))    
        model.add(Conv1D(filters=8,kernel_size=10,strides=1))
        model.add(Flatten())
        model.add(Dense(128,activation="relu",name='embedding_layer'))
        model.add(Dropout(0.5))
        model.add(Dense(1,activation="sigmoid"))
        model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=["accuracy"])
        return model

    def prediction(self,x):
        return self.model.predict(x).ravel()

    def train(self,x,y):
        #self.model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=["accuracy"])
        return self.model.fit(x,y,validation_split=0.25,verbose=1,shuffle=True)

    def data_generator(self,data,):
        start = time.time()

        graph_list,label = zip(*data)
        data_train=list()
        label_train=list()
        for index,graph in enumerate(graph_list):
            make = Maker(graph=graph,
                         data_process=self.data_processor,
                         width=self.width,
                         k_size=self.k,
                         strides=self.strides,
                         label=self.label,)
            node_example,edge_example = make.select_node_sequence()
            data_train.append((node_example,edge_example))

            label_train.append(label[index])#要把他转换成one_hot编码
        end = time.time()
        print('data_generator cost time:%5fS'%(end-start))
        return data_train,label_train
        


