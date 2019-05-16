from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,Conv1D,Conv2D,Add,add,Dropout,Flatten,MaxPooling2D
from tensorflow.keras.initializers import RandomNormal,RandomUniform,Constant
from tensorflow.keras import backend as back
from keras.optimizers import SGD
from numpy as np

width = 5
k_size = 10
attr_dim1 = 6
attr_dim2 = 3

# n = Input(shape=(width*k_size,attr_dim1))
# e = Input(shape=(width*k_size*k_size,attr_dim2))
# node_1 = Conv1D(filters=5,kernel_size=k_size,strides=k_size)
# node = node_1(n)

# edge = Conv1D(filters=5,kernel_size=k_size*k_size,strides=k_size*k_size)(e)

# data = Add()([node,edge])
# mult = Multiply()([node,edge])

print(mult)

def pscn_model(width,k,category):
    node_input = Input(shape=(width*k,-1),name='node_input',dtype=float)
    edge_input = Input(shape=(width*k*k,-1),name='edge_input',dtype=float)
    conv1_output = 5
    conv2_output = 10
    conv1_node=Conv1D(filters=conv1_output,
                        kernel_size=k,strides=k,
                        kernel_initializer=RandomNormal(mean=0.0,stddev=0.05,seed=None))(node_input)

    conv1_edge=Conv1D(filters=conv1_output,
                        kernel_size=k*k,
                        strides=k*k,
                        kernel_initializer=RandomNormal(mean=0.0,stddev=0.05,seed=None))(edge_input)

    merge = add([conv1_node,conv1_edge])

    conv2 = Conv2D(filters=conv2_output,
                    kernel_size=[3,3],
                    kernel_initializer=RandomNormal(mean=0.0,stddev=0.05,seed=None),
                    use_bias=True,
                    bias_initializer=Constant(value=1.0)
                    strides=[1,1],
                    padding='valid')(merge)

    max_pool = MaxPooling2D(pooling_size=(2,2))(conv2)

    flatten = Flatten()(max_pool)

    dense1 = Dense(100,activation='relu')(flatten)
    dense2 = Dense(category,activation='softmax')(dense1)
    output = dense2()(dense1)
    model = Model(inputs=[node_input,edge_input],outputs=[output])

def exe(model,node_data,edge_data,label,width,k,category):#label应该是one_hot编码的

    model.compile(optimizer='rmsprop',metrics=['accuracy'])
    model.fit(x={'node_input':node_data,'edge_input':edge_data},y=label)









