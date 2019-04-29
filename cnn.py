import tensorflow as tf
import os
from data_processor import data_process
#create two convolution max pooling activation function
#parameters:
'''
kernel_size:the size ofconvolution kernel
kernel_num:the number of convolution kernel
padding:default is "same" ,"valid"
use_bias:default is "True"
activation function:relu
'''
class cnn_2layers:
    
    def __init__(self,
                train,#train dataset
                test,#test dataset
                batch,height,width,channels):#data tensor

        self.train_data=train
        self.test_data=test
        self.height=height
        self.width=width
        self.batch=batch
        self.channels=channels

    def weight_variable(shape):
        initial = tf.truncated_normal(shape,stddev=0.1)#create a tensor,size is shape;value stddev
        return tf.Variable(initial)
    
    def bias_variable(shape):
        initial = tf.constant(0.1,shape=shape)#create a constant value tensor
        return tf.Variable(initial)

    def conv2d(x,w):#input format(batch,height,width,channels)
        return tf.cnn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

    def model(data_batch,label_batch):
        # if os.path.exists("./dataset"):#read parameter from file
        #     pass
        # else:
        #     pass

        #dropout
        keep_prob = tf.placeholder(tf.float32)

        weight_conv1 = self.weight_variable([5,5,1,16])
        beias_conv1 = self.bias_variable([16])

        wight_conv2 = self.weight_variable([5,5,16,32])
        bias_conv2 = self.bias_variable([32])
        

        #convolution layer 1
        result_conv1 = conv2d(data,weight_conv1)+bias_conv1
        result_pool1 = max_pool_2x2(tf.nn.relu(result_conv1))
        #convolution layer 2
        result_conv2 = conv2d(result_pool1,weight_conv2)+bias_conv2
        result_pool2 = max_pool_2x2(tf.nn.relu(result_conv2))
        #fully connected layer

        
        #calculate the input tensor size
        height=(self.height-1)//4+1
        width=(self.wdith-1)//4+1


        weight_fc1 = self.weight_variable([height*wdith*32],1024)
        bias_fc1 = self.self.bias_variable([1024])
        result_pool2_flat = tf.reshape(result_pool2,[-1,height*width*32])

        result_fc1 = tf.nn.relu(tf.matul(result_pool2_flat,weight_fc1)+bias_fc1)
        result_fc1_drop  = tf.nn.dropout(result_fc1,keep_prob)

        weight_fc2 = self.weight_variable([1024,7])
        bias_fc2 = self.bias_variable([7])
        prediction = tf.nn.softmax(tf.matul(result_fc1_drop,weight_fc2)+bias_fc2)
        #loss
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(label*tf.log(prediction),reduction_indices=[1]))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        for in range(1000):
            data_batch
        