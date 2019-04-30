import tensorflow as tf
import os
#create two convolution max pooling activation function
#parameters:
'''
kernel_size:the size ofconvolution kernel
kernel_num:the number of convolution kernel
padding:default is "same" ,"valid"
use_bias:default is "True"
activation function:relu
'''
class tfrecords_read:
    def __init__(self,size,attr_num):
        self.size = size
        self.attr_num = attr_num

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
        data = tf.reshape(data,(data.shape[0],data.shape[1],1))

        label = tf.decode_raw(parse_features['label'],tf.float64)
        #label = tf.reshape(label,(7))
        return data,label

    def load_tfrecords(self,src_tfrecordfiles):
        try:
            f = open(src_tfrecordfiles,'r')
            f.close()
        except FileNotFoundError:
            print("File is not found.")
            return False
        except PermissionError:
            print("You don't have permission to access this file.")
            return False

        dataset = tf.data.TFRecordDataset(src_tfrecordfiles)
        dataset = dataset.map(self._parse_fuction)

        return dataset

class cnn_2layers:
    
    def __init__(self,
                #train,#train dataset
                #test,#test dataset
                #batch,height,width,channels
                ):#data tensor
        pass
        # self.train_data=train
        # self.test_data=test
        # self.height=height
        # self.width=width
        # self.batch=batch
        # self.channels=channels
    
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape,stddev=0.1,dtype=tf.float64)#create a tensor,size is shape;value stddev
        return tf.Variable(initial)
    
    def bias_variable(self,shape):
        initial = tf.constant(0.1,shape=shape,dtype=tf.float64)#create a constant value tensor
        return tf.Variable(initial)

    def conv2d(self,x,w):#input format(batch,height,width,channels)
        return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME',data_format="NHWC")

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

    def model(self,dataset,batch,epochs,height,width,channels,category):
        # if os.path.exists("./dataset"):#read parameter from file
        #     pass
        # else:
        #     pass

        #dropout
        keep_prob = tf.placeholder(tf.float64)
        conv_prob = tf.placeholder(tf.float64)

        data = tf.placeholder(tf.float64,[None,height,width,channels])
        label = tf.placeholder(tf.float64,[None,category])

        weight_conv1 = self.weight_variable([5,5,1,16])
        bias_conv1 = self.bias_variable([16])

        weight_conv2 = self.weight_variable([5,5,16,32])
        bias_conv2 = self.bias_variable([32])
        

        #convolution layer 1
        result_conv1 = tf.nn.relu(self.conv2d(data,weight_conv1)+bias_conv1)
        result_pool1 = self.max_pool_2x2(result_conv1)
        result_pool1 = tf.nn.dropout(result_pool1,conv_prob)
        #convolution layer 2
        result_conv2 = self.conv2d(result_pool1,weight_conv2)+bias_conv2
        result_pool2 = self.max_pool_2x2(tf.nn.relu(result_conv2))
        result_pool2 = tf.nn.dropout(result_pool2,conv_prob)
        #fully connected layer
        
        #calculate the input tensor size
        height_pool2=tf.cast(result_pool2.shape[1],tf.int64)
        width_pool2 = tf.cast(result_pool2.shape[2],tf.int64)
        print(height_pool2*width_pool2*32)

        input('continue')

        weight_fc1 = self.weight_variable([height_pool2*width_pool2*32,100])

        

        bias_fc1 = self.bias_variable([100])
        print(result_pool2.shape)
        result_pool2_flat = tf.reshape(result_pool2,[-1,height_pool2*width_pool2*32])
        input('continue')

        result_fc1 = tf.nn.relu(tf.matmul(result_pool2_flat,weight_fc1)+bias_fc1)
        result_fc1_drop  = tf.nn.dropout(result_fc1,keep_prob)

        weight_fc2 = self.weight_variable([100,category])
        bias_fc2 = self.bias_variable([7])
        prediction = tf.nn.softmax(tf.matmul(result_fc1_drop,weight_fc2)+bias_fc2)
        #loss
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=label,labels=prediction))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(label,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float64))

        #accuracy = tf.metrics.accuracy(labels = label,predictions=data)

        #dataset = dataset.shuffle(buffer_size=3000)
        train = dataset.repeat(epochs)
        train = train.batch(batch)

        #test = dataset.shuffle(buffer_size=1000)
        
        iterator = train.make_one_shot_iterator()
        batch_data,batch_label = iterator.get_next()

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            batch_data,batch_label = sess.run([batch_data,batch_label])
            #sess.run(tf.local_variables_initializer())
            for i in range(1000):
                if i%20==0:
                    train_accuracy = accuracy.eval(feed_dict={data:batch_data,label:batch_label,keep_prob:1.0,conv_prob:1.0})
                    print("step %d,accuracy:%g"%(i,train_accuracy))
                    input("continue")
                else:
                    sess.run(train_step,feed_dict={data:batch_data,label:batch_label,keep_prob:0.5,conv_prob:0.8})


tfrecords_path = './data.tfrecords'
reader = tfrecords_read(8,1433)
dataset = reader.load_tfrecords(tfrecords_path)
cnn = cnn_2layers()
cnn.model(dataset,50,1000,8,1433+8,1,7)