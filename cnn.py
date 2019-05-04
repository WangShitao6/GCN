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
    
    def __init__(self):
        pass
    
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

    def model(self,dataset,model_path,batch,epochs,height,width,channels,category):

        keep_prob = tf.placeholder(tf.float64)
        conv_prob = tf.placeholder(tf.float64)

        data = tf.placeholder(tf.float64,[None,height,width,channels])
        label = tf.placeholder(tf.float64,[None,category])

        weight_conv1 = self.weight_variable([5,5,1,8])
        bias_conv1 = self.bias_variable([8])

        weight_conv2 = self.weight_variable([5,5,8,16])
        bias_conv2 = self.bias_variable([16])
        

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
        print(height_pool2*width_pool2*16)

        #input('continue')

        weight_fc1 = self.weight_variable([height_pool2*width_pool2*16,100])

        

        bias_fc1 = self.bias_variable([100])
        print(result_pool2.shape)
        result_pool2_flat = tf.reshape(result_pool2,[-1,height_pool2*width_pool2*16])
        #input('continue')

        result_fc1 = tf.nn.relu(tf.matmul(result_pool2_flat,weight_fc1)+bias_fc1)
        result_fc1_drop  = tf.nn.dropout(result_fc1,keep_prob)

        weight_fc2 = self.weight_variable([100,category])
        bias_fc2 = self.bias_variable([category])
        prediction = tf.nn.softmax(tf.matmul(result_fc1_drop,weight_fc2)+bias_fc2)
        #loss
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=label,labels=prediction))
        train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(label,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float64))
        saver = tf.train.Saver([weight_conv1,bias_conv1,
                        weight_conv2,bias_conv2,
                        weight_fc1,bias_fc1,
                        weight_fc2,bias_fc2],
                        max_to_keep=1)

        train = dataset.batch(batch)
        

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            if os.path.exists(model_path) and os.listdir(model_path):
                model_file = tf.train.latest_checkpoint(model_path)
                saver.restore(sess,model_file)
            
            #sess.run(tf.local_variables_initializer())
            for i in range(epochs):
                iterator = train.make_one_shot_iterator()
                batch_data,batch_label = iterator.get_next()
                avg_accuracy=list()
                if i%2!=0:
                    print("%d epochs training!"%i)
                else:
                    print("%d epochs testing"%i)
                while True:
                    try:
                        data_1,label_1 = sess.run([batch_data,batch_label])
                        #print(label_1)
                        if i%2==0:
                            train_accuracy = accuracy.eval(feed_dict={data:data_1,label:label_1,keep_prob:1.0,conv_prob:1.0})
                            avg_accuracy.append(train_accuracy)
                            #print("step %d,accuracy:%g"%(i,train_accuracy))
                            #input("continue test")
                        else:
                            #input("continue train")
                            sess.run(train_step,feed_dict={data:data_1,label:label_1,keep_prob:0.5,conv_prob:0.8})
                            saver.save(sess,'./model/my-model',global_step=i)
                            #print(i,end=' ')
                    except tf.errors.OutOfRangeError:
                        if i%2==0:
                            print("%d epochs accuracy:%g"%(i,sum(avg_accuracy)/len(avg_accuracy)))
                        else:
                            pass
                        break


#------------------------------------------------------------



tfrecords_path = './data.tfrecords'
model_path = './model'
reader = tfrecords_read(8,1433)
dataset = reader.load_tfrecords(tfrecords_path)
cnn = cnn_2layers()
cnn.model(dataset,model_path,50,10,8,1433+8,1,7)


# dataset = dataset.repeat(10)
# dataset = dataset.batch(100)
# iterator = dataset.make_one_shot_iterator()
# data = iterator.get_next()
# i=0
# with tf.Session() as sess:
#     while True:
#         try:
#             data_1 = sess.run(data)
#             #print(data_1[1])
#             i+=1
#         except tf.errors.OutOfRangeError:
#             print(i)
#             break

