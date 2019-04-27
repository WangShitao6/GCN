#in this moudle,acceptive_field will be builded to tfrecard.

import tensorflow as tf
import numpy as np

train = dict()

for i in range(7):
    for j in range(10):
        train[i] = np.zeros((4,4))
train[8]=np.array([[1,2,3,4],[1,2,3,4],[3,2,1,3],[2,3,4,5]])


def _int64_feature(value):  
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def save_tfrecords(data,label,desfile):
    with tf.python_io.TFRecordWriter(desfile) as writer:
        for i in range(len(data)):
            features = tf.train.Features(
                feature = {
                    "data":_bytes_feature(value=data[i].astype(np.float64).tostring()),
                    "label":_int64_feature(value=label[i])
                }
            )
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())

desfile = "./data.tfrecords"

save_tfrecords(list(train.values()),list(train.keys()),desfile)