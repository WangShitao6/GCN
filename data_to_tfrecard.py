#in this moudle,acceptive_field will be builded to tfrecard.

import tensorflow as tf
import numpy as np

train = dict()

for i in range(7):
    for j in range(10):
        train[i] = np.zeros((4,4))


def _int64_feature(value):  
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def save_tfrecords(data,label,desfile):
    with tf.python_io.TFRecordWriter(desfile) as writer:
        for i in range(len(data)):
            feature = tf.train.Feature(
                features = {
                    "data":_bytes_feature(value=[data.tobytes()]),
                    "label":_int64_feature(value=[label])
                }
            )
            example = tf.train.Example(features=features)
            writer.writer(example.SerializeToString())

desfile = "./"

save_tfrecords(train.values(),train.keys(),desfile)