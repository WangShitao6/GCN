import tensorflow as tf
import numpy as np

def _parse_function(example):
    features = {"node":tf.FixedLenFeature((),tf.string),
                "edge":tf.FixedLenFeature((),tf.string),
                "label":tf.FixedLenFeature((),tf.int64)}

    parse_features = tf.parse_single_example(example,features)
    node = tf.decode_raw(parse_features["node"],tf.float64)
    #This can create a class member variable denote the size
    node = tf.reshape(node,(10,1433))
    edge = tf.decode_raw(parse_features["edge"],tf.float64)
    edge = tf.reshape(edge,(10,10))
    return node,edge,parse_features["label"]


def load_tfrecords(srcfile):
    sess = tf.Session()
    
    dataset = tf.data.TFRecordDataset(srcfile)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat(2)
    dataset = dataset.batch(5)
    
    iterator = dataset.make_one_shot_iterator()
    next_data = iterator.get_next()

    while True:
        try:
            node,edge,label = sess.run(next_data)
            print(node)
            #print(data.reshape((4,4)))
            print(edge)
            print(label)
        except tf.errors.OutOfRangeError:
            break

srcfile = './data.tfrecords'

load_tfrecords(srcfile)