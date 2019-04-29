import tensorflow as tf
import numpy as np

def _parse_function(example):
    features = {"node":tf.FixedLenFeature((),tf.string),
                "edge":tf.FixedLenFeature((),tf.string),
                "label":tf.FixedLenFeature((),tf.string)}

    parse_features = tf.parse_single_example(example,features)
    node = tf.decode_raw(parse_features["node"],tf.float64)
    #This can create a class member variable denote the size
    node = tf.reshape(node,(12,1433))

    edge = tf.decode_raw(parse_features["edge"],tf.float64)
    edge = tf.reshape(edge,(12,12))

    label = tf.decode_raw(parse_features['label'],tf.float64)
    label = tf.reshape(label,(1,7))
    return node,edge,label


def load_tfrecords(srcfile):
    sess = tf.Session()
    
    dataset = tf.data.TFRecordDataset(srcfile)
    dataset = dataset.map(_parse_function)
    #dataset = dataset.repeat(2)
    #dataset = dataset.batch(5)
    
    return dataset