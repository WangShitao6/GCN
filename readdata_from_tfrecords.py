import tensorflow as tf

def _parse_function(example):
    features = {"data":tf.FixedLenFeature((),tf.string),
                "label":tf.FixedLenFeature((),tf.int64)}

    parse_features = tf.parse_single_example(example,features)
    data = tf.decode_raw(parse_features["data"],tf.float64)
    return data,parse_features["label"]


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
            data,label = sess.run(next_data)
            print(data)
            #print(data.reshape((4,4)))
            print(label)
        except tf.errors.OutOfRangeError:
            break

srcfile = './data.tfrecords'

load_tfrecords(srcfile)