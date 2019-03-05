#!/usr/bin/python
import os
import tensorflow as tf
g_features = {}
g_features['item_emb'] = tf.FixedLenFeature(shape=[128],dtype=tf.float32)

def read_and_decode(filename,batch_size,num_epochs):
  reader = tf.TFRecordReader()
  filename_queue = tf.train.string_input_producer([filename],
      shuffle=False,num_epochs=num_epochs)

  _, serialized_example = reader.read_up_to(filename_queue, batch_size)
  #my_example_features = {'sparse': tf.SparseFeature(index_key=['index'],
  #  value_key='values',
  #  dtype=tf.int64,
  #  size=[1,100])}

  tensor = tf.parse_example(
          serialized=serialized_example,
          features=g_features)
  print tensor
  #tensor = tf.parse_example(serialized_example,my_example_features)

  #dense_tensor = tf.sparse_tensor_to_dense(tensor['sparse'])
  #tensor = tf.reshape(tensor, tf.stack([-1,128]))
  #print tensor
  #data = tf.cast(dense_tensor, tf.int32)
  data = tf.train.batch([tensor['item_emb']],
      batch_size=batch_size, capacity=batch_size*200,
      enqueue_many=True)
  return data

if __name__ == '__main__':
  data = read_and_decode("test.tfrecord",10, 1)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer()) 
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
      i = 0
      while True:
        d = sess.run([data])
        i += 1
        print "iter %d" %(i)
        #print (val.shape, l)
        print (d)
    except tf.errors.OutOfRangeError:
      print "OutOfRangeError"
    finally:
      coord.request_stop()
      coord.join(threads)
