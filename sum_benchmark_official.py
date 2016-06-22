import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import immediate
import gc

def test_np():
  arr = np.ones((N,), dtype=dtype)
  times = []
  for i in range(iters):
    start_time = time.time()
    arr.sum()
    end_time = time.time()
    times.append(end_time-start_time)

  return np.asarray(times)

def test_tf():
  tf.reset_default_graph()
  arr = tf.Variable(tf.ones_initializer(N), dtype=dtype)
  result = tf.reduce_sum(arr)
  result_fetch = tf.group(result)
  sess = tf.Session()
  sess.run(arr.initializer)
  times = []
  for i in range(iters):
    start_time = time.time()
    sess.run(result_fetch)
    end_time = time.time()
    times.append(end_time-start_time)

  return np.asarray(times)
  
def test_tf_persistent():
  tf.reset_default_graph()
  arr = tf.ones(N, dtype=dtype)
  arr_handle_op = tf.get_session_handle(tf.identity(arr))
  sess = tf.Session()
  arr_handle = sess.run(arr_handle_op)
  holder, dynamic_arr = tf.get_session_tensor(arr_handle.handle, dtype=dtype)
  result = tf.reduce_sum(dynamic_arr)
  result_fetch = tf.group(result)

  times = []
  for i in range(iters):
    start_time = time.time()
    sess.run(result_fetch, feed_dict={holder: arr_handle.handle})
    end_time = time.time()
    times.append(end_time-start_time)

  return np.asarray(times)

  
if __name__=='__main__':
  # turn off Python garbage collector to not mess with times
  gc.disable()
  dtype = np.float32

  N = 10**8
  iters=10
  np_times = test_np()
  tf_times = test_tf()
  tf_persistent_times = test_tf_persistent()
  np.set_printoptions(precision=4)
  print "%20s %.5f"%("numpy time", min(np_times))
  print "%20s %.5f"%("tf time", min(tf_times))
  print "%20s %.5f"%("tf persistent", min(tf_persistent_times))
