import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import immediate
import gc

def test_numpy(N, iters):
  arr = np.ones((N,), dtype=dtype)
  times = []
  for i in range(iters):
    start_time = time.time()
    arr.sum()
    end_time = time.time()
    times.append(end_time-start_time)

  return np.asarray(times)

def test_tf(N, iters):
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
  
def test_tf_persistent(N, iters):
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

def test_tf_env(N, iters):
  env = immediate.Env(tf)
  env.disable_gc()
  arr = env.tf.ones((N,), dtype=dtype)
  times = []
  for i in range(iters):
    start_time = time.time()
    env.tf.reduce_sum(arr)
    end_time = time.time()
    times.append(end_time-start_time)
    
  return np.asarray(times)

  
if __name__=='__main__':
  # turn off Python garbage collector to not mess with times
  gc.disable()
  dtype = np.float32
  np.set_printoptions(precision=6)

  print np.min(test_numpy(N=10**5, iters=5000))
#  print np.min(test_tf(N=10**5, iters=5000))
#  print np.min(test_tf_persistent(N=10**5, iters=5000))
#  print np.min(test_tf_env(N=10**5, iters=5000))
