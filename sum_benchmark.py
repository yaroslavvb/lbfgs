import time
import numpy as np
import gc
import sys, os

def test_numpy(N, iters):
  arr = np.ones((N,), dtype=dtype)
  times = []
  for i in range(iters):
    start_time = time.time()
    arr.sum()
    end_time = time.time()
    times.append(end_time-start_time)

  return np.asarray(times)*10**6

def test_tf(N, iters):
  import tensorflow as tf
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

  return np.asarray(times)*10**6
  
def test_tf_persistent(N, iters):
  import tensorflow as tf
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

  print tf.get_default_graph().as_graph_def()
  return np.asarray(times)*10**6

def test_tf_env(N, iters):
  from tensorflow.contrib import immediate
  import tensorflow as tf
  env = immediate.Env(tf)
  env.disable_gc()
  arr = env.tf.ones((N,), dtype=dtype)
  times = []
  for i in range(iters):
    start_time = time.time()
    env.tf.reduce_sum(arr)
    end_time = time.time()
    times.append(end_time-start_time)

  print env.graph.as_graph_def()
  return np.asarray(times)*10**6


if __name__=='__main__':
  # turn off Python garbage collector to not mess with times
  gc.disable()
  dtype = np.float32
  np.set_printoptions(precision=6)

  benchmark_type = sys.argv[1]
  if benchmark_type == 'np':
    print np.min(test_numpy(N=10**5, iters=5000))
  elif benchmark_type == 'tf':
    print np.min(test_tf(N=10**5, iters=5000))
  elif benchmark_type == 'tf_persistent':
    print np.min(test_tf_persistent(N=10**5, iters=5000))
  elif benchmark_type == 'tf_env':
    print np.min(test_tf_env(N=10**5, iters=5000))
  else:
    print 'unknown benchmark', benchmark_type
