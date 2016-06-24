import time
import numpy as np
import gc
import sys, os

import tensorflow as tf
from tensorflow.contrib import immediate
from tensorflow.python.client import timeline

def test_numpy(N, iters):
  arr = np.ones((N,), dtype=dtype)
  times = []
  for i in range(iters):
    start_time = time.time()
    arr*arr
    end_time = time.time()
    times.append(end_time-start_time)

  return np.asarray(times)*10**6

def test_tf(N, iters):
  import tensorflow as tf
  tf.reset_default_graph()
  arr = tf.Variable(tf.ones_initializer(N), dtype=dtype)
  result = tf.mul(arr, arr)
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

  arr = tf.ones(N, dtype=dtype)
  arr_handle_op = tf.get_session_handle(arr)
  sess = env.session

  arr_handle = sess.run(arr_handle_op)
  holder1, dynamic_arr1 = tf.get_session_tensor(arr_handle.handle, dtype=dtype)
  holder2, dynamic_arr2 = tf.get_session_tensor(arr_handle.handle, dtype=dtype)
  result = tf.get_session_handle(tf.mul(dynamic_arr1, dynamic_arr2))
  
  run_metadata = tf.RunMetadata()

  times = []
  for i in range(iters):
    start_time = time.time()
    # collect metadata from last step

    feeds = {holder1: arr_handle.handle, holder2: arr_handle.handle}
    if i == iters-1:
      sess.run(result, feed_dict=feeds,
               options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
               run_metadata=run_metadata)
    else:
      sess.run(result, feed_dict=feeds)
    
    end_time = time.time()
    times.append(end_time-start_time)

  g = tf.get_default_graph()
  open("tf_persistent_timeline.pbtxt","w").write(str(run_metadata.step_stats))

  trace = timeline.Timeline(step_stats=run_metadata.step_stats)
  trace_file = open('tf_persistent.ctf', 'w')
  trace_file.write(trace.generate_chrome_trace_format())
  trace_file.close()

  open("tf_persistent.pbtxt", "w").write(str(g.as_graph_def()))

  return np.asarray(times)*10**6

def test_tf_env(N, iters):
  import tensorflow as tf
  with env.graph.as_default():
    tf_arr = tf.ones(N, dtype=dtype)
  arr = env.tensor_to_itensor(tf_arr)
  #arr = env.tf.ones(N, dtype=dtype)

  env.tf.mul(arr, arr)

  times = []
  for i in range(iters):
    start_time = time.time()
    if i == iters-1:
      env.enable_tracing()
      
    env.tf.mul(arr, arr)
    end_time = time.time()
    times.append(end_time-start_time)

  open("tf_env_timeline.pbtxt","w").write(str(env._run_metadata.step_stats))
  trace = timeline.Timeline(step_stats=env._run_metadata.step_stats)
  trace_file = open('tf_env.ctf', 'w')
  trace_file.write(trace.generate_chrome_trace_format())
  trace_file.close()

  open("tf_env.pbtxt", "w").write(str(env.graph.as_graph_def()))
  return np.asarray(times)*10**6

if __name__=='__main__':
  # turn off Python garbage collector to not mess with times
  gc.disable()
  dtype = np.float32
  np.set_printoptions(precision=6)

  config = tf.ConfigProto()
  config.intra_op_parallelism_threads=1
  config.inter_op_parallelism_threads=1
  config.allow_soft_placement = True
  config

  env = immediate.Env(tf, config=config)
  controller = env.graph.as_default()
  controller.__enter__()

  benchmark_type = sys.argv[1]
  if benchmark_type == 'np':
    print np.min(test_numpy(N=10**5, iters=10000))
  elif benchmark_type == 'tf':
    print np.min(test_tf(N=10**5, iters=10000))
  elif benchmark_type == 'tf_persistent':
    print np.min(test_tf_persistent(N=10**5, iters=1000))
  elif benchmark_type == 'tf_env':
    result = test_tf_env(N=10**5, iters=1000)
    print np.min(result)
  else:
    print 'unknown benchmark', benchmark_type
