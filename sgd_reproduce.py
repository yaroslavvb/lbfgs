"""Script to run TensorFlow SGD on first 100 examples of MNIST and compare
results with train_on_mnist.lua Linear version.

Should see something like this
Gradient errors max 0.00137255739256 sequence 
 [  0.0000e+00   5.9753e-06   1.3726e-03   1.3726e-03   2.5356e-04
   2.5357e-04   1.5147e-05   5.1441e-04   1.0183e-05   6.0016e-05]
Parameter errors max 0.00137255739256 sequence 
 [  1.1960e-09   1.3862e-05   1.3726e-03   8.1670e-06   2.5356e-04
   1.2473e-05   6.7295e-06   6.7279e-06   5.9188e-06   6.0016e-05]
Fval errors max 1.5897629802e-08 sequence 
 [  1.0643e-15   8.4862e-09   1.4943e-08   1.3644e-08   1.4104e-08
   1.5898e-08   1.5074e-08   1.4197e-08   1.4909e-08   1.1135e-08]
"""

import sys, os

import numpy as np
import tensorflow as tf


dtype = tf.float64
batchSize = 100
learningRate = 0.1
train_data = np.load("mnist.t7/train_32x32.npy")
train_data_flat = train_data.reshape((-1, 1024))
train_labels = np.load("mnist.t7/train_labels.npy")

# use mnist_save.lua to convert existing train/test set to CSV, then use
# function below to convert to numpy array
def convert_lua_mnist_to_numpy():
  mnist_test_prefix = "mnist.t7/test_32x32.t7.csv"
  mnist_train_prefix = "mnist.t7/train_32x32.t7.csv"
  data = np.loadtxt(mnist_csv_prefix+".data",
                    dtype=np.uint8).reshape(10000, 32, 32)
  np.save("mnist.t7/test_32x32.npy", data)

  data = np.loadtxt(mnist_train_prefix+".data",
                    dtype=np.uint8).reshape(60000, 32, 32)
  np.save("mnist.t7/train_32x32.npy", data)

  data = np.loadtxt(mnist_train_prefix+".labels", dtype=np.uint8)
  np.save("mnist.t7/train_labels.npy", data)

  data = np.loadtxt(mnist_test_prefix+".labels", dtype=np.uint8)
  np.save("mnist.t7/test_labels.npy", data)
    

def read_torch_values():
  """Read results of mnist_sgd.lua"""
    
  lua_grads = []
  lua_params = []
  lua_fvals = []
  for i in range(1, 1000):
    grad_fname = "stepdata.sgd/grad-"+str(i)
    param_fname = "stepdata.sgd/params-"+str(i)
    fval_fname = "stepdata.sgd/fval-"+str(i)
    if not os.path.exists(grad_fname):
        break
    lua_grads.append(np.loadtxt(grad_fname))
    lua_params.append(np.loadtxt(param_fname))
    lua_fvals.append(np.loadtxt(fval_fname))
  return lua_grads, lua_params, lua_fvals

def concat_flatten(tensors):
    """Flattens tensors, concats them"""
    flat_tensors = []
    for tensor in tensors:
        flat_tensors.append(tf.reshape(tensor, [-1]))
    return tf.concat(0, flat_tensors)

def compute_tf_values(num_steps=10):
  """Run tensorflow SGD for a few steps, return tuple of gradient history,
  parameter history and loss value history."""
  
  W = tf.Variable(tf.ones_initializer((1024, 10), dtype=dtype))
  b = tf.Variable(tf.ones_initializer((1, 10), dtype=dtype))
  x = tf.Variable(tf.zeros_initializer((batchSize, 1024), dtype=dtype))
  targets = tf.Variable(tf.zeros_initializer((batchSize, 10), dtype=dtype))
  logits = tf.matmul(x, W) + b

  # cross entropy expects batch dimension to be first, transpose inputs
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, targets)
  cross_entropy_loss = tf.reduce_mean(cross_entropy)
  Wnorm = tf.reduce_sum(tf.square(W))
  bnorm = tf.reduce_sum(tf.square(b))
  loss = cross_entropy_loss + (bnorm + Wnorm)/2

  opt = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
  grads_and_vars = opt.compute_gradients(loss, [W, b])
  train_step = opt.apply_gradients(grads_and_vars)

  # to match Lua gradient, must transpose W
  W_grad = grads_and_vars[0][0]
  b_grad = grads_and_vars[1][0]
  flat_grad = concat_flatten([tf.transpose(W_grad), b_grad])
  flat_params = concat_flatten([tf.transpose(W), b])

  # initialize x and targets
  x_placeholder = tf.placeholder(dtype=dtype)
  x_init = x.assign(x_placeholder)
  
  # initialize labels, sub 1 since Lua starts at 1
  labels_placeholder = tf.placeholder(shape=(batchSize), dtype=tf.int32)
  labels_onehot = tf.one_hot(labels_placeholder - 1, 10, dtype=dtype) 
  targets_init = targets.assign(labels_onehot)

  sess = tf.InteractiveSession()
  sess.run(x_init, feed_dict={x_placeholder:train_data_flat[:batchSize]})
  sess.run(targets_init, feed_dict={labels_placeholder:
                                    train_labels[:batchSize]})
  sess.run([W.initializer, b.initializer])

  tf_grads = []
  tf_params = []
  tf_fvals = []
  for i in range(num_steps):
    tf_grads.append(sess.run(flat_grad))
    tf_params.append(sess.run(flat_params))
    tf_fvals.append(sess.run(loss))
    sess.run(train_step)

  return tf_grads, tf_params, tf_fvals

def compare_vals(torch_vals, tf_vals):
  pass

def max_relative_error(vec1, vec2):
  return np.max(np.abs((vec1-vec2)/vec1))

def report_error(message, values1, values2):
  errors = np.array([max_relative_error(v[0],v[1]) for v in
                     zip(values1, values2)])
  np.set_printoptions(precision=4)
  print("%s max %s sequence \n %s" %(message, str(np.max(errors)),
                                  str(errors)))

  
if __name__=="__main__":
  lua_grads,lua_params,lua_fvals = read_torch_values()
  tf_grads, tf_params, tf_fvals = compute_tf_values(len(lua_fvals))
  tf_fvals_np = np.array(tf_fvals)
  lua_fvals_np = np.array(lua_fvals)

  report_error("Gradient errors", lua_params, tf_params)
  report_error("Parameter errors", lua_grads, tf_grads)
  report_error("Fval errors", lua_fvals, tf_fvals)
  assert max_relative_error(lua_grads[-1], tf_grads[-1]) < 7e-05
