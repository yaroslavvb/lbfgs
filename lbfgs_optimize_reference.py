from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Reproduce first few steps of Torch l-BFGS on MNIST
# (mnist_lbfgs.lua)

# TODO: try to use print function

import types
import numpy as np
import os, sys

import tensorflow as tf
from tensorflow.contrib import immediate

def verbose_func(s):
  print(s)
  
def dot(a, b):
  return im.reduce_sum(a*b)

def lbfgs(opfunc, x, config, state):
  maxIter = config.maxIter or 20
  maxEval = config.maxEval or maxIter*1.25
  tolFun = config.tolFun or 1e-5
  tolX = config.tolX or 1e-9
  nCorrection = config.nCorrection or 100
  lineSearch = config.lineSearch
  lineSearchOpts = config.lineSearchOptions
  learningRate = env.numpy_to_itensor(config.learningRate or 1,
                                      dtype=dtype)
  isverbose = config.verbose or False

  # verbose function
  if isverbose:
    verbose = verbose_func
  else:
    verbose = lambda x: None

    # evaluate initial f(x) and df/dx
  f, g = opfunc(x)

  f_hist = [f]
  currentFuncEval = 1
  state.funcEval = state.funcEval + 1
  p = g.shape[0]

  # check optimality of initial point
  tmp1 = im.abs(g)
  if im.reduce_sum(tmp1) <= tolFun:
    verbose("optimality condition below tolFun")
    return x, f_hist

  # optimize for a max of maxIter iterations
  nIter = 0
  while nIter < maxIter:
    print(nIter, "val", f)
    # keep track of nb of iterations
    nIter = nIter + 1
    state.nIter = state.nIter + 1

    ############################################################
    ## compute gradient descent direction
    ############################################################
    if state.nIter == 1:
      d = -g
      old_dirs = []
      old_stps = []
      Hdiag = 1
    else:
      # do lbfgs update (update memory)
      y = g - g_old
      s = d*t
      ys = dot(y, s)
      
      if ys > 1e-10:
        # updating memory
        if len(old_dirs) == nCorrection:
          # shift history by one (limited-memory)
          del old_dirs[0]
          del old_stps[0]

        # store new direction/step
        old_dirs.append(s)
        old_stps.append(y)

        # update scale of initial Hessian approximation
        Hdiag = ys/dot(y, y)

      # compute the approximate (L-BFGS) inverse Hessian 
      # multiplied by the gradient
      k = len(old_dirs)

      # need to be accessed element-by-element, so don't re-type tensor:
      ro = [0]*nCorrection
      for i in range(k):
        ro[i] = 1/dot(old_stps[i], old_dirs[i])
        

      # iteration in L-BFGS loop collapsed to use just one buffer
      # need to be accessed element-by-element, so don't re-type tensor:
      al = [0]*nCorrection

      q = -g
      for i in range(k-1, -1, -1):
        al[i] = dot(old_dirs[i], q) * ro[i]
        q = q - al[i]*old_stps[i]

      # multiply by initial Hessian
      r = q*Hdiag
      for i in range(k):
        be_i = dot(old_stps[i], r) * ro[i]
        r += (al[i]-be_i)*old_dirs[i]
        
      d = r
      # final direction is in r/d (same object)

    g_old = g
    f_old = f
    
    ############################################################
    ## compute step length
    ############################################################
    # directional derivative
    gtd = dot(g, d)

    # check that progress can be made along that direction
    if gtd > -tolX:
      verbose("Can not make progress along direction.")
      break

    # reset initial guess for step size
    if state.nIter == 1:
      tmp1 = im.abs(g)
      t = min(1, 1/im.reduce_sum(tmp1))
    else:
      t = learningRate


    # optional line search: user function
    lsFuncEval = 0
    if lineSearch and isinstance(lineSearch) == types.FunctionType:
      # perform line search, using user function
      f,g,x,t,lsFuncEval = lineSearch(opfunc,x,t,d,f,g,gtd,lineSearchOpts)
      f_hist.append(f)
    else:
      # no line search, simply move with fixed-step
      x += t*d
      
      if nIter != maxIter:
        # re-evaluate function only if not in last iteration
        # the reason we do this: in a stochastic setting,
        # no use to re-evaluate that function here
        f, g = opfunc(x)
        
        lsFuncEval = 1
        f_hist.append(f)


    # update func eval
    currentFuncEval = currentFuncEval + lsFuncEval
    state.funcEval = state.funcEval + lsFuncEval

    ############################################################
    ## check conditions
    ############################################################
    if nIter == maxIter:
      # no use to run tests
      verbose('reached max number of iterations')
      break

    if currentFuncEval >= maxEval:
      # max nb of function evals
      verbose('max nb of function evals')
      break

    tmp1 = im.abs(g)
    if im.reduce_sum(tmp1) <=tolFun:
      # check optimality
      verbose('optimality condition below tolFun')
      break
    
    tmp1 = im.abs(d*t)
    if im.reduce_sum(tmp1) <= tolX:
      # step size below tolX
      verbose('step size below tolX')
      break

    if im.abs(f-f_old) < tolX:
      # function value changing less than tolX
      verbose('function value changing less than tolX'+str(im.abs(f-f_old)))
      break

    #    np.save("stepdata.temp.tf/fdiff-"+str(nIter), (f-f_old).as_numpy())


  # save state
  state.old_dirs = old_dirs
  state.old_stps = old_stps
  state.Hdiag = Hdiag
  state.g_old = g_old
  state.f_old = f_old
  state.t = t
  state.d = d

  return x, f_hist, currentFuncEval
      


# constuct model
def initialize_model(sess, train_data_flat, train_labels):
  """Reproduce model from train-on-mnist/mnist_lbfgs"""

  batchSize = 100
  learningRate = 0.1

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
  loss_handle_op = tf.get_session_handle(loss)

  # grads = tf.gradients(loss, [W, b])
  opt = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
  grads_and_vars = opt.compute_gradients(loss, [W, b])
  train_step = opt.apply_gradients(grads_and_vars)

  W_grad = grads_and_vars[0][0]
  b_grad = grads_and_vars[1][0]
  flat_grad = concat_flatten([tf.transpose(W_grad), b_grad])
  flat_grad_handle_op = tf.get_session_handle(flat_grad)
  flat_params = concat_flatten([tf.transpose(W), b])

  # initialize x and targets
  x_placeholder = tf.placeholder(dtype=dtype)
  x_init = x.assign(x_placeholder)

  # initialize labels
  labels_placeholder = tf.placeholder(shape=(batchSize), dtype=tf.int32)
  # Lua labels are off-by-one hence -1
  labels_onehot = tf.one_hot(labels_placeholder - 1, 10, dtype=dtype)
  targets_init = targets.assign(labels_onehot)

  sess.run(x_init, feed_dict={x_placeholder:train_data_flat[:batchSize]})
  sess.run(targets_init, feed_dict={labels_placeholder:
                                    train_labels[:batchSize]})
  sess.run([W.initializer, b.initializer])
  [(Wgrad, W), (bgrad, b)] = grads_and_vars
  return [loss, loss_handle_op, flat_params, flat_grad, flat_grad_handle_op,
          W, b, train_step]


def concat_flatten(tensors):
  """Flattens tensors, concats them into a single flat tensor."""
  flat_tensors = []
  for tensor in tensors:
    flat_tensors.append(tf.reshape(tensor, [-1]))
  return tf.concat(0, flat_tensors)


# Lua-like struct object with 0 defaults
class dummy(object):
  pass

class Struct(dummy):
  def __getattribute__(self, key):
    if key == '__dict__':
      return super(dummy, self).__getattribute__('__dict__')
    return self.__dict__.get(key, 0)

def rel_error(a, b):
  if isinstance(a, np.ndarray):
    return np.max((a-b)/b)
  return (a-b)/b

def load_lua_vals_lbfgs(fvals_only=True):
  """Load values generated by mnist_lbfgs.lua"""
  
  lua_grads = []
  lua_params = []
  lua_fvals = []
  for i in range(0, 1000):
    fval_fname = "stepdata.lbfgs/f-"+str(i)
    if not os.path.exists(fval_fname):
        break
    if not fvals_only or i == 0:
      grad_fname = "stepdata.lbfgs/g-"+str(i)
      param_fname = "stepdata.lbfgs/x-"+str(i)
      lua_grads.append(np.loadtxt(grad_fname))
      lua_params.append(np.loadtxt(param_fname))
    lua_fvals.append(np.loadtxt(fval_fname))
    
  return lua_grads, lua_params, lua_fvals

def max_relative_error(vec1, vec2):
  vec1 = np.asarray(vec1)
  vec2 = np.asarray(vec2)

  return np.max(np.abs((vec1-vec2)/vec1))

def report_error(message, values1, values2):
  errors = np.array([max_relative_error(v[0],v[1]) for v in
                     zip(values1, values2)])
  np.set_printoptions(precision=4)
  print("%s max %s sequence \n %s" %(message, str(np.max(errors)),
                                  str(errors)))


def doit():
  state = Struct()
  config = Struct()
  config.maxIter = 30
  config.verbose = True

  # load lua values from saved files
  lua_grads, lua_params, lua_fvals = load_lua_vals_lbfgs()

  # run our TensorFlow model to get same values
  train_data = np.load("mnist.t7/train_32x32.npy").reshape((-1, 1024))
  train_labels = np.load("mnist.t7/train_labels.npy")

  [loss, loss_handle_op, flat_params, flat_grad, flat_grad_handle_op, W, b,
   train_step] = initialize_model(sess, train_data, train_labels)


  def opfunc(x):
    """Evaluate model and gradient for parameters x."""
    x0 = x.as_numpy()
    lua_W_flat = x0[:-10]
    lua_W = lua_W_flat.reshape((10, 1024))
    tf_W = lua_W.T
    
    lua_b = x0[-10:]
    tf_b = lua_b.reshape((1, -1))

    loss_handle, grad_handle = sess.run([loss_handle_op, flat_grad_handle_op],
                                        feed_dict={W: tf_W, b: tf_b})
    return [env.handle_to_itensor(loss_handle),
            env.handle_to_itensor(grad_handle)]

  x0 = env.numpy_to_itensor(lua_params[0], dtype=dtype)
  #  import pdb; pdb.set_trace()
  x, f_hist, currentFuncEval = lbfgs(opfunc, x0, config, state)
  f_hist = [t.as_numpy() for t in f_hist]
  report_error("Fval errors", f_hist, lua_fvals)


# create immediate environment
env = immediate.Env(tf)
env.disable_gc()
sess = env.sess
controller = env.g.as_default()
controller.__enter__()
im = env.tf
dtype = tf.float64


if __name__=='__main__':
  try:
    doit()
  except:
    import sys, pdb, traceback
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)
