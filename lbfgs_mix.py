from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Example of running l-BFGS in immediate mode
# mixing tf and immediate execution

import types
import time
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
  times = []
  while nIter < maxIter:
    start_time = time.time()
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


    times.append(time.time()-start_time)


  # save state
  state.old_dirs = old_dirs
  state.old_stps = old_stps
  state.Hdiag = Hdiag
  state.g_old = g_old
  state.f_old = f_old
  state.t = t
  state.d = d

  np.set_printoptions(precision=4)
  print(np.array(sorted(times)))
  
  return x, f_hist, currentFuncEval
      

def mnist_model(train_data_flat, train_labels, x0):
  """Creates a simple linear model that evaluates cross-entropy loss and
  gradient on MNIST dataset. Mirrors 'linear' model from train-on-mnist.lua

  Result is a Python callable that accepts ITensor parameter vector and returns
  ITensor loss and gradient.
  """
  
  batchSize = 60000

  # reshape flat parameter vector into W and b parameter matrices
  holder, param = tf.get_session_tensor(x0.tf_handle, x0.dtype)
  W_flat = tf.slice(param, [0], [10240])
  W = tf.reshape(W_flat, [1024, 10])
  b_flat = tf.slice(param, [10240], [10])
  b = tf.reshape(b_flat, [1, 10])

  # create model
  data = tf.Variable(tf.zeros_initializer((batchSize, 1024), dtype=dtype))
  targets = tf.Variable(tf.zeros_initializer((batchSize, 10), dtype=dtype))
  logits = tf.matmul(data, W) + b
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, targets)

  # create loss and gradient ops
  cross_entropy_loss = tf.reduce_mean(cross_entropy)
  Wnorm = tf.reduce_sum(tf.square(W))
  bnorm = tf.reduce_sum(tf.square(b))
  loss = cross_entropy_loss + (bnorm + Wnorm)/2
  [grad] = tf.gradients(loss, [param])

  # get handle ops that will be used to initialize ITensors
  loss_handle_tensor = tf.get_session_handle(loss)
  grad_handle_tensor = tf.get_session_handle(grad)

  # initialize data and targets
  data_placeholder = tf.placeholder(dtype=dtype)
  data_init = data.assign(data_placeholder)
  labels_placeholder = tf.placeholder(shape=(batchSize), dtype=tf.int32)
  labels_onehot = tf.one_hot(labels_placeholder - 1, 10, dtype=dtype)
  targets_init = targets.assign(labels_onehot)
  sess.run(data_init, feed_dict={data_placeholder:train_data_flat[:batchSize]})
  sess.run(targets_init, feed_dict={labels_placeholder:
                                    train_labels[:batchSize]})

  # Create our callable that get tensor handles
  def eval_model(x):
    loss_handle, grad_handle = sess.run([loss_handle_tensor,
                                         grad_handle_tensor],
                                        feed_dict={holder: x.tf_handle})
    return [env.handle_to_itensor(loss_handle),
            env.handle_to_itensor(grad_handle)]

  return eval_model


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

def doit():
  state = Struct()
  config = Struct()
  config.maxIter = 10
  config.verbose = True

  train_data = np.load("mnist.t7/train_32x32.npy").reshape((-1, 1024))
  train_labels = np.load("mnist.t7/train_labels.npy")

  x0 = env.tensor_to_itensor(tf.ones((10250), dtype=dtype))
  opfunc = mnist_model(train_data, train_labels, x0)
  x, f_hist, currentFuncEval = lbfgs(opfunc, x0, config, state)


if __name__=='__main__':
  # create immediate environment
  env = immediate.Env(tf)
  env.disable_gc()
  sess = env.sess
  controller = env.g.as_default()
  controller.__enter__()
  im = env.tf
  dtype = tf.float32
  
  try:
    doit()
  except:
    import sys, pdb, traceback
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)
