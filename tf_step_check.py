# Reproducing lbfgs in numpy from mnist_lbfgs.lua run

import tensorflow as tf
import numpy as np
import port_util

import tensorflow as tf
from tensorflow.contrib import immediate

env = immediate.Env(tf)
im = env.tf


def check(a, b, msg):
  """Check that two values are close to each other"""
  b = b.as_numpy()
  diff = port_util.ulp_diff(np.asarray(a), np.asarray(b))
  assert diff == 0
  if diff>0:
    print("%s %f"%(msg, diff))
  else:
    print("%s OK"%(msg))

steps = [port_util.loadStep(i) for i in range(0, 3)]
assert steps[0].f == 5127.302585093

def dot(a, b):
    return im.reduce_sum(a*b)


old_dirs = []
old_stps = []
nCorrection = 100
#ro = im.zeros((nCorrection,), dtype=im.float64)
ro = [0]*nCorrection
al = [0]*nCorrection
#al = im.zeros((nCorrection,), dtype=im.float64)


# step 0
learningRate = env.numpy_to_itensor(1., dtype=im.float64)
x = env.numpy_to_itensor(steps[0].x, dtype=im.float64)
g = env.numpy_to_itensor(steps[0].g, dtype=im.float64)
f = env.numpy_to_itensor(steps[0].f, dtype=im.float64)

# step 1
ii=1
d = -g
check(steps[ii].d, d, "d")

f_old = f
g_old = g

gtd = dot(g, d)
check(steps[ii].gtd, gtd, "gtd")

tmp1 = im.abs(g)
t = min(1, im.div(1, im.reduce_sum(tmp1)))
check(steps[ii].t, t, "t")

f = env.numpy_to_itensor(steps[ii].f, dtype=im.float64)
x = x + t*d
check(steps[ii].x, x, "x")
g = env.numpy_to_itensor(steps[ii].g, dtype=im.float64)

tmp1 = im.abs(g)
check(steps[ii].tmp1, tmp1, "tmp1")

tmp11 = im.abs(d*t)
check(steps[ii].tmp11, tmp11, "tmp11")
check(steps[ii].fdiff, f-f_old, "fdiff")

# step 2
ii = 2
y = g - g_old
check(steps[ii].y, y, "y")
s = d*t
check(steps[ii].s, s, "s")

old_dirs.append(s)
old_stps.append(y)

ys = dot(y, s)
check(steps[ii].ys, ys, "ys")

Hdiag = im.div(ys, dot(y, y))
check(steps[ii].Hdiag, Hdiag, "Hdiag")


k = len(old_dirs)
for i in range(k):
    ro[i] = im.div(1, dot(old_stps[i], old_dirs[i]))
    
check(steps[ii].ro, im.pack(ro), "ro")

q = -g
for i in range(k-1, -1, -1):
    al[i] = dot(old_dirs[i], q) * ro[i]
    q = q - al[i]*old_stps[i]

check(steps[ii].q, q, "q")
check(steps[ii].al, im.pack(al), "al")

r = q*Hdiag
for i in range(k):
    be_i = dot(old_stps[i], r) * ro[i]
    r += (al[i]-be_i)*old_dirs[i]
d = r

check(steps[ii].d, d, "d")
check(steps[ii].r, r, "r")

g_old = g
f_old = f

gtd = dot(g, d)
check(steps[ii].gtd, gtd, "gtd")

t = learningRate
check(steps[ii].t, t, "t")

x = x + t*d
check(steps[ii].x, x, "x")

f = env.numpy_to_itensor(steps[ii].f, dtype=im.float64)
g = env.numpy_to_itensor(steps[ii].g, dtype=im.float64)

tmp1 = im.abs(g)
check(steps[ii].tmp1, tmp1, "tmp1")

check(steps[ii].fdiff, f - f_old, "fdiff")
