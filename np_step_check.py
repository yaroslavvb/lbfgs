# Reproducing lbfgs in numpy from mnist_lbfgs.lua run

import numpy as np
import port_util

def check(a, b, msg):
  """Check that two values are close to each other"""
  diff = port_util.ulp_diff(np.asarray(a), np.asarray(b))
  assert diff == 0
  if diff>0:
    print("%s %f"%(msg, diff))
  else:
    print("%s OK"%(msg))

steps = [port_util.loadStep(i) for i in range(0, 3)]
assert steps[0].f == 5127.302585093

old_dirs = []
old_stps = []
nCorrection = 100
ro = np.zeros((nCorrection,), dtype=np.float64)
al = np.zeros((nCorrection,), dtype=np.float64)

# step 0
learningRate = 1.
x = steps[0].x
g = steps[0].g
f = steps[0].f

# step 1
ii=1
d = -g
check(steps[ii].d, d, "d")

f_old = f
g_old = g

gtd = np.dot(g, d)
check(steps[ii].gtd, gtd, "gtd")

tmp1 = np.abs(np.array(g))
t = min(1, 1/tmp1.sum())
check(steps[ii].t, t, "t")

f = steps[ii].f
x = x + t*d
check(steps[ii].x, x, "x")
g = steps[ii].g

tmp1 = np.abs(g)
check(steps[ii].tmp1, tmp1, "tmp1")

tmp11 = np.abs(d*t)
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

ys = np.dot(y, s)
check(steps[ii].ys, ys, "ys")

Hdiag = ys/np.dot(y, y)
check(steps[ii].Hdiag, Hdiag, "Hdiag")


k = len(old_dirs)
for i in range(k):
    ro[i] = 1/old_stps[i].dot(old_dirs[i])
check(steps[ii].ro, ro, "ro")

q = -g
for i in range(k-1, -1, -1):
    al[i] = old_dirs[i].dot(q) * ro[i]
    q = q - al[i]*old_stps[i]
    
check(steps[ii].q, q, "q")
check(steps[ii].al, al, "al")

r = q*Hdiag
for i in range(k):
    be_i = old_stps[i].dot(r) * ro[i]
    r += (al[i]-be_i)*old_dirs[i]
d = r
    
check(steps[ii].d, d, "d")
check(steps[ii].r, r, "r")

g_old = g
f_old = f

gtd = np.dot(g, d)
check(steps[ii].gtd, gtd, "gtd")

t = learningRate
check(steps[ii].t, t, "t")

x = x + t*d
check(steps[ii].x, x, "x")
f = steps[ii].f
g = steps[ii].g

tmp1 = np.abs(g)
check(steps[ii].tmp1, tmp1, "tmp1")

check(steps[ii].fdiff, f - f_old, "fdiff")
