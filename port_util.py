import numpy as np
import os
from numpy.testing.utils import nulp_diff

class Step(object):
    pass

def tryload(fname):
    if os.path.exists(fname):
        return np.loadtxt(fname, dtype=np.float64)
    else:
        return None
    
def loadStep(i):
  step = Step()
  step.al = tryload("stepdata/al-"+str(i))
  step.q = tryload("stepdata/q-"+str(i))
  step.ro = tryload("stepdata/ro-"+str(i))
  step.g = tryload("stepdata/g-"+str(i))
  step.x = tryload("stepdata/x-"+str(i))
  step.f = tryload("stepdata/f-"+str(i))
  step.Hdiag = tryload("stepdata/Hdiag-"+str(i))
  step.d = tryload("stepdata/d-"+str(i))
  step.be_i = tryload("stepdata/be_i-"+str(i))
  step.r = tryload("stepdata/r-"+str(i))
  step.y = tryload("stepdata/y-"+str(i))
  step.s = tryload("stepdata/s-"+str(i))
  step.tmp1 = tryload("stepdata/tmp1-"+str(i))
  step.tmp11 = tryload("stepdata/tmp11-"+str(i))
  step.ys = tryload("stepdata/ys-"+str(i))
  step.gtd = tryload("stepdata/gtd-"+str(i))
  step.t = tryload("stepdata/t-"+str(i))
  step.fdiff = tryload("stepdata/fdiff-"+str(i))
  return step


def ulp_diff(a, b):
  """Return max difference in ulps between a and b, in float32"""
  try:
    return max(nulp_diff(a.astype(np.float32), b.astype(np.float32)))
  except:
    return nulp_diff(a.astype(np.float32), b.astype(np.float32))
    
