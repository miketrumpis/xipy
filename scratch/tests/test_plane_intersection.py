from scratch import plane_intersection
import numpy as np
from nose.tools import assert_true, assert_equal, assert_false

def test_intersecting_tracks():
    t = [ np.array([ [-1, 0, 0], [1, 0, 0], [2, 0, 0] ]),
          np.array([ [-1, 0, 0], [-1, -1, 0], [-2, -1, 0] ]),
          np.array([ [1, 1, 0], [1, 1, 1], [0, 1, 1] ] ),
          np.array([ [1, 1, 0], [1, 1, 1], [-1, 1, 1] ] ) ]
    p0 = np.zeros(3)
    nm = np.array([1,0,0])
    idx = plane_intersection.intersecting_tracks(t, p0, nm)
    yield assert_true, 0 in idx, 'first track not in idx'
    yield assert_true, 2 in idx, 'third track not in idx'
    yield assert_true, 3 in idx, 'fourth track not in idx'

    
