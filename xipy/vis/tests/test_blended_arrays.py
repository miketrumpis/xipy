import numpy as np
import numpy.testing as npt
from nose.tools import assert_true, assert_equal, assert_false

from xipy.external import decotest
import xipy.vis.color_mapping as cm

# the code to test
from xipy.vis.rgba_blending import *

def test_blended_arrays_init():
    ba = BlendedArrays()
    yield assert_false, ba.over_cmap is None
    yield assert_false, ba.main_cmap is None

@decotest.parametric
def test_mapping():
    ba = BlendedArrays()
    # check to see that the LUT is an identity function
    # for this case (grayscale cmap)
    ba.main_cmap = cm.gray
    idx_arr = np.random.randint(0, high=255, size=100)
    ba._main_idx = idx_arr
    foo = np.multiply.outer(idx_arr, np.ones((3,), 'i'))
    yield npt.assert_array_equal(foo, ba.main_rgba[:,:3])

    ba.main_alpha = 0.5

    yield npt.assert_array_equal(ba.main_rgba[:,-1],
                                 np.array([255/2]*100))

    ba.main_alpha = np.arange(256)/255.

    # Now the alpha channel is also an identity function,
    # so make sure the alpha column matches the color column
    # (this seems susceptible to round-off error maybe?)
    yield npt.assert_array_equal(ba.main_rgba[:,-1],
                                 ba.main_rgba[:,-2])

@decotest.parametric
def test_blending_unblending():
    ba = BlendedArrays(over_cmap=cm.gray, over_alpha = 0.5)

    idx_arr1 = np.random.randint(0, high=255, size=100)
    idx_arr2 = np.random.randint(0, high=255, size=100)

    ba._main_idx = idx_arr1

    yield npt.assert_array_equal(ba.main_rgba, ba.blended_rgba)

    ba._over_idx = idx_arr2

    blended = (idx_arr1 + idx_arr2)/2
    foo = np.multiply.outer(blended, np.ones((3,), 'i'))
    # first test blending -- allow for +/- 1 imperfection on the blending
    yield np.abs(foo-ba.blended_rgba[:,:3]).max() <= 1

    blended = ba.blended_rgba.copy()

    ba.over_alpha = 1
    # recall the total-clobbering quirk.. 
    yield np.abs(ba.blended_rgba-ba.over_rgba).max() <= 1

    ba.over_alpha = 0
    yield npt.assert_array_equal(ba.blended_rgba, ba.main_rgba)

    ba.over_alpha = 0.5
    yield npt.assert_array_equal(ba.blended_rgba, blended)
                                 
@decotest.parametric
def test_props_update():
    ba = BlendedArrays(main_cmap=cm.jet)

    idx_arr1 = np.random.randint(0, high=255, size=100)
    idx_arr2 = np.random.randint(0, high=255, size=100)

    ba._main_idx = idx_arr1
    ba._over_idx = idx_arr2

    ba.update_over_props(cmap=cm.gray, alpha=0)

    yield (ba.over_rgba[:,-1]==0).all()
    yield (ba.over_rgba[:,0]==idx_arr2).all()

    ba.update_main_props(cmap=cm.gray, alpha=0.5)

    yield (ba.main_rgba[:,-1]==255/2).all()
    yield (ba.main_rgba[:,0]==idx_arr1).all()

    
    
    
